import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { Markup, Telegraf, type Context } from "telegraf";
import { message } from "telegraf/filters";
import { getProvider, runAgent } from "./agent.js";
import { config } from "./config.js";
import { log } from "./logger.js";
import { ABOUT_TEXT, EXAMPLES, MENU, ONLY_BUTTONS_TEXT, START_ACTIONS, START_TEXT } from "./prompts.js";
import { formatQuizResult, getQuiz, GUESSES, judge, type Guess } from "./quiz.js";
import { getScenarios, scenarioQuestion } from "./scenarios.js";
import { Sessions } from "./sessions.js";
import { sim } from "./simClient.js";
import { activityMedia, type ToolContext } from "./tools.js";
import { formatAnswerBlock, scenariosFor, type Comparison, type SimInfo } from "./verdict.js";

if (!config.telegramToken) throw new Error("TELEGRAM_BOT_TOKEN is not set");

// The bot is driven only by buttons: a persistent menu keyboard plus inline choices.
// Typed text is not interpreted; it just brings the menu back.

const bot = new Telegraf(config.telegramToken, { handlerTimeout: config.agentTimeoutMs + 30_000 });
const sessions = new Sessions(config.historyMessages, config.dailyLimit, config.adminIds);
const TG_LIMIT = 4000;
const SLOW_TOOLS = new Set(["simulate", "render_activity", "compare"]);

const menuKeyboard = Markup.keyboard([[MENU.quiz], [MENU.break, MENU.experiments], [MENU.about]])
  .resize()
  .persistent()
  .placeholder("Натискай кнопки 👇");

const inline = (rows: [string, string][][]) =>
  Markup.inlineKeyboard(rows.map((row) => row.map(([label, data]) => Markup.button.callback(label, data))));

// ---------------------------------------------------------------- menu

bot.start(async (ctx) => {
  sessions.reset(String(ctx.chat.id));
  await ctx.reply(START_TEXT, menuKeyboard);
  await ctx.reply("З чого почнемо?", inline(START_ACTIONS.map((a, i) => [[a.label, `st:${i}`]])));
});

bot.hears(MENU.quiz, (ctx) => quizMenu(ctx));
bot.hears(MENU.break, (ctx) => breakMenu(ctx));
bot.hears(MENU.experiments, (ctx) =>
  ctx.reply(
    `🔬 Досліди: на них відповідає LLM-агент, запускаючи симуляцію (до ${config.dailyLimit} на добу).`,
    inline(EXAMPLES.map((e, i) => [[e.label, `ex:${i}`]])),
  ),
);
bot.hears(MENU.about, (ctx) => ctx.reply(ABOUT_TEXT, { link_preview_options: { is_disabled: true } }));

// anything typed, sent or forwarded: only buttons are understood
bot.on(message(), (ctx) => ctx.reply(ONLY_BUTTONS_TEXT, menuKeyboard));

bot.action(/^st:(\d+)$/, async (ctx) => {
  await ctx.answerCbQuery();
  const a = START_ACTIONS[Number(ctx.match[1])];
  if (a?.quiz) await askQuiz(ctx, a.quiz);
  else if (a?.menu === "break") await breakMenu(ctx);
});

bot.action(/^ex:(\d+)$/, async (ctx) => {
  await ctx.answerCbQuery();
  const ex = EXAMPLES[Number(ctx.match[1])];
  if (!ex) return;
  await ctx.reply(`❓ ${ex.question}`);
  await askAgent(ctx, ex.question);
});

// ---------------------------------------------------------------- "break the fly"

async function breakMenu(ctx: Context): Promise<void> {
  const scenarios = await getScenarios();
  if (!scenarios.length) {
    await ctx.reply("Сценарії зараз недоступні, спробуй пізніше.");
    return;
  }
  await ctx.reply(
    "🔧 Як зламаємо муху? Той самий подразник отримають звичайний мозок і змінений 👇",
    inline(scenarios.map((s) => [[s.button, `br:${s.key}`]])),
  );
}

bot.action(/^br:([a-z_]+)$/, async (ctx) => {
  await ctx.answerCbQuery();
  const s = (await getScenarios()).find((x) => x.key === ctx.match[1]);
  if (!s) return;
  await ctx.reply(`🔧 ${s.title}`);
  await askAgent(ctx, scenarioQuestion(s));
});

// ---------------------------------------------------------------- "guess what the fly does"
// Deterministic flow without the LLM: ask, run a fixed tested simulation, judge the guess by code.

async function quizMenu(ctx: Context): Promise<void> {
  const items = await getQuiz();
  if (!items.length) {
    await ctx.reply("Гра зараз недоступна, спробуй пізніше.");
    return;
  }
  await ctx.reply("🎲 Вгадай, що зробить муха. Обери дослід:", inline(items.map((q) => [[q.button, `qz:${q.key}`]])));
}

async function askQuiz(ctx: Context, key: string): Promise<void> {
  const item = (await getQuiz()).find((q) => q.key === key);
  if (!item) return;
  const b = (g: (typeof GUESSES)[number]): [string, string] => [g.label, `gs:${key}:${g.key}`];
  await ctx.reply(
    `${item.prompt}\n\nВгадай 👇 — а потім я запущу справжню симуляцію мозку.`,
    inline([GUESSES.slice(0, 2).map(b), GUESSES.slice(2, 4).map(b), GUESSES.slice(4).map(b)]),
  );
}

bot.action("qzmenu", async (ctx) => {
  await ctx.answerCbQuery();
  await quizMenu(ctx);
});

bot.action(/^qz:([a-z_]+)$/, async (ctx) => {
  await ctx.answerCbQuery();
  await askQuiz(ctx, ctx.match[1]);
});

bot.action(/^gs:([a-z_]+):(escape|freeze|feeding|grooming|nothing)$/, async (ctx) => {
  await ctx.answerCbQuery();
  const item = (await getQuiz()).find((q) => q.key === ctx.match[1]);
  const guess = ctx.match[2] as Guess;
  if (!item) return;
  const chatId = String(ctx.chat!.id);
  if (!sessions.tryLock(chatId)) {
    await ctx.reply("Ще рахую попередній дослід — зачекай трохи ⏳");
    return;
  }
  // one answer per question: drop the guess buttons
  await ctx.editMessageReplyMarkup(undefined).catch(() => {});
  const label = GUESSES.find((g) => g.key === guess)!.label;
  await ctx.reply(`Твій варіант: ${label}\n🧠 Запускаю нейрони…`);
  const typing = setInterval(() => ctx.sendChatAction("typing").catch(() => {}), 4500);
  const t0 = Date.now();
  try {
    const res = await sim.json<Record<string, any>>("POST", "/simulate", {
      ...item.simulate,
      duration_ms: 1000,
      n_trials: 10,
      seed: 0,
      record_spikes: true,
    });
    const info: SimInfo = { behavior: res.behavior, stimulated: res.stimulated.items, silenced: res.silenced.items };
    const text = formatQuizResult(item, guess, info);
    log("quiz", { chat_id: chatId, quiz: item.key, guess, result: judge(guess, info.behavior), ms: Date.now() - t0 });
    sessions.recordText(chatId, `[гра «Вгадай»] ${item.prompt} Мій варіант: ${label}`, text);
    const media = await activityMedia(res.sim_id);
    await sendAnswer(ctx, text, media, await answerKeyboard([info], [], true));
  } catch (err) {
    log("error", { chat_id: chatId, quiz: item.key, error: describe(err) });
    await ctx.reply("Не вдалося запустити симуляцію. Спробуй ще раз трохи пізніше.").catch(() => {});
  } finally {
    clearInterval(typing);
    sessions.unlock(chatId);
  }
});

// ---------------------------------------------------------------- agent (experiments, break explanations)

async function askAgent(ctx: Context, question: string): Promise<void> {
  const chatId = String(ctx.chat!.id);
  const userId = String(ctx.from!.id);
  const meta = { chat_id: chatId, user_id: userId };
  if (!sessions.tryLock(chatId)) {
    await ctx.reply("Ще думаю над попереднім дослідом — зачекай трохи ⏳");
    return;
  }
  const left = sessions.consume(userId);
  if (left < 0) {
    sessions.unlock(chatId);
    await ctx.reply(`На сьогодні досліди з агентом вичерпано (${config.dailyLimit}). Гра «Вгадай» працює без ліміту 🎲`);
    return;
  }

  log("question", { ...meta, question, quota_left: left });
  const typing = setInterval(() => ctx.sendChatAction("typing").catch(() => {}), 4500);
  void ctx.sendChatAction("typing").catch(() => {});
  let statusSent = false;
  try {
    const res = await runAgent(sessions.get(chatId), question, {
      meta,
      onToolStart: (name) => {
        if (SLOW_TOOLS.has(name) && !statusSent) {
          statusSent = true;
          void ctx.reply("🧠 Запускаю нейрони…").catch(() => {});
        }
      },
    });
    sessions.record(chatId, question, res);
    log("answer", { ...meta, ms: res.ms, iterations: res.iterations, tools: res.toolTrace.length, images: res.images.length, tokens: res.usage });
    // The verdict block is computed by code from the simulation; the LLM only explains it.
    const verdict = formatAnswerBlock(res.sims, res.comparisons);
    const text = verdict ? `${verdict}\n\n${res.text}` : res.text;
    await sendAnswer(ctx, text, res.images, await answerKeyboard(res.sims, res.comparisons));
  } catch (err) {
    sessions.refund(userId);
    log("error", { ...meta, error: describe(err) });
    await ctx.reply(errorText(err)).catch(() => {});
  } finally {
    clearInterval(typing);
    sessions.unlock(chatId);
  }
}

// ---------------------------------------------------------------- sending

/** Validated "break the fly" buttons that fit the answer, plus "another experiment" after a quiz. */
async function answerKeyboard(sims: SimInfo[], comparisons: Comparison[], quizMore = false) {
  const offer = scenariosFor(sims, comparisons);
  const scenarios = offer.length ? (await getScenarios()).filter((s) => offer.includes(s.key)) : [];
  const rows: [string, string][][] = scenarios.map((s) => [[s.button, `br:${s.key}`]]);
  if (quizMore) rows.push([["🎲 Інший дослід", "qzmenu"]]);
  return rows.length ? inline(rows) : undefined;
}

/** Text first, then media; the keyboard goes under the last message. */
async function sendAnswer(
  ctx: Context,
  text: string,
  media: ToolContext["images"],
  keyboard: Awaited<ReturnType<typeof answerKeyboard>>,
): Promise<void> {
  const chunks = split(text, TG_LIMIT);
  for (const [i, chunk] of chunks.entries()) {
    await ctx.reply(chunk, i === chunks.length - 1 && media.length === 0 ? keyboard : undefined);
  }
  for (const [i, img] of media.entries()) {
    const extra = { caption: img.caption, ...(i === media.length - 1 ? keyboard : {}) };
    if (img.kind === "animation") await ctx.replyWithAnimation({ source: img.data, filename: "fly.mp4" }, extra);
    else await ctx.replyWithPhoto({ source: img.data }, extra);
  }
}

function errorText(err: unknown): string {
  if (err instanceof Anthropic.RateLimitError || err instanceof OpenAI.RateLimitError) {
    return "Зараз забагато запитів до мовної моделі. Спробуй за хвилину.";
  }
  if (err instanceof Anthropic.APIConnectionTimeoutError || err instanceof OpenAI.APIConnectionTimeoutError) {
    return "Відповідь зайняла забагато часу. Спробуй інший дослід.";
  }
  if (err instanceof Anthropic.APIError || err instanceof OpenAI.APIError) {
    return "Мовна модель тимчасово недоступна. Гра «Вгадай» працює і без неї 🎲";
  }
  return "Щось пішло не так. Спробуй ще раз.";
}

function describe(err: unknown): string {
  if (err instanceof Anthropic.APIError || err instanceof OpenAI.APIError) {
    return `${err.constructor.name} ${err.status}: ${err.message}`;
  }
  return err instanceof Error ? `${err.name}: ${err.message}` : String(err);
}

function split(text: string, limit: number): string[] {
  const parts: string[] = [];
  let rest = text;
  while (rest.length > limit) {
    let cut = rest.lastIndexOf("\n", limit);
    if (cut < limit / 2) cut = rest.lastIndexOf(" ", limit);
    if (cut <= 0) cut = limit;
    parts.push(rest.slice(0, cut));
    rest = rest.slice(cut).trimStart();
  }
  if (rest) parts.push(rest);
  return parts;
}

bot.catch((err) => log("telegraf_error", { error: describe(err) }));

// Commands are not used: remove the old command list from the Telegram menu.
await bot.telegram.deleteMyCommands();
// Fail fast on a wrong API key or model id instead of on the first experiment.
const provider = getProvider();
try {
  await provider.check();
} catch (err) {
  log("startup_error", { provider: provider.name, model: provider.model, error: describe(err) });
  throw new Error(`Cannot use ${provider.name} model "${provider.model}" — check the API key and LLM_MODEL.`);
}
log("bot_started", { provider: provider.name, model: provider.model, sim: config.simServiceUrl });
void bot.launch();
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
