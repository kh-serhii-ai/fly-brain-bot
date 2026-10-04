import { z } from "zod";
import { config } from "./config.js";
import { log } from "./logger.js";
import { sim, SimServiceError } from "./simClient.js";
import type { Comparison, SimInfo } from "./verdict.js";

const TARGET_DESC =
  "Ключ готової групи (наприклад sugar_grn, lc4), назва типу клітин FlyWire (наприклад DNp01, LPLC2) " +
  "або root id нейрона. Суфікс ':left' / ':right' обирає один бік мозку (наприклад 'sugar_grn:left').";

/** Provider-neutral tool definition; providers wrap it in their own format (strict JSON schema). */
export interface ToolSpec {
  name: string;
  description: string;
  parameters: { type: "object"; properties: Record<string, unknown>; required: string[]; additionalProperties: false };
}

// A fixed, ordered list so the cached prompt prefix never changes.
export const toolSpecs: ToolSpec[] = [
  {
    name: "search_neurons",
    description:
      "Шукає готові групи і типи нейронів у анотаціях FlyWire за назвою, типом або класом. " +
      "Використовуй, коли користувач згадує нейрони, яких немає серед готових груп, щоб дізнатися точну назву типу.",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "Рядок пошуку латиницею, напр. 'giant', 'MBON', 'DNa02'." } },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    name: "list_groups",
    description: "Повертає всі готові групи нейронів з описом, кількістю нейронів і рекомендованою частотою стимуляції.",
    parameters: { type: "object", properties: {}, required: [], additionalProperties: false },
  },
  {
    name: "simulate",
    description:
      "Запускає LIF-симуляцію всього мозку мухи: обрані нейрони отримують пуассонівську стимуляцію, " +
      "решта реагує через коннектом. Повертає найактивніші нейрони, активовані групи і чи дійшов сигнал до " +
      "виходів мозку (MN9, Giant Fiber, низхідні нейрони, мотонейрони). Повертає sim_id для render_activity.",
    parameters: {
      type: "object",
      properties: {
        stimulate: { type: "array", items: { type: "string" }, description: "Що стимулювати. " + TARGET_DESC },
        silence: {
          type: "array",
          items: { type: "string" },
          description: "Що «вимкнути» (їхні синапси не передають сигнал). Порожній масив, якщо нічого. " + TARGET_DESC,
        },
        rate_hz: {
          type: ["number", "null"],
          description: "Частота стимуляції, 1–300 Гц. null = рекомендована для групи.",
        },
        duration_ms: { type: ["number", "null"], description: "Тривалість, 50–2000 мс. null = 1000." },
      },
      required: ["stimulate", "silence", "rate_hz", "duration_ms"],
      additionalProperties: false,
    },
  },
  {
    name: "find_path",
    description:
      "Шукає найсильніші ланцюжки синаптичних зʼєднань від одних нейронів до інших (до 3 шляхів). " +
      "Сила ребра = частка всіх вхідних синапсів наступного нейрона.",
    parameters: {
      type: "object",
      properties: {
        from: { type: "string", description: "Звідки. " + TARGET_DESC },
        to: { type: "string", description: "Куди. " + TARGET_DESC },
        max_hops: { type: ["integer", "null"], description: "Максимум кроків, 1–10. null = 6." },
      },
      required: ["from", "to", "max_hops"],
      additionalProperties: false,
    },
  },
  {
    name: "render_activity",
    description:
      "Надсилає користувачу ролик: ліворуч схематична муха, що виконує дію з вердикту (злітає, п'є, чистить вусики) " +
      "у момент першого спайку вихідного нейрона, праворуч — як сигнал біжить мозком (спалахи нейронів, сповільнено), " +
      "і PNG із результатом симуляції (топ нейронів + де вони в мозку). " +
      "Викликай після simulate для питань про поведінку/активність.",
    parameters: {
      type: "object",
      properties: { sim_id: { type: "string", description: "sim_id з результату simulate." } },
      required: ["sim_id"],
      additionalProperties: false,
    },
  },
  {
    name: "render_path",
    description: "Малює PNG-схему знайдених шляхів і надсилає користувачу. Ті самі параметри, що й find_path.",
    parameters: {
      type: "object",
      properties: {
        from: { type: "string", description: TARGET_DESC },
        to: { type: "string", description: TARGET_DESC },
        max_hops: { type: ["integer", "null"], description: "Максимум кроків, 1–10. null = 6." },
      },
      required: ["from", "to", "max_hops"],
      additionalProperties: false,
    },
  },
  {
    name: "compare",
    description:
      "«Зламай муху»: той самий подразник для звичайного мозку і для зміненого (вимкнені нейрони або " +
      "додатковий подразник), з однаковим випадковим шумом. Повертає вердикти поведінки для обох, що змінилось, " +
      "які нейрони замовкли, і сам надсилає користувачу картинку «до / після». Для готового перевіреного сценарію " +
      "передай scenario (add_bitter, cut_feeding, cut_escape) і порожні решту полів; для власного порівняння — " +
      "scenario = null, stimulate, а також silence та/або add_stimulate для зміненого мозку.",
    parameters: {
      type: "object",
      properties: {
        scenario: { type: ["string", "null"], description: "Ключ готового сценарію або null." },
        stimulate: { type: "array", items: { type: "string" }, description: "Подразник для обох мозків. " + TARGET_DESC },
        silence: { type: "array", items: { type: "string" }, description: "Що вимкнути у зміненому мозку." },
        add_stimulate: {
          type: "array",
          items: { type: "string" },
          description: "Що додатково стимулювати у зміненому мозку (наприклад, bitter_grn).",
        },
        rate_hz: { type: ["number", "null"], description: "Частота стимуляції, 1–300 Гц. null = рекомендована." },
      },
      required: ["scenario", "stimulate", "silence", "add_stimulate", "rate_hz"],
      additionalProperties: false,
    },
  },
];

const Schemas = {
  search_neurons: z.object({ query: z.string().min(1).max(64) }),
  list_groups: z.object({}),
  simulate: z.object({
    stimulate: z.array(z.string()).min(1).max(20),
    silence: z.array(z.string()).max(20),
    rate_hz: z.number().nullable(),
    duration_ms: z.number().nullable(),
  }),
  find_path: z.object({ from: z.string(), to: z.string(), max_hops: z.number().int().nullable() }),
  render_activity: z.object({ sim_id: z.string() }),
  render_path: z.object({ from: z.string(), to: z.string(), max_hops: z.number().int().nullable() }),
  compare: z
    .object({
      scenario: z.string().nullable(),
      stimulate: z.array(z.string()).max(20),
      silence: z.array(z.string()).max(20),
      add_stimulate: z.array(z.string()).max(20),
      rate_hz: z.number().nullable(),
    })
    .refine((i) => i.scenario || (i.stimulate.length > 0 && i.silence.length + i.add_stimulate.length > 0), {
      message: "потрібен scenario, або stimulate разом із silence чи add_stimulate",
    }),
};
type ToolName = keyof typeof Schemas;

export interface ToolContext {
  /** Images produced by render tools, sent to the user after the answer. */
  /** Media produced by render tools, sent to the user after the answer (animation = MP4). */
  images: { data: Buffer; kind: "activity" | "path" | "compare" | "animation"; caption?: string }[];
  /** Simulation verdicts, shown to the user as the "what did the fly do" block. */
  sims: SimInfo[];
  /** "Break the fly" comparisons, shown as a before/after block. */
  comparisons: Comparison[];
  /** Called before a slow tool starts, e.g. to show "запускаю нейрони…". */
  onToolStart?: (name: string) => void;
}

export interface ToolOutcome {
  content: string;
  isError: boolean;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Executes one tool call. Never throws: errors become an is_error tool_result the model can react to. */
export async function runTool(name: string, rawInput: unknown, ctx: ToolContext): Promise<ToolOutcome> {
  if (!(name in Schemas)) return { content: `Невідомий інструмент ${name}`, isError: true };
  const parsed = Schemas[name as ToolName].safeParse(rawInput);
  if (!parsed.success) return { content: `Некоректні аргументи: ${parsed.error.message}`, isError: true };
  const input = parsed.data as Record<string, unknown>;
  ctx.onToolStart?.(name);
  try {
    switch (name as ToolName) {
      case "search_neurons":
        return ok(await sim.json("GET", `/neurons/search?q=${encodeURIComponent(String(input.query))}&limit=12`));
      case "list_groups":
        return ok(await sim.json("GET", "/groups"));
      case "simulate": {
        const i = input as z.infer<typeof Schemas.simulate>;
        const res = await sim.json<Record<string, any>>("POST", "/simulate", {
          stimulate: i.stimulate,
          silence: i.silence,
          rate_hz: i.rate_hz === null ? null : clamp(i.rate_hz, 1, 300),
          duration_ms: i.duration_ms === null ? 1000 : clamp(i.duration_ms, 50, 2000),
          n_trials: 10,
          seed: 0,
          record_spikes: true, // cheap (one trial), needed for the signal animation
        });
        if (res.behavior) {
          ctx.sims.push({ behavior: res.behavior, stimulated: res.stimulated?.items ?? [], silenced: res.silenced?.items ?? [] });
        }
        return ok(res);
      }
      case "find_path": {
        const i = input as z.infer<typeof Schemas.find_path>;
        return ok(await sim.json("POST", "/path", pathBody(i)));
      }
      case "render_activity": {
        const media = await activityMedia(String(input.sim_id));
        ctx.images.push(...media);
        const animated = media.some((m) => m.kind === "animation");
        return ok({
          status: animated
            ? "Анімацію сигналу і картинку з активністю буде надіслано користувачу після твоєї відповіді."
            : "Картинку з активністю буде надіслано користувачу після твоєї відповіді.",
        });
      }
      case "render_path": {
        const i = input as z.infer<typeof Schemas.render_path>;
        const png = await sim.png("/render/path", pathBody(i));
        ctx.images.push({ data: png, kind: "path" });
        return ok({ status: "Схему шляху підготовлено, її буде надіслано користувачу після твоєї відповіді." });
      }
      case "compare": {
        const i = input as z.infer<typeof Schemas.compare>;
        const rate = i.rate_hz === null ? null : clamp(i.rate_hz, 1, 300);
        const body = i.scenario
          ? { scenario: i.scenario }
          : {
              baseline: { stimulate: i.stimulate, silence: [], rate_hz: rate },
              variant: { stimulate: [...i.stimulate, ...i.add_stimulate], silence: i.silence, rate_hz: rate },
            };
        const res = await sim.json<Comparison & Record<string, any>>("POST", "/compare", body);
        // both simulations are cached by the service, so the media cost only rendering time
        try {
          const anim = await sim.media("/render/compare_animation", body, config.animationTimeoutMs);
          ctx.images.push({
            data: anim.data,
            kind: "animation",
            caption:
              `Ліворуч — ${lowerFirst(res.labels.baseline)}, праворуч — ${lowerFirst(res.labels.variant)}. ` +
              "Подразник і випадковий шум однакові; що робити, вирішила симуляція кожного мозку.",
          });
        } catch (err) {
          log("animation_failed", { compare: i.scenario ?? "custom", error: String(err) });
        }
        ctx.images.push({ data: await sim.png("/render/compare", body), kind: "compare" });
        ctx.comparisons.push(res);
        return ok({ ...compactComparison(res), media: "Ролик із двома мухами і картинку «до / після» буде надіслано користувачу." });
      }
    }
  } catch (err) {
    if (err instanceof SimServiceError) {
      return { content: JSON.stringify({ error: err.message, status: err.status, details: err.body }), isError: true };
    }
    throw err;
  }
}

function pathBody(i: { from: string; to: string; max_hops: number | null }) {
  return { from: i.from, to: i.to, max_hops: i.max_hops === null ? 6 : clamp(i.max_hops, 1, 10), k: 3 };
}

/** What the model needs to explain a comparison; the full response is ~12 KB. */
function compactComparison(c: Record<string, any>) {
  const side = (s: Record<string, any>) => ({
    behavior: s.behavior,
    n_active: s.n_active,
    runaway: s.runaway,
    stimulated: s.stimulated,
    silenced: s.silenced,
  });
  return {
    scenario: c.scenario ? { key: c.scenario.key, title: c.scenario.title, story: c.scenario.story } : null,
    labels: c.labels,
    baseline: side(c.baseline),
    variant: side(c.variant),
    diff: c.diff,
  };
}

/** Signal animation (if it renders in time) followed by the static activity picture. */
export async function activityMedia(simId: string): Promise<ToolContext["images"]> {
  const out: ToolContext["images"] = [];
  try {
    const anim = await sim.media("/render/animation", { sim_id: simId }, config.animationTimeoutMs);
    const slowdown = anim.headers.get("x-slowdown");
    out.push({
      data: anim.data,
      kind: "animation",
      caption:
        "Праворуч — як сигнал біжить мозком" +
        (slowdown ? ` (сповільнено в ${slowdown} ${timesWord(Number(slowdown))})` : "") +
        ". Ліворуч — схематична муха: тіла в моделі немає, але що робити і коли, вирішила симуляція.",
    });
  } catch (err) {
    log("animation_failed", { sim_id: simId, error: String(err) });
  }
  out.push({ data: await sim.png("/render/activity", { sim_id: simId }), kind: "activity" });
  return out;
}

function timesWord(n: number): string {
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return "рази";
  return n % 10 === 1 && n % 100 !== 11 ? "раз" : "разів";
}

function ok(data: unknown): ToolOutcome {
  return { content: JSON.stringify(data), isError: false };
}

const lowerFirst = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);
