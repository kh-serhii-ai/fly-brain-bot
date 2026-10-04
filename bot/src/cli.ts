// Debug REPL: talk to the agent without Telegram. Images are saved to ./out/.
import { mkdirSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline/promises";
import { getProvider, runAgent } from "./agent.js";
import { config } from "./config.js";
import { Sessions } from "./sessions.js";
import { formatAnswerBlock } from "./verdict.js";

const sessions = new Sessions(config.historyMessages, Infinity);
const rl = createInterface({ input: process.stdin, output: process.stdout });
mkdirSync("out", { recursive: true });
const provider = getProvider();
await provider.check();
console.log(`Мозок мухи — CLI (${provider.name}: ${provider.model}, sim: ${config.simServiceUrl}). /reset — очистити історію, Ctrl+C — вихід.`);

let n = 0;
for (;;) {
  const q = (await rl.question("\nТи: ")).trim();
  if (!q) continue;
  if (q === "/reset") {
    sessions.reset("cli");
    continue;
  }
  const res = await runAgent(sessions.get("cli"), q, {
    onToolStart: (name) => console.log(`  ⚙ ${name}…`),
    meta: { chat_id: "cli" },
  });
  sessions.record("cli", q, res);
  const verdict = formatAnswerBlock(res.sims, res.comparisons);
  console.log(`\nБот:\n${verdict ? verdict + "\n\n" : ""}${res.text}`);
  for (const img of res.images) {
    const file = `out/${++n}-${img.kind}.${img.kind === "animation" ? "mp4" : "png"}`;
    writeFileSync(file, img.data);
    console.log(`  🖼 ${file}`);
  }
  console.log(`  (${(res.ms / 1000).toFixed(1)} с, ітерацій ${res.iterations}, токени in/out ${res.usage.input}/${res.usage.output}, cache read ${res.usage.cacheRead})`);
}
