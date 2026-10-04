import assert from "node:assert/strict";
import { test } from "node:test";
import { formatQuizResult, judge, type QuizItem } from "../src/quiz.js";
import type { Behavior, SimInfo } from "../src/verdict.js";

const out = (active: boolean, label: string, icon: string) => ({ active, rate_hz: active ? 80 : 0, neurons: "x", label, icon });
const behavior = (escape: boolean, feeding: boolean, grooming = false): Behavior => ({
  escape: out(escape, "Стрибок-втеча", "🦵"),
  feeding: out(feeding, "Витягнула хоботок (їсть)", "👅"),
  grooming: out(grooming, "Чистить вусики лапками", "🧹"),
  responding_neurons: 582,
  total_neurons: 138639,
  runaway: false,
});

test("judging guesses against the code verdict", () => {
  assert.equal(judge("escape", behavior(true, false)), "right");
  assert.equal(judge("feeding", behavior(true, false)), "wrong");
  assert.equal(judge("nothing", behavior(false, false)), "right");
  assert.equal(judge("nothing", behavior(false, true)), "wrong");
  assert.equal(judge("feeding", behavior(false, true)), "right");
  assert.equal(judge("freeze", behavior(true, false)), "unmeasurable");
  assert.equal(judge("grooming", behavior(false, false, true)), "right");
  assert.equal(judge("nothing", behavior(false, false, true)), "wrong");
  assert.equal(judge("grooming", behavior(true, false)), "wrong");
});

const item: QuizItem = {
  key: "swat", button: "🖐", prompt: "?", simulate: { stimulate: ["lplc2"], silence: [], rate_hz: 100 },
  expect: { escape: true, feeding: false }, explain: "Пояснення.",
};
const sim: SimInfo = { behavior: behavior(true, false), stimulated: [{ item: "lplc2" }], silenced: [] };

test("quiz result: verdict first, then the judgement and the explanation", () => {
  const right = formatQuizResult(item, "escape", sim);
  assert.match(right, /^🪰 Що зробила муха:\n🦵 Стрибок-втеча — ТАК/);
  assert.match(right, /Ти вгадав\(-ла\)! ✅ Твій варіант: 🛫 Злетить і втече/);
  assert.match(right, /Пояснення\./);
  assert.match(right, /не запис живої мухи/);
  assert.match(formatQuizResult(item, "feeding", sim), /Ні — і ось чому 👇 Твій варіант: 🍽 Почне їсти/);
  assert.match(formatQuizResult(item, "freeze", sim), /Завмирання модель виміряти не вміє/);
});
