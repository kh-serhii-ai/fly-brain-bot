import assert from "node:assert/strict";
import { test } from "node:test";
import { brainShare, formatVerdict, formatVerdicts, neuronsWord, type SimInfo } from "../src/verdict.js";

const item = (over: Partial<SimInfo["behavior"]["escape"]> = {}) => ({
  active: false, rate_hz: 0, neurons: "x", label: "Стрибок-втеча", icon: "🦵", weak: false, stimulated_directly: false, ...over,
});

function sim(over: Partial<SimInfo["behavior"]> = {}, silenced: SimInfo["silenced"] = []): SimInfo {
  return {
    behavior: {
      escape: item({ active: true, rate_hz: 85 }),
      feeding: item({ label: "Витягнула хоботок (їсть)", icon: "👅" }),
      grooming: item({ label: "Чистить вусики лапками", icon: "🧹" }),
      responding_neurons: 582,
      total_neurons: 138639,
      runaway: false,
      ...over,
    },
    stimulated: [{ item: "lplc2", name: "LPLC2 — детектори зіткнення" }],
    silenced,
  };
}

test("plural forms", () => {
  assert.deepEqual([1, 2, 4, 5, 11, 12, 21, 22, 25, 582, 111].map(neuronsWord),
    ["нейрон", "нейрони", "нейрони", "нейронів", "нейронів", "нейронів", "нейрон", "нейрони", "нейронів", "нейрони", "нейронів"]);
});

test("brain share", () => {
  assert.equal(brainShare(582, 138639), "0,4 %");
  assert.equal(brainShare(19, 138639), "0,01 %");
  assert.equal(brainShare(5, 138639), "менше 0,01 %");
  assert.equal(brainShare(25191, 138639), "18,2 %");
});

test("verdict block matches the spec layout", () => {
  assert.equal(
    formatVerdict(sim()),
    "🪰 Що зробила муха:\n🦵 Стрибок-втеча — ТАК\n👅 Витягнула хоботок (їсть) — НІ\n🧹 Чистить вусики лапками — НІ\n🧠 Задіяно 582 нейрони з 138 639 — це 0,4 % мозку",
  );
});

test("weak, direct and runaway are spelled out", () => {
  const s = sim({
    escape: item({ stimulated_directly: true, active: true }),
    feeding: item({ label: "Витягнула хоботок (їсть)", icon: "👅", weak: true, rate_hz: 20.4 }),
    runaway: true,
  });
  const text = formatVerdict(s);
  assert.match(text, /ТАК \(ці нейрони стимулювали напряму\)/);
  assert.match(text, /НІ \(лише слабкий сигнал, 20 Гц\)/);
  assert.match(text, /Лавина активності/);
});

test("several simulations get labelled blocks", () => {
  const text = formatVerdicts([sim(), sim({}, [{ item: "gaba", name: "GABA-ергічні нейрони" }])]);
  assert.match(text, /Дослід 1 — LPLC2 — детектори зіткнення:/);
  assert.match(text, /Дослід 2 — .*вимкнено: GABA-ергічні нейрони:/);
  assert.equal(formatVerdicts([]), "");
});

// ---------------------------------------------------------------- comparisons
import { formatComparison, scenariosFor, type Comparison } from "../src/verdict.js";

function cmp(feedB: boolean, feedV: boolean, rateB: number, rateV: number): Comparison {
  const beh = (feed: boolean, rate: number) => ({
    escape: item(),
    feeding: item({ label: "Витягнула хоботок (їсть)", icon: "👅", active: feed, rate_hz: rate }),
    grooming: item({ label: "Чистить вусики лапками", icon: "🧹" }),
    responding_neurons: 10,
    total_neurons: 138639,
    runaway: false,
  });
  return {
    scenario: { key: "cut_feeding", title: "Перерізати шлях до хоботка" },
    labels: { baseline: "Звичайна муха", variant: "Муха без 2 нейронів CB0553" },
    baseline: { behavior: beh(feedB, rateB) },
    variant: { behavior: beh(feedV, rateV) },
    diff: {
      behavior: {
        escape: { baseline: false, variant: false, rate_baseline: 0, rate_variant: 0, changed: false },
        feeding: { baseline: feedB, variant: feedV, rate_baseline: rateB, rate_variant: rateV, changed: feedB !== feedV },
        grooming: { baseline: false, variant: false, rate_baseline: 0, rate_variant: 0, changed: false },
      },
    },
  };
}

test("comparison block says what changed, computed from the verdicts", () => {
  assert.equal(
    formatComparison(cmp(true, false, 90.3, 1.1)),
    [
      "🔧 Перерізати шлях до хоботка",
      "🪰 Звичайна муха: 🦵 втеча — НІ · 👅 їсть — ТАК · 🧹 чиститься — НІ",
      "🔧 Муха без 2 нейронів CB0553: 🦵 втеча — НІ · 👅 їсть — НІ · 🧹 чиститься — НІ",
      "➡️ Змінилось: муха перестала їсти (MN9: 90 → 1 Гц)",
    ].join("\n"),
  );
  assert.match(formatComparison(cmp(true, true, 90, 85)), /Поведінка не змінилась/);
});

test("scenario buttons are offered only where they fit", () => {
  const sugar = sim({ escape: item(), feeding: item({ active: true, label: "x", icon: "👅" }) });
  sugar.stimulated = [{ item: "sugar_grn:left" }];
  assert.deepEqual(scenariosFor([sugar], []), ["add_bitter", "cut_feeding"]);
  assert.deepEqual(scenariosFor([sugar], [cmp(true, false, 90, 1)]), ["add_bitter"]); // already shown
  assert.deepEqual(scenariosFor([sim()], []), ["cut_escape"]); // lplc2 with escape
  const broken = sim({}, [{ item: "gaba" }]);
  assert.deepEqual(scenariosFor([broken], []), []); // already a manipulated brain
  const bitter = sim({ escape: item() });
  bitter.stimulated = [{ item: "bitter_grn" }];
  assert.deepEqual(scenariosFor([bitter], []), []);
});
