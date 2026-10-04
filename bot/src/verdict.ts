// "What did the fly do" block. Rendered by code from the sim-service verdict, never by the LLM.

export interface BehaviorItem {
  active: boolean;
  rate_hz: number;
  neurons: string;
  label: string;
  icon: string;
  weak?: boolean;
  stimulated_directly?: boolean;
}

export interface Behavior {
  escape: BehaviorItem;
  feeding: BehaviorItem;
  grooming: BehaviorItem;
  responding_neurons: number;
  total_neurons: number;
  runaway: boolean;
}

export interface SimInfo {
  behavior: Behavior;
  stimulated: { item: string; name?: string; side?: string }[];
  silenced: { item: string; name?: string }[];
}

const ORDER = ["escape", "feeding", "grooming"] as const;

/** Ukrainian plural: 1 нейрон, 2 нейрони, 5 нейронів. */
export function neuronsWord(n: number): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return "нейрон";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "нейрони";
  return "нейронів";
}

const num = (n: number) => n.toLocaleString("uk-UA").replace(/ /g, " ");

export function brainShare(n: number, total: number): string {
  const pct = (100 * n) / total;
  if (pct === 0) return "0 %";
  if (pct < 0.01) return "менше 0,01 %";
  return `${pct.toFixed(pct < 0.1 ? 2 : 1).replace(".", ",")} %`;
}

function line(b: BehaviorItem): string {
  if (b.stimulated_directly) return `${b.icon} ${b.label} — ТАК (ці нейрони стимулювали напряму)`;
  if (b.active) return `${b.icon} ${b.label} — ТАК`;
  if (b.weak) return `${b.icon} ${b.label} — НІ (лише слабкий сигнал, ${Math.round(b.rate_hz)} Гц)`;
  return `${b.icon} ${b.label} — НІ`;
}

/** Short description of the condition, used when several simulations are shown side by side. */
function condition(sim: SimInfo): string {
  const names = (xs: { item: string; name?: string }[]) => xs.map((x) => x.name ?? x.item).join(" + ");
  let s = names(sim.stimulated);
  if (sim.silenced.length) s += `; вимкнено: ${names(sim.silenced)}`;
  return s;
}

export function formatVerdict(sim: SimInfo, title = "🪰 Що зробила муха:"): string {
  const b = sim.behavior;
  const rows = [title, ...ORDER.map((k) => line(b[k]))];
  rows.push(
    `🧠 Задіяно ${num(b.responding_neurons)} ${neuronsWord(b.responding_neurons)} з ${num(b.total_neurons)} — це ${brainShare(b.responding_neurons, b.total_neurons)} мозку`,
  );
  if (b.runaway) rows.push("⚠️ Лавина активності: мозок «перезбуджений», тож поведінка хаотична");
  return rows.join("\n");
}

/** One block per simulation; with several, each gets a header naming its condition. */
export function formatVerdicts(sims: SimInfo[]): string {
  if (sims.length === 0) return "";
  if (sims.length === 1) return formatVerdict(sims[0]);
  return sims
    .slice(0, 3)
    .map((s, i) => formatVerdict(s, `🪰 Дослід ${i + 1} — ${condition(s)}:`))
    .join("\n\n");
}

// ---------------------------------------------------------------- "break the fly" comparisons

export interface BehaviorDiff {
  baseline: boolean;
  variant: boolean;
  rate_baseline: number;
  rate_variant: number;
  changed: boolean;
}

export interface Comparison {
  scenario: { key: string; title: string } | null;
  labels: { baseline: string; variant: string };
  baseline: { behavior: Behavior };
  variant: { behavior: Behavior };
  diff: { behavior: Record<"escape" | "feeding" | "grooming", BehaviorDiff> };
}

const SHORT = {
  escape: { icon: "🦵", name: "втеча" },
  feeding: { icon: "👅", name: "їсть" },
  grooming: { icon: "🧹", name: "чиститься" },
} as const;
const OUTPUT_NAME = { escape: "Giant Fiber", feeding: "MN9", grooming: "aDN1" } as const;
const CHANGE_TEXT = {
  escape: { lost: "муха не втекла", gained: "муха почала тікати" },
  feeding: { lost: "муха перестала їсти", gained: "муха почала їсти" },
  grooming: { lost: "муха не чистить вусики", gained: "муха почала чистити вусики" },
} as const;

function shortLine(b: Behavior): string {
  return ORDER.map((k) => `${SHORT[k].icon} ${SHORT[k].name} — ${b[k].active ? "ТАК" : "НІ"}`).join(" · ");
}

export function formatComparison(c: Comparison): string {
  const rows = [`🔧 ${c.scenario?.title ?? "Зламали муху"}`];
  rows.push(`🪰 ${c.labels.baseline}: ${shortLine(c.baseline.behavior)}`);
  rows.push(`🔧 ${c.labels.variant}: ${shortLine(c.variant.behavior)}`);
  const changes = ORDER.filter((k) => c.diff.behavior[k].changed).map((k) => {
    const d = c.diff.behavior[k];
    const what = d.baseline ? CHANGE_TEXT[k].lost : CHANGE_TEXT[k].gained;
    return `${what} (${OUTPUT_NAME[k]}: ${Math.round(d.rate_baseline)} → ${Math.round(d.rate_variant)} Гц)`;
  });
  rows.push(changes.length ? `➡️ Змінилось: ${changes.join("; ")}` : "➡️ Поведінка не змінилась");
  return rows.join("\n");
}

/** Code-rendered block for an answer: simulation verdicts, then comparisons. */
export function formatAnswerBlock(sims: SimInfo[], comparisons: Comparison[]): string {
  return [formatVerdicts(sims), ...comparisons.map(formatComparison)].filter(Boolean).join("\n\n");
}

/** Validated scenarios worth offering after an answer (keys from sim-service scenarios.yaml). */
export function scenariosFor(sims: SimInfo[], comparisons: Comparison[]): string[] {
  const done = new Set(comparisons.map((c) => c.scenario?.key).filter(Boolean));
  const keys = new Set<string>();
  for (const s of sims) {
    const stim = s.stimulated.map((i) => i.item.split(":")[0].toLowerCase());
    if (s.silenced.length) continue; // already a manipulated brain
    if (stim.includes("sugar_grn") && s.behavior.feeding.active) {
      keys.add("add_bitter");
      keys.add("cut_feeding");
    }
    if ((stim.includes("lc4") || stim.includes("lplc2")) && s.behavior.escape.active) keys.add("cut_escape");
  }
  return [...keys].filter((k) => !done.has(k));
}
