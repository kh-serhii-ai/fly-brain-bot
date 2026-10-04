// "Guess what the fly does": a deterministic flow, no LLM. The guess is judged by code against the
// verdict that the sim-service computes from validated outputs (Giant Fiber, MN9).
import { log } from "./logger.js";
import { sim } from "./simClient.js";
import { formatVerdict, type Behavior, type SimInfo } from "./verdict.js";

export interface QuizItem {
  key: string;
  button: string;
  prompt: string;
  simulate: { stimulate: string[]; silence: string[]; rate_hz: number };
  expect: Partial<Record<"escape" | "feeding" | "grooming", boolean>>;
  explain: string;
}

export type Guess = "escape" | "freeze" | "feeding" | "grooming" | "nothing";

export const GUESSES: { key: Guess; label: string }[] = [
  { key: "escape", label: "🛫 Злетить і втече" },
  { key: "freeze", label: "🧊 Завмре" },
  { key: "feeding", label: "🍽 Почне їсти" },
  { key: "grooming", label: "🧹 Почне чиститись" },
  { key: "nothing", label: "🤷 Нічого" },
];

let cache: QuizItem[] | undefined;

export async function getQuiz(): Promise<QuizItem[]> {
  if (cache) return cache;
  try {
    cache = await sim.json<QuizItem[]>("GET", "/quiz");
  } catch (err) {
    log("quiz_unavailable", { error: String(err) });
    return [];
  }
  return cache;
}

export type Judgement = "right" | "wrong" | "unmeasurable";

/** Compares the guess with the code-computed verdict. "Freeze" has no validated output in the model. */
export function judge(guess: Guess, b: Behavior): Judgement {
  switch (guess) {
    case "freeze":
      return "unmeasurable";
    case "escape":
      return b.escape.active ? "right" : "wrong";
    case "feeding":
      return b.feeding.active ? "right" : "wrong";
    case "grooming":
      return b.grooming.active ? "right" : "wrong";
    case "nothing":
      return !b.escape.active && !b.feeding.active && !b.grooming.active ? "right" : "wrong";
  }
}

export function formatQuizResult(item: QuizItem, guess: Guess, sim: SimInfo): string {
  const label = GUESSES.find((g) => g.key === guess)?.label ?? guess;
  const verdict = judge(guess, sim.behavior);
  const head =
    verdict === "right"
      ? `Ти вгадав(-ла)! ✅ Твій варіант: ${label}`
      : verdict === "wrong"
        ? `Ні — і ось чому 👇 Твій варіант: ${label}`
        : "🧊 Завмирання модель виміряти не вміє: у ній немає такого виходу. Ось що вона показує 👆";
  return [
    formatVerdict(sim),
    head,
    item.explain,
    "Це передбачення моделі за коннектомом FlyWire, а не запис живої мухи.",
  ].join("\n\n");
}
