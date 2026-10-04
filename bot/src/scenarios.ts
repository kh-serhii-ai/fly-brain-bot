import { log } from "./logger.js";
import { sim } from "./simClient.js";

export interface Scenario {
  key: string;
  title: string;
  button: string;
}

let cache: Scenario[] | undefined;

/** Validated "break the fly" scenarios from the sim-service (fetched once, then cached). */
export async function getScenarios(): Promise<Scenario[]> {
  if (cache) return cache;
  try {
    cache = await sim.json<Scenario[]>("GET", "/scenarios");
  } catch (err) {
    log("scenarios_unavailable", { error: String(err) });
    return [];
  }
  return cache;
}

/** The instruction the agent gets when a scenario button is pressed. */
export function scenarioQuestion(s: Scenario): string {
  return `Зламай муху за готовим сценарієм «${s.title}»: виклич compare зі scenario="${s.key}". Поясни простими словами, що змінилось і чому.`;
}
