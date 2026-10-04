const env = process.env;

function num(name: string, fallback: number): number {
  const v = env[name];
  if (v === undefined || v === "") return fallback;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`${name} must be a number, got "${v}"`);
  return n;
}

type Effort = "low" | "medium" | "high" | "xhigh" | "max";

function provider(): "openai" | "anthropic" {
  const p = (env.LLM_PROVIDER ?? "openai").toLowerCase();
  if (p !== "openai" && p !== "anthropic") throw new Error(`LLM_PROVIDER must be openai or anthropic, got "${p}"`);
  return p;
}

export const config = {
  telegramToken: env.TELEGRAM_BOT_TOKEN ?? "",
  simServiceUrl: (env.SIM_SERVICE_URL ?? "http://localhost:8000").replace(/\/$/, ""),
  /** "openai" (Chat Completions) or "anthropic" (Messages API). */
  provider: provider(),
  model: env.LLM_MODEL || (provider() === "anthropic" ? "claude-opus-5-5" : "gpt-5.5"),
  /** Reasoning effort; empty = don't send (for models without reasoning, e.g. gpt-4o-mini). */
  effort: (env.LLM_EFFORT ?? "medium") as Effort | "",
  /** Anthropic only: server-side refusal fallback ("default" routes by category); "off" disables it. */
  fallbacks: env.LLM_FALLBACKS ?? "default",
  maxIterations: num("AGENT_MAX_ITERATIONS", 6),
  historyMessages: num("HISTORY_MESSAGES", 10),
  dailyLimit: num("DAILY_LIMIT", 20),
  adminIds: new Set((env.ADMIN_IDS ?? "").split(",").map((s) => s.trim()).filter(Boolean)),
  /** Private bot: only these Telegram user ids are served. Empty = open to everyone. */
  allowedUsers: new Set((env.ALLOWED_USERS ?? "").split(",").map((s) => s.trim()).filter(Boolean)),
  simTimeoutMs: num("SIM_TIMEOUT_MS", 45_000),
  /** If the signal animation is not rendered in time, the bot sends the static PNG only. */
  animationTimeoutMs: num("ANIMATION_TIMEOUT_MS", 20_000),
  agentTimeoutMs: num("AGENT_TIMEOUT_MS", 120_000),
  logFile: env.LOG_FILE ?? "",
};
