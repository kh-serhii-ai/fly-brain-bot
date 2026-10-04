import { config } from "./config.js";
import { log } from "./logger.js";
import { anthropicProvider } from "./providers/anthropic.js";
import { openaiProvider } from "./providers/openai.js";
import type { LlmProvider, Turn, Usage } from "./providers/types.js";
import { runTool, type ToolContext } from "./tools.js";

export type { LlmProvider, Turn } from "./providers/types.js";

export interface AgentResult {
  text: string;
  images: ToolContext["images"];
  sims: ToolContext["sims"];
  comparisons: ToolContext["comparisons"];
  toolTrace: string[];
  usage: Usage;
  iterations: number;
  ms: number;
}

const REFUSAL_TEXT = "Вибач, на це питання я не можу відповісти. Спитай мене щось про мозок мухи 🪰";

let defaultProvider: LlmProvider | undefined;

/** The provider selected by LLM_PROVIDER (created lazily, so tests never need API keys). */
export function getProvider(): LlmProvider {
  return (defaultProvider ??= config.provider === "anthropic" ? anthropicProvider() : openaiProvider());
}

/** Runs one user question through the provider's tool-use loop. */
export async function runAgent(
  history: Turn[],
  question: string,
  hooks: { onToolStart?: (name: string) => void; meta?: Record<string, unknown>; provider?: LlmProvider } = {},
): Promise<AgentResult> {
  const started = Date.now();
  const provider = hooks.provider ?? getProvider();
  const ctx: ToolContext = { images: [], sims: [], comparisons: [], onToolStart: hooks.onToolStart };
  const toolTrace: string[] = [];

  const res = await provider.run({
    history,
    question,
    maxIterations: config.maxIterations,
    execTool: async (name, input) => {
      const t0 = Date.now();
      const out = await runTool(name, input, ctx);
      log("tool_call", {
        ...hooks.meta,
        tool: name,
        input,
        ms: Date.now() - t0,
        is_error: out.isError,
        result_chars: out.content.length,
      });
      toolTrace.push(traceLine(name, input, out.content, out.isError));
      return out;
    },
  });

  if (res.refused) log("agent_refusal", { ...hooks.meta, reason: res.refused });
  return {
    text: res.refused ? REFUSAL_TEXT : res.text,
    images: ctx.images,
    sims: ctx.sims,
    comparisons: ctx.comparisons,
    toolTrace,
    usage: res.usage,
    iterations: res.iterations,
    ms: Date.now() - started,
  };
}

/** Short summary of a tool call, kept in text history so follow-up questions have context. */
function traceLine(name: string, input: unknown, content: string, isError: boolean): string {
  const args = JSON.stringify(input);
  if (isError) return `${name}(${args}) → помилка`;
  if (name === "simulate") {
    try {
      const r = JSON.parse(content);
      // keep the strongest rate per output so follow-up questions can compare conditions
      const reached = Object.entries(r.outputs ?? {})
        .filter(([, v]) => (v as { reached: boolean }).reached)
        .map(([k, v]) => {
          const top = (v as { top?: { rate_hz: number }[] }).top ?? [];
          return top.length ? `${k} (до ${Math.max(...top.map((n) => n.rate_hz))} Гц)` : k;
        });
      return `${name}(${args}) → sim_id=${r.sim_id}, n_active=${r.n_active}, runaway=${r.runaway}, виходи: ${reached.join(", ") || "немає"}`;
    } catch {
      /* fall through */
    }
  }
  if (name === "compare") {
    try {
      const r = JSON.parse(content);
      const changes = Object.entries(r.diff?.behavior ?? {}).map(([k, v]) => {
        const d = v as { baseline: boolean; variant: boolean; rate_baseline: number; rate_variant: number };
        return `${k}: ${d.baseline ? "так" : "ні"}→${d.variant ? "так" : "ні"} (${d.rate_baseline}→${d.rate_variant} Гц)`;
      });
      return `${name}(${args}) → ${r.labels?.baseline} vs ${r.labels?.variant}: ${changes.join(", ")}`;
    } catch {
      /* fall through */
    }
  }
  return `${name}(${args})`;
}
