import Anthropic from "@anthropic-ai/sdk";
import { config } from "../config.js";
import { SYSTEM_PROMPT } from "../prompts.js";
import { toolSpecs } from "../tools.js";
import { NO_ANSWER, type LlmProvider, type LoopArgs, type LoopResult } from "./types.js";

type MessageParam = Anthropic.Beta.BetaMessageParam;

const SYSTEM: Anthropic.Beta.BetaTextBlockParam[] = [{ type: "text", text: SYSTEM_PROMPT }];
const TOOLS: Anthropic.Beta.BetaTool[] = toolSpecs.map((t) => ({
  name: t.name,
  description: t.description,
  input_schema: t.parameters,
  strict: true,
}));

/**
 * Claude via the Anthropic Messages API.
 *
 * History between questions is plain text (no thinking/tool blocks), so trimming old turns never
 * edits a turn that a replayed thinking block depends on. Within one question it is append-only.
 */
export function anthropicProvider(client?: Anthropic, model = config.model): LlmProvider {
  const c = client ?? new Anthropic({ timeout: config.agentTimeoutMs, maxRetries: 2 });
  return {
    name: "anthropic",
    model,
    async check() {
      await c.models.retrieve(model);
    },
    async run({ history, question, maxIterations, execTool }: LoopArgs): Promise<LoopResult> {
      const messages: MessageParam[] = [
        ...history.map((t) => ({ role: t.role, content: t.text }) as MessageParam),
        { role: "user", content: question },
      ];
      const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
      let iterations = 0;

      while (true) {
        iterations++;
        const response = await c.beta.messages.create({
          model,
          max_tokens: 16000,
          system: SYSTEM,
          tools: TOOLS,
          // On the last iteration the model must answer with what it has.
          tool_choice: iterations >= maxIterations ? { type: "none" } : { type: "auto" },
          ...(config.effort ? { output_config: { effort: config.effort } } : {}),
          cache_control: { type: "ephemeral" },
          messages,
          ...(config.fallbacks !== "off"
            ? { betas: ["server-side-fallback-2026-07-01"], fallbacks: config.fallbacks as "default" }
            : {}),
        });
        usage.input += response.usage.input_tokens;
        usage.output += response.usage.output_tokens;
        usage.cacheRead += response.usage.cache_read_input_tokens ?? 0;
        usage.cacheWrite += response.usage.cache_creation_input_tokens ?? 0;

        if (response.stop_reason === "refusal") {
          return { text: "", usage, iterations, refused: response.stop_details?.category ?? "refusal" };
        }
        if (response.stop_reason === "pause_turn") {
          messages.push({ role: "assistant", content: response.content as Anthropic.Beta.BetaContentBlockParam[] });
          continue;
        }

        const toolUses = response.content.filter((b): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use");
        if (toolUses.length === 0 || response.stop_reason === "max_tokens") {
          const text = response.content
            .filter((b): b is Anthropic.Beta.BetaTextBlock => b.type === "text")
            .map((b) => b.text)
            .join("\n")
            .trim();
          return { text: text || NO_ANSWER, usage, iterations };
        }

        messages.push({ role: "assistant", content: response.content as Anthropic.Beta.BetaContentBlockParam[] });
        // Independent tool calls run concurrently; all results go back in one user message.
        const results = await Promise.all(
          toolUses.map(async (tu) => {
            const out = await execTool(tu.name, tu.input);
            return {
              type: "tool_result",
              tool_use_id: tu.id,
              content: out.content,
              is_error: out.isError || undefined,
            } satisfies Anthropic.Beta.BetaToolResultBlockParam;
          }),
        );
        messages.push({ role: "user", content: results });
      }
    },
  };
}
