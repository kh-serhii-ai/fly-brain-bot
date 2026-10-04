import OpenAI from "openai";
import { config } from "../config.js";
import { SYSTEM_PROMPT } from "../prompts.js";
import { toolSpecs } from "../tools.js";
import { NO_ANSWER, type LlmProvider, type LoopArgs, type LoopResult } from "./types.js";

type InputItem = OpenAI.Responses.ResponseInputItem;

const TOOLS: OpenAI.Responses.FunctionTool[] = toolSpecs.map((t) => ({
  type: "function",
  name: t.name,
  description: t.description,
  parameters: t.parameters,
  strict: true,
}));

/**
 * OpenAI via the Responses API (reasoning models only allow function tools there).
 *
 * Stateless: store=false, and reasoning items come back encrypted so they can be passed along
 * within one question's tool loop. Between questions the history is plain text.
 */
export function openaiProvider(client?: OpenAI, model = config.model): LlmProvider {
  const c = client ?? new OpenAI({ timeout: config.agentTimeoutMs, maxRetries: 2 });
  return {
    name: "openai",
    model,
    async check() {
      await c.models.retrieve(model);
    },
    async run({ history, question, maxIterations, execTool }: LoopArgs): Promise<LoopResult> {
      const input: InputItem[] = [
        ...history.map((t) => ({ role: t.role, content: t.text }) as InputItem),
        { role: "user", content: question },
      ];
      const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
      let iterations = 0;

      while (true) {
        iterations++;
        const res = await c.responses.create({
          model,
          instructions: SYSTEM_PROMPT,
          input,
          tools: TOOLS,
          // On the last iteration the model must answer with what it has.
          tool_choice: iterations >= maxIterations ? "none" : "auto",
          parallel_tool_calls: true,
          max_output_tokens: 16000,
          store: false,
          include: ["reasoning.encrypted_content"],
          prompt_cache_key: "fly-brain-bot",
          ...(config.effort ? { reasoning: { effort: config.effort } } : {}),
        });
        const cached = res.usage?.input_tokens_details?.cached_tokens ?? 0;
        usage.input += (res.usage?.input_tokens ?? 0) - cached;
        usage.cacheRead += cached;
        usage.output += res.usage?.output_tokens ?? 0;

        const refusal = res.output
          .flatMap((item) => (item.type === "message" ? item.content : []))
          .find((part) => part.type === "refusal");
        if (refusal || res.incomplete_details?.reason === "content_filter") {
          return { text: "", usage, iterations, refused: refusal?.refusal ?? "content_filter" };
        }

        const calls = res.output.filter(
          (item): item is OpenAI.Responses.ResponseFunctionToolCall => item.type === "function_call",
        );
        if (calls.length === 0 || res.status === "incomplete") {
          return { text: res.output_text?.trim() || NO_ANSWER, usage, iterations };
        }

        // Append the whole output (reasoning + calls) unchanged, then one output per call.
        input.push(...(res.output as InputItem[]));
        const outputs = await Promise.all(
          calls.map(async (call): Promise<InputItem> => {
            let args: unknown;
            try {
              args = JSON.parse(call.arguments || "{}");
            } catch {
              return { type: "function_call_output", call_id: call.call_id, output: "ПОМИЛКА: аргументи — некоректний JSON, повтори виклик." };
            }
            const out = await execTool(call.name, args);
            // No is_error flag in this API, so errors are marked in the output itself.
            return {
              type: "function_call_output",
              call_id: call.call_id,
              output: out.isError ? `ПОМИЛКА: ${out.content}` : out.content,
            };
          }),
        );
        input.push(...outputs);
      }
    },
  };
}
