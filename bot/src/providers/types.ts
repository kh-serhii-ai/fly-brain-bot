import type { ToolOutcome } from "../tools.js";

/** Text-only memory of earlier turns, per chat. */
export interface Turn {
  role: "user" | "assistant";
  text: string;
}

export interface Usage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
}

export interface LoopArgs {
  history: Turn[];
  question: string;
  maxIterations: number;
  /** Runs one tool call (logging and tracing included). Never throws. */
  execTool: (name: string, input: unknown) => Promise<ToolOutcome>;
}

export interface LoopResult {
  text: string;
  usage: Usage;
  iterations: number;
  /** Set when the provider declined to answer (safety refusal / content filter). */
  refused?: string | null;
}

/** One LLM backend: runs the tool-use loop for a single question. */
export interface LlmProvider {
  readonly name: string;
  readonly model: string;
  run(args: LoopArgs): Promise<LoopResult>;
  /** Fails fast at startup with a readable error if the key or model id is wrong. */
  check(): Promise<void>;
}

export const NO_ANSWER = "Не вдалося сформулювати відповідь — спробуй переформулювати питання.";
