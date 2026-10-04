import type { AgentResult, Turn } from "./agent.js";

/** In-memory dialogue history and daily quotas. Lost on restart, which is fine for the MVP. */
export class Sessions {
  private history = new Map<string, Turn[]>();
  private usage = new Map<string, { day: string; count: number }>();
  private busy = new Set<string>();

  constructor(
    private readonly maxMessages: number,
    private readonly dailyLimit: number,
    private readonly unlimited: Set<string> = new Set(),
  ) {}

  get(chatId: string): Turn[] {
    return this.history.get(chatId) ?? [];
  }

  record(chatId: string, question: string, result: AgentResult): void {
    const note = result.toolTrace.length ? `[інструменти цього ходу: ${result.toolTrace.join("; ")}]\n\n` : "";
    this.recordText(chatId, question, note + result.text);
  }

  /** Adds a question/answer pair produced without the agent (e.g. the quiz). */
  recordText(chatId: string, question: string, answer: string): void {
    const turns = [...this.get(chatId), { role: "user", text: question } as Turn, { role: "assistant", text: answer } as Turn];
    // keep whole question/answer pairs so the history always starts with a user turn
    const keep = Math.max(2, this.maxMessages - (this.maxMessages % 2));
    this.history.set(chatId, turns.slice(-keep));
  }

  reset(chatId: string): void {
    this.history.delete(chatId);
  }

  /** Returns remaining quota after consuming one request, or -1 if the limit is exhausted. */
  consume(userId: string, now = new Date()): number {
    if (this.unlimited.has(userId)) return Infinity;
    const day = now.toISOString().slice(0, 10);
    const u = this.usage.get(userId);
    const count = u && u.day === day ? u.count : 0;
    if (count >= this.dailyLimit) return -1;
    this.usage.set(userId, { day, count: count + 1 });
    return this.dailyLimit - count - 1;
  }

  /** Gives a request back, e.g. when it failed for reasons outside the user's control. */
  refund(userId: string): void {
    const u = this.usage.get(userId);
    if (u && u.count > 0) u.count--;
  }

  tryLock(chatId: string): boolean {
    if (this.busy.has(chatId)) return false;
    this.busy.add(chatId);
    return true;
  }

  unlock(chatId: string): void {
    this.busy.delete(chatId);
  }
}
