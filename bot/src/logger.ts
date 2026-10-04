import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { config } from "./config.js";

if (config.logFile) {
  try {
    mkdirSync(dirname(config.logFile), { recursive: true });
  } catch {
    /* fall back to stdout only */
  }
}

/** One JSON line per event, to stdout and (optionally) a file. */
export function log(event: string, data: Record<string, unknown> = {}): void {
  const line = JSON.stringify({ ts: new Date().toISOString(), event, ...data });
  console.log(line);
  if (config.logFile) {
    try {
      appendFileSync(config.logFile, line + "\n");
    } catch {
      /* logging must never break the bot */
    }
  }
}
