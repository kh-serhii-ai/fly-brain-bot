import { config } from "./config.js";

export class SimServiceError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: unknown,
  ) {
    super(message);
  }
}

async function request(method: "GET" | "POST", path: string, body?: unknown, timeoutMs = config.simTimeoutMs): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(config.simServiceUrl + path, {
      method,
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new SimServiceError("Симуляція зайняла забагато часу — спробуй простіший запит.", 504, null);
    }
    throw new SimServiceError("Сервіс симуляції недоступний.", 503, String(err));
  }
  if (!res.ok) {
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = JSON.parse(text);
    } catch {
      /* keep text */
    }
    const msg = (parsed as { message?: string })?.message ?? `HTTP ${res.status}`;
    throw new SimServiceError(msg, res.status, parsed);
  }
  return res;
}

export const sim = {
  async json<T = unknown>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
    return (await request(method, path, body)).json() as Promise<T>;
  },
  /** Binary response (image / video) plus its headers. */
  async media(path: string, body: unknown, timeoutMs?: number): Promise<{ data: Buffer; headers: Headers }> {
    const res = await request("POST", path, body, timeoutMs);
    return { data: Buffer.from(await res.arrayBuffer()), headers: res.headers };
  },
  async png(path: string, body: unknown): Promise<Buffer> {
    return Buffer.from(await (await request("POST", path, body)).arrayBuffer());
  },
};
