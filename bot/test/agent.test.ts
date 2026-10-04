import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import type Anthropic from "@anthropic-ai/sdk";
import type OpenAI from "openai";
import { anthropicProvider } from "../src/providers/anthropic.js";
import { openaiProvider } from "../src/providers/openai.js";
import { runAgent } from "../src/agent.js";
import { Sessions } from "../src/sessions.js";
import { runTool } from "../src/tools.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

/** Fake sim-service: records calls, answers /simulate with a tiny summary and renders with a PNG. */
function mockSim(calls: { url: string; body: unknown }[]) {
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url: String(url), body });
    if (String(url).endsWith("/simulate")) {
      return new Response(JSON.stringify({ sim_id: "abc", n_active: 361, runaway: false, outputs: { mn9: { reached: true } } }));
    }
    if (String(url).includes("/render/")) return new Response(new Uint8Array([0x89, 0x50, 0x4e, 0x47]));
    return new Response(JSON.stringify({ error: "unknown_target", message: "Не знайдено" }), { status: 404 });
  }) as typeof fetch;
}

function usage() {
  return { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 };
}

/** Fake Anthropic client that replays scripted responses and records requests. */
function fakeClient(responses: unknown[], requests: any[]): Anthropic {
  return {
    beta: {
      messages: {
        create: async (req: unknown) => {
          requests.push(structuredClone(req));
          const r = responses.shift();
          if (!r) throw new Error("no more scripted responses");
          return r;
        },
      },
    },
  } as unknown as Anthropic;
}

test("agent runs tools, collects images and returns the final text", async () => {
  const calls: { url: string; body: any }[] = [];
  mockSim(calls);
  const requests: any[] = [];
  const client = fakeClient(
    [
      {
        stop_reason: "tool_use",
        usage: usage(),
        content: [{ type: "tool_use", id: "t1", name: "simulate", input: { stimulate: ["sugar_grn:left"], silence: [], rate_hz: null, duration_ms: null } }],
      },
      {
        stop_reason: "tool_use",
        usage: usage(),
        content: [{ type: "tool_use", id: "t2", name: "render_activity", input: { sim_id: "abc" } }],
      },
      { stop_reason: "end_turn", usage: usage(), content: [{ type: "text", text: "MN9 активувався." }] },
    ],
    requests,
  );
  const started: string[] = [];
  const res = await runAgent([], "Що буде з цукром?", { provider: anthropicProvider(client, "test"), onToolStart: (n) => started.push(n) });

  assert.equal(res.text, "MN9 активувався.");
  assert.deepEqual(res.images.map((i) => i.kind), ["animation", "activity"]);
  assert.equal(calls[0].body.record_spikes, true);
  assert.equal(res.iterations, 3);
  assert.deepEqual(started, ["simulate", "render_activity"]);
  assert.equal(calls[0].body.n_trials, 10);
  assert.equal(calls[0].body.duration_ms, 1000);
  // transcript is append-only: each request extends the previous one
  assert.equal(requests[1].messages.length, 3);
  assert.deepEqual(requests[2].messages.slice(0, 3), requests[1].messages);
  assert.equal(requests[2].messages[4].content[0].tool_use_id, "t2");
  assert.match(res.toolTrace[0], /sim_id=abc.*виходи: mn9/);
});

test("sim-service errors become is_error tool results, not exceptions", async () => {
  mockSim([]);
  const out = await runTool("find_path", { from: "foo", to: "mn9", max_hops: null }, { images: [], sims: [], comparisons: [] });
  assert.equal(out.isError, true);
  assert.match(out.content, /Не знайдено/);
});

test("invalid tool input is rejected before calling the service", async () => {
  const calls: any[] = [];
  mockSim(calls);
  const out = await runTool("simulate", { stimulate: [] }, { images: [], sims: [], comparisons: [] });
  assert.equal(out.isError, true);
  assert.equal(calls.length, 0);
});

test("last iteration forbids tools so the model must answer", async () => {
  mockSim([]);
  const requests: any[] = [];
  const toolTurn = () => ({
    stop_reason: "tool_use",
    usage: usage(),
    content: [{ type: "tool_use", id: `t${Math.random()}`, name: "list_groups", input: {} }],
  });
  const client = fakeClient(
    [toolTurn(), toolTurn(), toolTurn(), toolTurn(), toolTurn(), { stop_reason: "end_turn", usage: usage(), content: [{ type: "text", text: "ok" }] }],
    requests,
  );
  const res = await runAgent([], "q", { provider: anthropicProvider(client, "test") });
  assert.equal(res.text, "ok");
  assert.equal(requests.length, 6);
  assert.deepEqual(requests[5].tool_choice, { type: "none" });
  assert.deepEqual(requests[0].tool_choice, { type: "auto" });
});

test("refusal returns a polite message", async () => {
  const client = fakeClient([{ stop_reason: "refusal", stop_details: { category: null }, usage: usage(), content: [] }], []);
  const res = await runAgent([], "q", { provider: anthropicProvider(client, "test") });
  assert.match(res.text, /не можу/);
});

test("sessions keep whole pairs and enforce the daily limit", () => {
  const s = new Sessions(4, 2);
  const r = (t: string) => ({ text: t, images: [], toolTrace: [], usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, iterations: 1, ms: 1 });
  s.record("c", "q1", r("a1"));
  s.record("c", "q2", r("a2"));
  s.record("c", "q3", r("a3"));
  const h = s.get("c");
  assert.equal(h.length, 4);
  assert.equal(h[0].role, "user");
  assert.equal(h[0].text, "q2");

  assert.equal(s.consume("u"), 1);
  assert.equal(s.consume("u"), 0);
  assert.equal(s.consume("u"), -1);
  s.refund("u");
  assert.equal(s.consume("u"), 0);
  assert.equal(s.consume("u", new Date(Date.now() + 86_400_000)), 1); // next day
});

// ---------------------------------------------------------------- OpenAI provider (Responses API)

function oaiUsage() {
  return { input_tokens: 100, output_tokens: 20, total_tokens: 120, input_tokens_details: { cached_tokens: 60 } };
}

function oaiCall(callId: string, name: string, args: unknown) {
  return { type: "function_call", id: `fc_${callId}`, call_id: callId, name, arguments: JSON.stringify(args), status: "completed" };
}

const reasoning = { type: "reasoning", id: "rs_1", summary: [], encrypted_content: "enc" };

function done(text: string) {
  return { status: "completed", usage: oaiUsage(), output_text: text, output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text }] }] };
}

function fakeOpenAI(responses: unknown[], requests: any[]): OpenAI {
  return {
    responses: {
      create: async (req: unknown) => {
        requests.push(structuredClone(req));
        const r = responses.shift();
        if (!r) throw new Error("no more scripted responses");
        return r;
      },
    },
  } as unknown as OpenAI;
}

test("openai: parallel tool calls, reasoning passed back, images and usage", async () => {
  mockSim([]);
  const requests: any[] = [];
  const client = fakeOpenAI(
    [
      {
        status: "completed",
        usage: oaiUsage(),
        output_text: "",
        output: [
          reasoning,
          oaiCall("c1", "simulate", { stimulate: ["lplc2"], silence: [], rate_hz: null, duration_ms: null }),
          oaiCall("c2", "render_path", { from: "lplc2", to: "giant_fiber", max_hops: null }),
        ],
      },
      done("Giant Fiber спрацював."),
    ],
    requests,
  );
  const res = await runAgent([{ role: "user", text: "привіт" }, { role: "assistant", text: "Вітаю!" }], "Як муха тікає?", {
    provider: openaiProvider(client, "test-model"),
  });
  assert.equal(res.text, "Giant Fiber спрацював.");
  assert.equal(res.images.length, 1);
  assert.deepEqual(res.usage, { input: 80, output: 40, cacheRead: 120, cacheWrite: 0 });
  const second = requests[1];
  assert.equal(typeof second.instructions, "string");
  assert.equal(second.store, false);
  assert.deepEqual(second.include, ["reasoning.encrypted_content"]);
  assert.equal(second.input[0].content, "привіт");
  assert.deepEqual(second.input.slice(3).map((i: any) => i.type), ["reasoning", "function_call", "function_call", "function_call_output", "function_call_output"]);
  assert.deepEqual(second.input.slice(6).map((i: any) => i.call_id), ["c1", "c2"]);
  assert.equal(requests[0].tools[0].strict, true);
  assert.equal(requests[0].tool_choice, "auto");
});

test("openai: tool errors are marked, bad JSON arguments do not crash", async () => {
  mockSim([]);
  const requests: any[] = [];
  const client = fakeOpenAI(
    [
      {
        status: "completed",
        usage: oaiUsage(),
        output_text: "",
        output: [
          oaiCall("c1", "find_path", { from: "foo", to: "mn9", max_hops: null }),
          { type: "function_call", id: "fc_2", call_id: "c2", name: "list_groups", arguments: "{oops" },
        ],
      },
      done("Не знайшов такого нейрона."),
    ],
    requests,
  );
  const res = await runAgent([], "шлях від foo", { provider: openaiProvider(client, "m") });
  assert.equal(res.text, "Не знайшов такого нейрона.");
  const outs = requests[1].input.filter((i: any) => i.type === "function_call_output");
  assert.match(outs[0].output, /^ПОМИЛКА: .*Не знайдено/);
  assert.match(outs[1].output, /^ПОМИЛКА: .*JSON/);
});

test("openai: last iteration sets tool_choice none; refusal is handled", async () => {
  mockSim([]);
  const requests: any[] = [];
  const turn = () => ({ status: "completed", usage: oaiUsage(), output_text: "", output: [oaiCall(`c${Math.random()}`, "list_groups", {})] });
  const client = fakeOpenAI([turn(), turn(), turn(), turn(), turn(), done("ok")], requests);
  assert.equal((await runAgent([], "q", { provider: openaiProvider(client, "m") })).text, "ok");
  assert.equal(requests[5].tool_choice, "none");

  const refuse = fakeOpenAI(
    [{ status: "completed", usage: oaiUsage(), output_text: "", output: [{ type: "message", role: "assistant", content: [{ type: "refusal", refusal: "no" }] }] }],
    [],
  );
  assert.match((await runAgent([], "q", { provider: openaiProvider(refuse, "m") })).text, /не можу/);
});

test("compare tool: scenario request, picture and comparison collected", async () => {
  const calls: { url: string; body: any }[] = [];
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url: String(url), body });
    if (String(url).endsWith("/compare")) {
      return new Response(JSON.stringify({
        scenario: { key: "cut_feeding", title: "t", story: "s" }, labels: { baseline: "a", variant: "b" },
        baseline: { behavior: {}, n_active: 1, top_neurons: [1, 2, 3] }, variant: { behavior: {}, n_active: 0 },
        diff: { behavior: { feeding: { baseline: true, variant: false, rate_baseline: 90, rate_variant: 1 } } },
      }));
    }
    return new Response(new Uint8Array([0x89, 0x50]));
  }) as typeof fetch;
  const ctx = { images: [] as any[], sims: [], comparisons: [] as any[] };
  const out = await runTool("compare", { scenario: "cut_feeding", stimulate: [], silence: [], add_stimulate: [], rate_hz: null }, ctx);
  assert.equal(out.isError, false);
  assert.deepEqual(calls.map((c) => c.url.split("/").pop()), ["compare", "compare_animation", "compare"]);
  assert.ok(calls.every((c) => c.body.scenario === "cut_feeding"));
  assert.deepEqual(ctx.images.map((i) => i.kind), ["animation", "compare"]);
  assert.match(ctx.images[0].caption, /^Ліворуч — a, праворуч — b/);
  assert.equal(ctx.comparisons.length, 1);
  assert.ok(!out.content.includes("top_neurons"), "compact result for the model");

  const custom = await runTool("compare", { scenario: null, stimulate: ["lc4"], silence: ["PVLP122b"], add_stimulate: [], rate_hz: 500 }, ctx);
  assert.equal(custom.isError, false);
  assert.deepEqual(calls[3].body, {
    baseline: { stimulate: ["lc4"], silence: [], rate_hz: 300 },
    variant: { stimulate: ["lc4"], silence: ["PVLP122b"], rate_hz: 300 },
  });
  const bad = await runTool("compare", { scenario: null, stimulate: ["lc4"], silence: [], add_stimulate: [], rate_hz: null }, ctx);
  assert.equal(bad.isError, true); // nothing changed between the two brains
});


test("render_activity falls back to the PNG when the animation fails", async () => {
  globalThis.fetch = (async (url: string) => {
    if (String(url).endsWith("/render/animation")) return new Response("boom", { status: 500 });
    return new Response(new Uint8Array([0x89, 0x50]));
  }) as typeof fetch;
  const ctx = { images: [] as any[], sims: [], comparisons: [] };
  const out = await runTool("render_activity", { sim_id: "x" }, ctx);
  assert.equal(out.isError, false);
  assert.deepEqual(ctx.images.map((i) => i.kind), ["activity"]);
});

test("animation caption carries the slowdown from the service", async () => {
  globalThis.fetch = (async (url: string) => {
    if (String(url).endsWith("/render/animation")) {
      return new Response(new Uint8Array([0, 0, 0, 0]), { headers: { "x-slowdown": "77" } });
    }
    return new Response(new Uint8Array([0x89, 0x50]));
  }) as typeof fetch;
  const ctx = { images: [] as any[], sims: [], comparisons: [] };
  await runTool("render_activity", { sim_id: "x" }, ctx);
  assert.match(ctx.images[0].caption!, /^Праворуч — як сигнал біжить мозком \(сповільнено в 77 разів\)\. Ліворуч — схематична муха/);
});
