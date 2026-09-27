#!/usr/bin/env bun
// A/B bench against a running head: greedy, prompt cache off, per-prompt tok/s and draft counters
// read from the server's own timings. Ported from the bench.py that produced the rows under
// heads/<head>/evidence/ — the request body, the prompt set and the warm-up are unchanged, so a
// rerun is comparable against those recorded numbers. Every row names what answered (the pack
// file, the engine build, whether it drafts: /props and /slots, read before the first request),
// so a run against the wrong port is visible in its own rows; <label>.jsonl is never overwritten.
//
//   bun scripts/bench-head.ts <port> <label> [max_tokens]

import { writeFileSync } from "node:fs";
import { CryptoHasher } from "bun";

const port = process.argv[2];
const label = process.argv[3];
const maxTokens = process.argv[4] === undefined ? 512 : Number(process.argv[4]);

if (port === undefined || label === undefined || !Number.isFinite(maxTokens)) {
  console.error("usage: bun scripts/bench-head.ts <port> <label> [max_tokens]");
  process.exit(2);
}

const PROMPTS = {
  code: "Write a Python module implementing an LRU cache with TTL expiry, thread-safe, with docstrings and a small test at the bottom.",
  sql: "Design a PostgreSQL schema for a multi-tenant invoicing system (tenants, customers, invoices, line items, payments) and write the DDL with indexes and constraints, then three analytical queries.",
  prose:
    "Write a reflective essay on why cities feel different at night, with concrete sensory detail and no lists.",
  reasoning:
    "A train leaves city A at 9:00 going 80 km/h; another leaves city B, 300 km away, at 9:30 going 100 km/h toward A. Work out exactly when and where they meet, then generalize to a formula and check it against the numbers.",
} as const;

type Name = keyof typeof PROMPTS;

const out = `${label}.jsonl`;
if (await Bun.file(out).exists()) {
  console.error(
    `${out} exists: a second run under one label would replace the first; pick a new label`,
  );
  process.exit(1);
}

async function get(path: string): Promise<unknown> {
  const res = await fetch(`http://127.0.0.1:${port}${path}`, {
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok)
    throw new Error(`head on :${port} answered ${path} with ${res.status} ${res.statusText}`);
  return res.json();
}
// the engine omits draft_n from timings when a request drafted nothing, so "no draft head" and
// "drafted nothing" differ only here: /slots says whether a slot can speculate (null: not served)
const props = (await get("/props")) as { model_path?: string; build_info?: string };
const slots = await get("/slots").catch(() => null);
const subject = {
  model: props.model_path?.split("/").at(-1) ?? null,
  build: props.build_info ?? null,
  speculative: Array.isArray(slots)
    ? slots.some((slot: { speculative?: boolean }) => slot.speculative === true)
    : null,
};
console.log(
  `${label}: :${port} serves ${subject.model} (${subject.build}), drafting ${subject.speculative ?? "unknown (/slots not served)"}`,
);

type Completion = {
  choices: { message: { reasoning_content?: string; content?: string } }[];
  timings: {
    predicted_n: number;
    predicted_ms: number;
    predicted_per_second: number;
    draft_n?: number;
    draft_n_accepted?: number;
  };
};

type Row = typeof subject & {
  prompt: Name;
  thinking: boolean;
  n: number;
  ms: number;
  tps: number;
  draft_n: number;
  draft_acc: number;
  sha: string;
  text: string;
};

async function run(name: Name, prompt: string, thinking: boolean): Promise<Row> {
  const res = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature: 0,
      top_k: 1,
      seed: 1,
      cache_prompt: false,
      chat_template_kwargs: { enable_thinking: thinking },
    }),
    signal: AbortSignal.timeout(1_800_000),
  });
  if (!res.ok) throw new Error(`head on :${port} answered ${res.status} ${res.statusText}`);
  const d = (await res.json()) as Completion;
  const m = d.choices[0]!.message;
  const t = d.timings;
  const text = `${m.reasoning_content ?? ""}\n---\n${m.content ?? ""}`;
  return {
    ...subject,
    prompt: name,
    thinking,
    n: t.predicted_n,
    ms: t.predicted_ms,
    tps: Math.round(t.predicted_per_second * 10) / 10,
    draft_n: t.draft_n ?? 0,
    draft_acc: t.draft_n_accepted ?? 0,
    sha: new CryptoHasher("sha256").update(text).digest("hex").slice(0, 12),
    text,
  };
}

const rows: Row[] = [];
await run("code", PROMPTS.code, false); // warm-up, discarded
for (const name of Object.keys(PROMPTS) as Name[]) rows.push(await run(name, PROMPTS[name], false));
for (const name of ["code", "reasoning"] as const) rows.push(await run(name, PROMPTS[name], true));

// "wx": the check above is minutes old by now
writeFileSync(out, `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`, { flag: "wx" });

// the server's own milliseconds, not n over the rounded rate (a 0 rate made the aggregate Infinity)
const totN = rows.reduce((s, r) => s + r.n, 0);
const totT = rows.reduce((s, r) => s + r.ms, 0) / 1000;
for (const r of rows) {
  const acc =
    r.draft_n > 0
      ? `${r.draft_acc}/${r.draft_n} (${(r.draft_acc / r.draft_n).toFixed(2)})`
      : subject.speculative === true
        ? "none drafted"
        : subject.speculative === false
          ? "no draft head"
          : "not reported";
  console.log(
    `${label.padEnd(12)} ${r.prompt.padEnd(9)} think=${r.thinking ? 1 : 0} ` +
      `n=${String(r.n).padStart(4)} tps=${r.tps.toFixed(1).padStart(6)} ` +
      `accepted=${acc.padEnd(18)} sha=${r.sha}`,
  );
}
console.log(
  `${label.padEnd(12)} AGGREGATE ${totN} tokens / ${totT.toFixed(1)}s = ${(totN / totT).toFixed(1)} tok/s`,
);
