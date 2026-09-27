// Probes against the LIVE head, the way several seats use it: four concurrent long-context
// sessions in the shared KV pool, each returning its own marker and no other (slot isolation and
// cache eviction show up here or nowhere), and aggregate throughput at M = 1, 2, 4.
import type { HeadClient } from "../head-client.ts";
import { liveHead, type Probe, sum } from "./probe.ts";

/** a slice of the filler starting at `offset` (wrapping, so a shorter filler still gives every
 *  session its own start and no shared prefix to cache), grown to `target` tokens by measured
 *  feedback */
async function slice(
  client: HeadClient,
  filler: string,
  offset: number,
  target: number,
): Promise<string> {
  const from = offset % filler.length;
  const rotated = filler.slice(from) + filler.slice(0, from);
  const take = (chars: number) => {
    let text = "";
    while (text.length < chars) text += rotated;
    return text.slice(0, chars);
  };
  let chars = target * 3;
  for (let attempt = 0; attempt < 8; attempt++) {
    const text = take(chars);
    const tokens = await client.countTokens(text);
    if (Math.abs(tokens - target) < target * 0.02) return text;
    chars = Math.round(chars * (target / Math.max(tokens, 1)));
  }
  return take(chars);
}

interface SessionPrompt {
  name: string;
  marker: string;
  depth: number;
  question: string;
  prompt: string;
  tokens: number;
}

interface SessionAnswer extends SessionPrompt {
  answer: string;
  usage: number;
}

interface SessionRow {
  name: string;
  own: boolean;
  foreign: string[];
  usage: number;
  answer: string;
}

export const sessionsProbe: Probe = {
  name: "sessions",
  needs: "live",
  async run(ctx) {
    const client = liveHead(ctx);
    const cfg = ctx.gates.sessions;
    const filler = await ctx.corpus();

    // disjoint slices: no shared prefix to cache
    const prompts: SessionPrompt[] = await Promise.all(
      cfg.sessions.map(async (session, index) => {
        const body = await slice(client, filler, index * 700_000, cfg.tokens_each);
        const cut = Math.floor(body.length * session.depth);
        const note = `\n\nNOTE: ${session.question} is ${session.marker}. Remember it.\n\n`;
        const text = `${body.slice(0, cut)}${note}${body.slice(cut)}`;
        return {
          ...session,
          prompt: `${text}\n\nWhat is ${session.question}? Answer with the code only.`,
          tokens: await client.countTokens(text),
        };
      }),
    );

    const started = Date.now();
    const answers: SessionAnswer[] = await Promise.all(
      prompts.map(async (session) => {
        try {
          const reply = await client.chat(session.prompt, {
            maxTokens: 40,
            greedy: false,
            timeoutMs: false,
          });
          return { ...session, answer: reply.text.trim(), usage: reply.promptTokens };
        } catch (error) {
          const message = (error as Error).message.slice(0, 120);
          return { ...session, answer: `ERROR ${message}`, usage: 0 };
        }
      }),
    );
    const secs = (Date.now() - started) / 1000;

    const rows: SessionRow[] = answers.map((session) => {
      const own = session.answer.includes(session.marker);
      const foreign = cfg.sessions
        .filter((other) => other.name !== session.name && session.answer.includes(other.marker))
        .map((other) => other.name);
      return {
        name: session.name,
        own,
        foreign,
        usage: session.usage,
        answer: session.answer.slice(0, 40),
      };
    });
    const lines = [
      ...prompts.map(
        (session) =>
          `  ${session.name}: ~${session.tokens.toLocaleString()} tokens, marker at ${(session.depth * 100).toFixed(0)}%`,
      ),
      ...rows.map((row) => {
        const verdict = row.own && row.foreign.length === 0 ? "PASS" : "FAIL";
        const leaked = row.foreign.length ? `  LEAKED FROM: ${row.foreign.join(",")}` : "";
        return `  ${verdict}  ${row.name.padEnd(8)} prompt ${String(row.usage).padStart(6)} tok  own marker ${row.own}${leaked}  ${JSON.stringify(row.answer)}`;
      }),
      `  ${cfg.sessions.length} concurrent long-context sessions in ${secs.toFixed(1)} s`,
    ];
    const failed = rows.filter((row) => !row.own || row.foreign.length > 0).length;
    return {
      name: "sessions",
      pass: failed === 0,
      summary: `${rows.length - failed}/${rows.length} sessions retrieved their own marker and no other`,
      lines,
      data: { rows, secs },
    };
  },
};

interface BatchRow {
  m: number;
  secs: number;
  tokens: number;
  aggregate: number;
}

export const concurrencyProbe: Probe = {
  name: "concurrency",
  needs: "live",
  async run(ctx) {
    const client = liveHead(ctx);
    const cfg = ctx.gates.concurrency;
    const oneStream = async () => {
      const reply = await client.chat(cfg.prompt, { maxTokens: cfg.max_tokens, greedy: false });
      return reply.completionTokens;
    };

    const rows: BatchRow[] = [];
    for (const m of cfg.batches) {
      const started = Date.now();
      const counts = await Promise.all(Array.from({ length: m }, oneStream));
      const secs = (Date.now() - started) / 1000;
      const tokens = sum(counts);
      rows.push({ m, secs, tokens, aggregate: tokens / secs });
    }

    const lines = rows.map(
      (row) =>
        `  M=${row.m}  wall=${row.secs.toFixed(1)} s  tokens=${String(row.tokens).padStart(4)}  aggregate=${row.aggregate.toFixed(0).padStart(3)} tok/s  per-stream=${(row.aggregate / row.m).toFixed(0).padStart(3)} tok/s`,
    );
    const first = rows[0];
    const last = rows.at(-1);
    if (!first || !last) throw new Error("concurrency: gates.toml names no batches");
    const gain = last.aggregate / first.aggregate;
    return {
      name: "concurrency",
      pass: gain >= cfg.min_gain,
      summary: `aggregate M=${first.m} ${first.aggregate.toFixed(0)} → M=${last.m} ${last.aggregate.toFixed(0)} tok/s (${gain.toFixed(2)}x)`,
      lines,
      data: rows,
    };
  },
};
