// Needle-in-a-haystack on the served pack with the head's KV settings, at the depth the cache
// type endangers: the haystack is grown to N tokens by the server's own count, the needles sit
// at line boundaries at their depths, and every secret must come back.
import type { HeadClient } from "../head-client.ts";
import type { Probe } from "./probe.ts";

export interface Needle {
  depth: number;
  secret: string;
  question: string;
}

/** the filler grown to `target` tokens by measured feedback, never by a chars/token guess */
export async function grow(client: HeadClient, filler: string, target: number): Promise<string> {
  let hay = filler;
  let tokens = await client.countTokens(hay);
  while (tokens < target) {
    hay += filler;
    tokens = await client.countTokens(hay);
  }
  return hay.slice(0, Math.floor((hay.length * target) / tokens));
}

/** the needles at their depths, deepest first so earlier insertions do not move later ones */
export function plant(hay: string, needles: Needle[]): string {
  const lines = hay.split("\n");
  const deepestFirst = [...needles].sort((left, right) => right.depth - left.depth);
  for (const needle of deepestFirst) {
    const at = Math.floor(lines.length * needle.depth);
    lines.splice(at, 0, `\nNOTE: ${needle.question} is ${needle.secret}. Remember it.\n`);
  }
  return lines.join("\n");
}

export const needleProbe: Probe = {
  name: "needle",
  needs: "server",
  async run(ctx) {
    const cfg = ctx.gates.needle;
    const filler = await ctx.corpus();
    const leg = { label: "needle", pack: ctx.head.servedPath, ctx: cfg.ctx, slots: 1 };
    return ctx.server.leg(leg, async (client) => {
      const haystack = plant(await grow(client, filler, cfg.tokens), cfg.needles);
      const promptTokens = await client.countTokens(haystack);
      const questions = cfg.needles
        .map((needle, index) => `${index + 1}. What is ${needle.question}?`)
        .join("\n");
      const question = `Above is a long document with ${cfg.needles.length} NOTE lines buried in it.\n${questions}\nAnswer with just the codes, one per line. Do not explain.`;

      const started = Date.now();
      const reply = await client.chat(`${haystack}\n\n${question}`, {
        maxTokens: 2048,
        thinking: true,
        greedy: false,
        timeoutMs: false,
      });
      const secs = (Date.now() - started) / 1000;
      const answer = `${reply.reasoning}\n${reply.text}`;
      const found = cfg.needles.map((needle) => answer.includes(needle.secret));

      const lines = [
        `haystack ${promptTokens.toLocaleString()} tokens; server usage ${reply.promptTokens.toLocaleString()} prompt / ${reply.completionTokens} completion; ${secs.toFixed(1)} s`,
        ...cfg.needles.map(
          (needle, index) =>
            `  ${found[index] ? "PASS" : "FAIL"}  depth ${(needle.depth * 100).toFixed(0).padStart(3)}%  ${needle.secret}`,
        ),
      ];
      return {
        name: "needle",
        pass: found.every(Boolean),
        summary: `${found.filter(Boolean).length}/${cfg.needles.length} needles at ${promptTokens.toLocaleString()} tokens`,
        lines,
        data: { promptTokens, found, answer: reply.text.slice(0, 400) },
      };
    });
  },
};
