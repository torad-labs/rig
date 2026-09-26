// Decode at the depth the head is used. A coding agent fills one slot's window past 245K tokens,
// where a decode step reads ~4.5 GB of KV cache beside ~6.8 GB of weights and the draft head reads
// its own KV at every draft step: a speed measured on short prompts says nothing there. Two legs
// on the served pack, one slot each on a fresh server: the corpus grown to `tokens` by the
// server's own count, then each question in turn, greedy with thinking on. The first question pays
// the whole prefill; the others resume from the slot's checkpoint (the prompt cache on, for this
// probe only). The drafted leg must decode at least min_gain times the plain leg's tok/s, and the
// draft must run.
import type { HeadClient } from "../head-client.ts";
import { grow } from "./needle-probe.ts";
import { fixed1, type Probe, sum } from "./probe.ts";

export interface DepthLeg {
  /** decode tok/s per question */
  tps: number[];
  /** the first question's prompt: the whole document, prefilled cold */
  promptTokens: number;
  prefillSecs: number;
  draftN: number;
  accepted: number;
}

export interface DepthVerdict {
  gain: number;
  ran: boolean;
  fast: boolean;
}

const mean = (values: number[]) => sum(values) / (values.length || 1);

/** a plain leg that measured no decode (no timings, nothing predicted) is no baseline: its gain is
 *  0, never the drafted leg's own tok/s over 1 */
export function depthVerdict(plain: DepthLeg, drafted: DepthLeg, minGain: number): DepthVerdict {
  const base = mean(plain.tps);
  const gain = base > 0 ? mean(drafted.tps) / base : 0;
  return { gain, ran: drafted.draftN > 0, fast: gain >= minGain };
}

/** the questions the template wraps around the document: room kept for them under `tokens` */
const QUESTION_ROOM = 512;

async function askAll(
  client: HeadClient,
  filler: string,
  tokens: number,
  questions: string[],
  maxTokens: number,
): Promise<DepthLeg> {
  const document = await grow(client, filler, tokens - QUESTION_ROOM);
  const leg: DepthLeg = { tps: [], promptTokens: 0, prefillSecs: 0, draftN: 0, accepted: 0 };
  for (const question of questions) {
    const reply = await client.chat(`${document}\n\n${question}`, {
      maxTokens,
      thinking: true,
      cachePrompt: true,
      timeoutMs: false,
    });
    if (leg.tps.length === 0) {
      leg.promptTokens = reply.promptTokens;
      leg.prefillSecs = (reply.timings.prompt_ms ?? 0) / 1000;
    }
    leg.tps.push(reply.timings.predicted_per_second ?? 0);
    leg.draftN += reply.timings.draft_n ?? 0;
    leg.accepted += reply.timings.draft_n_accepted ?? 0;
  }
  return leg;
}

export const longctxProbe: Probe = {
  name: "longctx",
  needs: "server",
  applies: (head) => (head.speculative ? null : "the head declares no draft head ([speculative])"),
  async run(ctx) {
    const { tokens, ctx: window, max_tokens, questions, min_gain } = ctx.gates.longctx;
    const filler = await ctx.corpus();
    const leg = { pack: ctx.head.servedPath, ctx: window, slots: 1 };
    const measure = (client: HeadClient) => askAll(client, filler, tokens, questions, max_tokens);
    const plain = await ctx.server.leg({ label: "longctx-plain", ...leg }, measure);
    const drafted = await ctx.server.leg(
      { label: "longctx-drafted", ...leg, draft: true },
      measure,
    );

    const verdict = depthVerdict(plain, drafted, min_gain);
    const acceptance = drafted.draftN
      ? `${((drafted.accepted / drafted.draftN) * 100).toFixed(0)} % of ${drafted.draftN} drafted accepted`
      : "the draft did not run";
    const at = drafted.promptTokens.toLocaleString();
    const lines = [
      `at ${at} prompt tokens: plain ${fixed1(mean(plain.tps))} tok/s, drafted ${fixed1(mean(drafted.tps))} tok/s (x${verdict.gain.toFixed(2)}, min ${min_gain}); ${acceptance}`,
      `  plain by question   ${plain.tps.map(fixed1).join(" / ")} tok/s; cold prefill ${fixed1(plain.prefillSecs)} s`,
      `  drafted by question ${drafted.tps.map(fixed1).join(" / ")} tok/s; cold prefill ${fixed1(drafted.prefillSecs)} s`,
    ];
    return {
      name: "longctx",
      pass: verdict.ran && verdict.fast,
      summary: `at ${at} tokens ${fixed1(mean(plain.tps))} → ${fixed1(mean(drafted.tps))} tok/s (x${verdict.gain.toFixed(2)}), ${acceptance}`,
      lines,
      data: { plain, drafted, verdict },
    };
  },
};
