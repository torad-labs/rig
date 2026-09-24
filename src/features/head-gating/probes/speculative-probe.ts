// The draft head changes the clock, never the choice: two legs on the served pack, the same
// greedy prompts, one with the head's draft and one without. A drafted token is only ever one the
// pack verified — but the verify batch puts 1 + n rows through the recurrence at once, and that
// kernel's accumulation order is not the one-token step's, so at a near-tie the two legs can pick
// different tokens and part for good (evidence.md, "Draft head"). The contract is therefore: the
// two answers are byte-identical, or their first differing token is a near-tie in the plain leg's
// own distribution — the drafted token sits within `tie_gap` nats of the plain leg's top-1 there.
// A drafted token the pack did not rate is a wrong token and fails; so does a draft that did not
// run (draft_n = 0) and a drafted leg slower than min_gain says (a host whose CPU cannot keep up
// with the draft rounds should opt out rather than ship a slower head).
import type { HeadClient, TokenLogprob } from "../head-client.ts";
import { answerText, fixed1, type Probe, short, sum } from "./probe.ts";

/** ranked alternatives asked for per token: the drafted token must appear here to be measured */
export const TOP_LOGPROBS = 10;

export interface SpecLeg {
  answers: string[];
  tokens: TokenLogprob[][];
  tps: number;
  draftN: number;
  accepted: number;
}

/** where two greedy token streams first part: the index, both tokens, and how far the drafted
 *  token sat below the plain leg's top-1 there (null when the plain leg did not rank it) */
export interface Divergence {
  index: number;
  plain: string;
  drafted: string;
  gap: number | null;
}

export interface PromptVerdict {
  identical: boolean;
  divergence: Divergence | null;
  tie: boolean;
}

export interface SpecVerdict {
  prompts: PromptVerdict[];
  ran: boolean;
  fast: boolean;
}

export function firstDivergence(plain: TokenLogprob[], drafted: TokenLogprob[]): Divergence | null {
  const length = Math.max(plain.length, drafted.length);
  for (let index = 0; index < length; index++) {
    const own = plain[index];
    const other = drafted[index];
    if (own && other && own.token === other.token) continue;
    const top = own?.top ?? [];
    const rated = other ? top.find((choice) => choice.token === other.token) : undefined;
    const best = top.length ? Math.max(...top.map((choice) => choice.logprob)) : null;
    return {
      index,
      plain: own?.token ?? "",
      drafted: other?.token ?? "",
      gap: rated && best !== null ? best - rated.logprob : null,
    };
  }
  return null;
}

export function speculativeVerdict(
  plain: SpecLeg,
  drafted: SpecLeg,
  minGain: number,
  tieGap: number,
): SpecVerdict {
  return {
    prompts: plain.answers.map((answer, index) => {
      const identical = answer === drafted.answers[index];
      const divergence = identical
        ? null
        : firstDivergence(plain.tokens[index] ?? [], drafted.tokens[index] ?? []);
      const tie = divergence?.gap != null && divergence.gap <= tieGap;
      return { identical, divergence, tie };
    }),
    ran: drafted.draftN > 0,
    fast: drafted.tps >= plain.tps * minGain,
  };
}

async function ask(client: HeadClient, prompts: string[], maxTokens: number): Promise<SpecLeg> {
  const answers: string[] = [];
  const tokens: TokenLogprob[][] = [];
  const speeds: number[] = [];
  let draftN = 0;
  let accepted = 0;
  for (const prompt of prompts) {
    const reply = await client.chat(prompt, { maxTokens, topLogprobs: TOP_LOGPROBS });
    answers.push(answerText(reply));
    tokens.push(reply.tokens);
    speeds.push(reply.timings.predicted_per_second ?? 0);
    draftN += reply.timings.draft_n ?? 0;
    accepted += reply.timings.draft_n_accepted ?? 0;
  }
  return { answers, tokens, tps: sum(speeds) / speeds.length, draftN, accepted };
}

function describe(verdict: PromptVerdict, tieGap: number): string {
  if (verdict.identical) return "identical";
  const at = verdict.divergence;
  if (!at) return "DIFFERS (no token logprobs to locate the divergence)";
  const where = `at token ${at.index}: plain ${JSON.stringify(at.plain)} vs drafted ${JSON.stringify(at.drafted)}`;
  if (at.gap === null)
    return `DIFFERS ${where}, drafted token not in the pack's top ${TOP_LOGPROBS}`;
  const below = `${at.gap.toFixed(3)} nats below the pack's top-1`;
  return verdict.tie
    ? `near-tie ${where}, ${below} (tie ≤ ${tieGap})`
    : `DIFFERS ${where}, ${below} (tie ≤ ${tieGap})`;
}

export const speculativeProbe: Probe = {
  name: "speculative",
  needs: "server",
  applies: (head) => (head.speculative ? null : "the head declares no draft head ([speculative])"),
  async run(ctx) {
    const { max_tokens, prompts, min_gain, tie_gap } = ctx.gates.speculative;
    const leg = { pack: ctx.head.servedPath, ctx: 8192, slots: 1 };
    const plain = await ctx.server.leg({ label: "speculative-plain", ...leg }, (client) =>
      ask(client, prompts, max_tokens),
    );
    const drafted = await ctx.server.leg(
      { label: "speculative-drafted", ...leg, draft: true },
      (client) => ask(client, prompts, max_tokens),
    );

    const verdict = speculativeVerdict(plain, drafted, min_gain, tie_gap);
    const acceptance = drafted.draftN
      ? `${((drafted.accepted / drafted.draftN) * 100).toFixed(0)} % of ${drafted.draftN} drafted accepted`
      : "the draft did not run";
    const gain = (drafted.tps / (plain.tps || 1)).toFixed(2);
    const lines = [
      `plain ${fixed1(plain.tps)} tok/s, drafted ${fixed1(drafted.tps)} tok/s (x${gain}, min ${min_gain}); ${acceptance}`,
      ...verdict.prompts.map(
        (prompt, index) =>
          `  ${index} ${describe(prompt, tie_gap)}  ${short(drafted.answers[index] ?? "", 60)}`,
      ),
    ];
    const identical = verdict.prompts.filter((prompt) => prompt.identical).length;
    const ties = verdict.prompts.filter((prompt) => prompt.tie).length;
    return {
      name: "speculative",
      pass:
        verdict.prompts.every((prompt) => prompt.identical || prompt.tie) &&
        verdict.ran &&
        verdict.fast,
      summary: `${identical}/${verdict.prompts.length} identical (${ties} at a near-tie), ${acceptance}, ${fixed1(plain.tps)} → ${fixed1(drafted.tps)} tok/s`,
      lines,
      data: { plain, drafted, verdict },
    };
  },
};
