// Doubled words ("the the") and immediate token repeats over 300 greedy tokens per prompt, base
// pack and served pack: the artifact an uncapped ablation produced (6 doubled on 11 prompts vs
// the base's 2). The served pack may not exceed the base on either counter.
import type { HeadClient } from "../head-client.ts";
import { answerText, type Probe, type ProbeContext, sum } from "./probe.ts";

export interface PromptCounts {
  tokens: number;
  doubled: string[];
  repeats: number;
  refusal: boolean;
}

export interface FluencyCounts {
  doubled: number;
  repeats: number;
  tokens: number;
  refusals: number;
  perPrompt: PromptCounts[];
}

export function countDoubled(text: string): string[] {
  const out: string[] = [];
  const repeatedWord = /\b(\w+) \1\b/gi;
  for (const match of text.matchAll(repeatedWord)) out.push(match[1] ?? "");
  return out;
}

export function countRepeats(text: string): number {
  const words = text.match(/\S+/g) ?? [];
  let repeats = 0;
  for (let index = 1; index < words.length; index++) {
    const word = words[index];
    if (word !== undefined && word === words[index - 1] && word.length > 1) repeats++;
  }
  return repeats;
}

async function measure(client: HeadClient, ctx: ProbeContext): Promise<FluencyCounts> {
  const refusal = new RegExp(ctx.gates.refusal.pattern, "i");
  const perPrompt: PromptCounts[] = [];
  for (const prompt of ctx.gates.fluency.prompts) {
    const reply = await client.chat(prompt, { maxTokens: ctx.gates.fluency.max_tokens });
    const text = answerText(reply);
    perPrompt.push({
      tokens: reply.completionTokens,
      doubled: countDoubled(text),
      repeats: countRepeats(text),
      refusal: refusal.test(text.slice(0, 200)),
    });
  }
  return {
    doubled: sum(perPrompt.map((counts) => counts.doubled.length)),
    repeats: sum(perPrompt.map((counts) => counts.repeats)),
    tokens: sum(perPrompt.map((counts) => counts.tokens)),
    refusals: perPrompt.filter((counts) => counts.refusal).length,
    perPrompt,
  };
}

export const fluencyProbe: Probe = {
  name: "fluency",
  needs: "server",
  async run(ctx) {
    const leg = { ctx: 8192, slots: 1 };
    const base = await ctx.server.leg(
      { label: "fluency-base", pack: ctx.head.sourcePath, ...leg },
      (client) => measure(client, ctx),
    );
    const served = await ctx.server.leg(
      { label: "fluency-served", pack: ctx.head.servedPath, ...leg },
      (client) => measure(client, ctx),
    );
    const line = (label: string, counts: FluencyCounts) =>
      `${label}: doubled words ${counts.doubled}, immediate repeats ${counts.repeats}, over ${counts.tokens} tokens; refusals ${counts.refusals}/${counts.perPrompt.length}`;
    const lines = [
      line("base  ", base),
      line("served", served),
      ...served.perPrompt.map(
        (counts, index) =>
          `  p${index}: ${String(counts.tokens).padStart(3)} tok  doubled=${counts.doubled.length} ${JSON.stringify(counts.doubled.slice(0, 4))}  repeats=${counts.repeats}  refusal=${counts.refusal}`,
      ),
    ];
    return {
      name: "fluency",
      pass: served.doubled <= base.doubled && served.repeats <= base.repeats,
      summary: `doubled ${base.doubled} → ${served.doubled}, repeats ${base.repeats} → ${served.repeats} over ${served.tokens} tokens`,
      lines,
      data: { base, served },
    };
  },
};
