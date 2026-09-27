// A-B-A decode: base, served, base again on fresh servers, 3 x N greedy tokens a leg with the
// first discarded as warmup. A 2-bit dequant cannot depend on the digit values, so the served
// leg must sit inside the base's own A-A spread (plus tolerance_pct); a gap outside it is the
// pack.
import type { HeadClient } from "../head-client.ts";
import { fixed1, type Probe, sum } from "./probe.ts";

export interface Leg {
  warmup: number;
  runs: number[];
  mean: number;
}

export function abaVerdict(
  base1: number,
  served: number,
  base2: number,
  tolerancePct: number,
): boolean {
  const spread = Math.abs(base1 - base2);
  const center = (base1 + base2) / 2;
  return Math.abs(served - center) <= spread + (center * tolerancePct) / 100;
}

async function measureLeg(client: HeadClient, prompt: string, nPredict: number): Promise<Leg> {
  const speeds: number[] = [];
  for (let run = 0; run < 3; run++) {
    const reply = await client.completion(prompt, { nPredict });
    speeds.push(reply.timings.predicted_per_second ?? 0);
  }
  const [warmup = 0, ...runs] = speeds;
  return { warmup, runs, mean: sum(runs) / runs.length };
}

export const decodeProbe: Probe = {
  name: "decode",
  needs: "server",
  async run(ctx) {
    const { prompt, n_predict, tolerance_pct } = ctx.gates.decode;
    const leg = { ctx: 8192, slots: 1 };
    const order = [
      ["base-1", ctx.head.sourcePath],
      ["served", ctx.head.servedPath],
      ["base-2", ctx.head.sourcePath],
    ] as const;
    const legs: Record<string, Leg> = {};
    for (const [label, pack] of order) {
      legs[label] = await ctx.server.leg({ label: `decode-${label}`, pack, ...leg }, (client) =>
        measureLeg(client, prompt, n_predict),
      );
    }
    const base1 = legs["base-1"];
    const served = legs.served;
    const base2 = legs["base-2"];
    if (!base1 || !served || !base2) throw new Error("decode: a leg did not run");

    const lines = Object.entries(legs).map(
      ([label, measured]) =>
        `${label.padEnd(8)} warmup ${fixed1(measured.warmup)}   runs ${measured.runs.map(fixed1).join(" ")}   mean(runs) ${fixed1(measured.mean)} tok/s`,
    );
    return {
      name: "decode",
      pass: abaVerdict(base1.mean, served.mean, base2.mean, tolerance_pct),
      summary: `base ${fixed1(base1.mean)} / served ${fixed1(served.mean)} / base ${fixed1(base2.mean)} tok/s`,
      lines,
      data: legs,
    };
  },
};
