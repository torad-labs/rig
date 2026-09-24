// llama-bench's depth matrix on the served pack with the head's cache types: prefill (pp512) and
// decode (tg64) at each context depth, three repeats, the table llama-bench prints. A measurement
// the evidence cites, not a verdict: the numbers are what a card does, and there is no threshold
// that means anything across cards.
import type { Probe } from "./probe.ts";

/** the head's --cache-type-k/-v as llama-bench spells them (-ctk/-ctv) */
export function cacheTypeArgs(runtimeArgs: readonly string[]): string[] {
  const out: string[] = [];
  runtimeArgs.forEach((arg, index) => {
    const value = runtimeArgs[index + 1];
    if (value === undefined) return;
    if (arg === "--cache-type-k") out.push("-ctk", value);
    if (arg === "--cache-type-v") out.push("-ctv", value);
  });
  return out;
}

export const depthProbe: Probe = {
  name: "depth",
  needs: "card",
  async run(ctx) {
    const cfg = ctx.gates.depth;
    const argv = [
      `${ctx.binDir}/llama-bench`,
      "-m",
      ctx.head.servedPath,
      "-ngl",
      "99",
      "-fa",
      "1",
      ...cacheTypeArgs(ctx.head.runtime.args),
      "-p",
      String(cfg.prompt),
      "-n",
      String(cfg.gen),
      "-d",
      cfg.depths.join(","),
      "-r",
      String(cfg.repeats),
      "-o",
      "md",
    ];
    const bench = await ctx.shell.run(argv, {
      env: {
        CUDA_DEVICE_ORDER: "PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES: String(ctx.gpu),
        LD_LIBRARY_PATH: ctx.binDir,
      },
      timeoutMs: 3_600_000,
    });
    if (bench.code !== 0) {
      return {
        name: "depth",
        pass: false,
        summary: `llama-bench exited ${bench.code}`,
        lines: bench.stderr.trim().split("\n").slice(-5),
        data: { cmd: argv, stderr: bench.stderr },
      };
    }
    return {
      name: "depth",
      pass: "measured",
      summary: `${cfg.depths.length} depths x pp${cfg.prompt}/tg${cfg.gen}, ${cfg.repeats} repeats`,
      lines: bench.stdout.trim().split("\n"),
      data: { cmd: argv, table: bench.stdout },
    };
  },
};
