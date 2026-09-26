// The kernels one decode step launches, counted from an Nsight Systems capture of llama-bench and
// held to gates.toml's table for the gate card's sm. A fold or fusion that silently stops firing
// moves a count exactly, where a throughput A/B cannot see a 1-2 % loss: after the fork's rms_norm
// + FWHT fold (#51) the conv-state fold ran on 1 of 48 Gated DeltaNet layers (fixed in #58):
// ssm_conv_state_update_f32 48 -> 1, concat_cont, ssm_conv_f32 and l2_norm_f32 0 -> 47,
// k_get_rows_float 1 -> 48 and cpy_scalar 64 -> 111 per step, and no gate failed.
// A step is one cudaGraphLaunch: every kernel of a replay carries the launch's correlationId.
// Launches are grouped by their kernel-count vector and the most frequent is the step; the others
// are reported, not gated; kernels launched outside a graph (the evaluations before capture) are
// ignored. A kernel is its CUPTI short name, the function without template arguments, nothing
// else normalized. The table is the pin's: a fork change that moves a count on purpose brings its
// new table to the rig change that moves the pin, and this probe prints the step as that table.
import { Database } from "bun:sqlite";
import { join } from "node:path";
import { cacheTypeArgs } from "./depth-probe.ts";
import type { Probe, ProbeResult } from "./probe.ts";

/** kernel short name -> launches */
export type KernelCounts = Record<string, number>;

/** one kernel a graph replay ran: the replay's cudaGraphLaunch correlationId and the kernel */
export interface ReplayKernel {
  launch: number;
  kernel: string;
}

export interface Census {
  step: KernelCounts;
  /** the replays that ran exactly the step, of all of them */
  steps: number;
  launches: number;
  /** the other replays' kernels, reported and not gated */
  others: { launches: number; counts: KernelCounts }[];
}

/** the step from the kernels graph replays ran; an error when there is no replay, or no single
 *  most frequent one */
export function censusOf(rows: Iterable<ReplayKernel>): Census | string {
  const perLaunch = new Map<number, KernelCounts>();
  for (const { launch, kernel } of rows) {
    const counts = perLaunch.get(launch) ?? {};
    counts[kernel] = (counts[kernel] ?? 0) + 1;
    perLaunch.set(launch, counts);
  }
  const vectors = new Map<string, { launches: number; counts: KernelCounts }>();
  for (const counts of perLaunch.values()) {
    const key = JSON.stringify(Object.entries(counts).sort(([a], [b]) => (a < b ? -1 : 1)));
    const seen = vectors.get(key);
    if (seen) seen.launches += 1;
    else vectors.set(key, { launches: 1, counts });
  }
  const [modal, ...others] = [...vectors.values()].sort((a, b) => b.launches - a.launches);
  if (!modal) return "no kernel was launched by a cudaGraphLaunch in the capture";
  if (others[0]?.launches === modal.launches) {
    return `no step: ${modal.launches + others[0].launches} of ${perLaunch.size} graph replays run two different kernel sets equally often`;
  }
  return { step: modal.counts, steps: modal.launches, launches: perLaunch.size, others };
}

/** every kernel whose count differs, a kernel only one side has included: [name, want, got] */
export function censusDiff(want: KernelCounts, got: KernelCounts): [string, number, number][] {
  const names = [...new Set([...Object.keys(want), ...Object.keys(got)])].sort();
  return names.flatMap((name): [string, number, number][] => {
    const [w, g] = [want[name] ?? 0, got[name] ?? 0];
    return w === g ? [] : [[name, w, g]];
  });
}

/** the step as gates.toml's table lines, most launched first */
export function censusTable(step: KernelCounts): string[] {
  return Object.entries(step)
    .sort(([a, x], [b, y]) => y - x || (a < b ? -1 : 1))
    .map(([name, launches]) => `${name} = ${launches}`);
}

const REPLAY_KERNELS = `
select k.correlationId as launch, s.value as kernel
from CUPTI_ACTIVITY_KIND_KERNEL k
join StringIds s on s.id = k.shortName
join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = k.correlationId
join StringIds rn on rn.id = r.nameId
where rn.value like 'cudaGraphLaunch%'`;

/** the kernels graph replays ran, from `nsys export -t sqlite` of a capture made with
 *  --cuda-graph-trace=node (without it a replay is one row, not its kernels) */
export function readCapture(path: string): ReplayKernel[] {
  const db = new Database(path, { readonly: true });
  try {
    return db.query(REPLAY_KERNELS).all() as ReplayKernel[];
  } finally {
    db.close();
  }
}

const sum = (counts: KernelCounts) => Object.values(counts).reduce((total, n) => total + n, 0);

export const censusProbe: Probe = {
  name: "census",
  needs: "card",
  async run(ctx) {
    const failed = (summary: string, lines: string[], data: unknown = null): ProbeResult => ({
      name: "census",
      pass: false,
      summary,
      lines,
      data,
    });
    const table = ctx.gates.census.kernels[`sm${ctx.cap}`];
    if (!table) {
      return failed(`gates.toml has no census for sm_${ctx.cap}`, [
        `add [census.kernels.sm${ctx.cap}]: the lines this probe prints for a capture on this card`,
      ]);
    }
    if (!(await ctx.shell.which("nsys"))) {
      return failed("nsys is not on PATH", [
        "the census counts kernels from an Nsight Systems capture (apt: cuda-nsight-systems-13-0)",
      ]);
    }
    const capture = join(ctx.runDir, "census");
    const bench = [
      `${ctx.binDir}/llama-bench`,
      "-m",
      ctx.head.servedPath,
      "-ngl",
      "99",
      "-fa",
      "1",
      ...cacheTypeArgs(ctx.head.runtime.args),
      "-p",
      "0",
      "-n",
      String(ctx.gates.census.gen),
      "-r",
      "1",
    ];
    const profile = ["nsys", "profile", "-t", "cuda", "--cuda-graph-trace=node", "-f", "true"];
    const cmd = [...profile, "-o", capture, ...bench];
    const profiled = await ctx.shell.run(cmd, {
      env: {
        CUDA_DEVICE_ORDER: "PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES: String(ctx.gpu),
        LD_LIBRARY_PATH: ctx.binDir,
      },
      timeoutMs: 1_800_000,
    });
    if (profiled.code !== 0) {
      const tail = profiled.stderr.trim().split("\n").slice(-5);
      return failed(`nsys profile exited ${profiled.code}`, tail, { cmd });
    }
    const exportCmd = ["nsys", "export", "-t", "sqlite", "-f", "true"];
    const exported = await ctx.shell.run(
      [...exportCmd, "-o", `${capture}.sqlite`, `${capture}.nsys-rep`],
      { timeoutMs: 600_000 },
    );
    if (exported.code !== 0) {
      const tail = exported.stderr.trim().split("\n").slice(-5);
      return failed(`nsys export exited ${exported.code}`, tail, { cmd });
    }
    const census = censusOf(readCapture(`${capture}.sqlite`));
    if (typeof census === "string") return failed(census, [], { cmd });

    const diff = censusDiff(table, census.step);
    const others = census.others.map(
      ({ launches, counts }) =>
        `${launches} of ${census.launches} replays are another graph (not gated): ${censusDiff(
          census.step,
          counts,
        )
          .map(([name, step, other]) => `${name} ${step} -> ${other}`)
          .join(", ")}`,
    );
    const replays = `${census.steps} of ${census.launches} graph replays`;
    const data = { cmd, step: census.step, steps: census.steps, launches: census.launches, diff };
    if (diff.length > 0) {
      return failed(
        `${diff.length} kernels launch a different number of times per step: ${sum(census.step)} launches against ${sum(table)} (${replays})`,
        [
          ...diff.map(([name, want, got]) => `${name}: ${want} -> ${got}`),
          ...others,
          `the step on this build, as [census.kernels.sm${ctx.cap}]:`,
          ...censusTable(census.step),
        ],
        data,
      );
    }
    return {
      name: "census",
      pass: true,
      summary: `${sum(census.step)} launches per step, ${Object.keys(table).length} kernels as the table (${replays})`,
      lines: others,
      data,
    };
  },
};
