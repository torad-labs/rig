import { Database } from "bun:sqlite";
import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FakeShell } from "../../../../test/fakes/index.ts";
import {
  censusDiff,
  censusOf,
  censusProbe,
  censusTable,
  type KernelCounts,
  readCapture,
} from "./census-probe.ts";
import type { ProbeContext } from "./probe.ts";

const root = `${import.meta.dir}/../../../..`;
const gates = Bun.TOML.parse(await Bun.file(`${root}/heads/bonsai-2-27b/gates.toml`).text()) as {
  census: { kernels: Record<string, KernelCounts> };
};
const dir = mkdtempSync(join(tmpdir(), "rig-census-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// Bonsai's step on the pin is gates.toml's sm120 table (c1518d4, RTX 5090, 2026-09-26). With
// GGML_CUDA_NORM_FWHT_FOLDS_LEGACY=1 the conv-state fold runs on 1 of the 48 Gated DeltaNet layers:
// against the fold on fork 7a46d1860 (chk-legacy.sqlite and chk-fix.sqlite, RTX 5070 Ti,
// 2026-09-25) six kernels moved by 47 launches each, cpy_scalar 64 -> 111
const FIX = gates.census.kernels.sm120 as KernelCounts;
const LEGACY: KernelCounts = {
  ...FIX,
  concat_cont: 47,
  cpy_scalar: (FIX.cpy_scalar ?? 0) + 47,
  k_get_rows_float: 48,
  l2_norm_f32: 47,
  ssm_conv_f32: 47,
  ssm_conv_state_update_f32: 1,
};
// the 7th replay of both captures, another graph
const OTHER: KernelCounts = { ...FIX, scale_f32_vec4: 96 };

/** a capture as nsys exports it, reduced to what the census reads: each replay's kernels under one
 *  cudaGraphLaunch, and eager kernels, each under its cudaLaunchKernel, which the census ignores */
function writeCapture(path: string, replays: KernelCounts[], eager: KernelCounts = {}) {
  rmSync(path, { force: true });
  const db = new Database(path, { create: true });
  db.run("create table StringIds (id integer primary key, value text not null)");
  db.run("create table CUPTI_ACTIVITY_KIND_RUNTIME (correlationId integer, nameId integer)");
  db.run("create table CUPTI_ACTIVITY_KIND_KERNEL (correlationId integer, shortName integer)");
  const ids = new Map<string, number>();
  const id = (value: string) => {
    const known = ids.get(value);
    if (known !== undefined) return known;
    ids.set(value, ids.size + 1);
    db.run("insert into StringIds values (?, ?)", [ids.size, value]);
    return ids.size;
  };
  let correlation = 0;
  const launch = (api: string, counts: KernelCounts) => {
    correlation += 1;
    db.run("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?)", [correlation, id(api)]);
    for (const [kernel, n] of Object.entries(counts)) {
      for (let i = 0; i < n; i++) {
        db.run("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?)", [correlation, id(kernel)]);
      }
    }
  };
  db.transaction(() => {
    for (const [kernel, n] of Object.entries(eager)) {
      for (let i = 0; i < n; i++) launch("cudaLaunchKernel_v7000", { [kernel]: 1 });
    }
    for (const counts of replays) launch("cudaGraphLaunch_v10000", counts);
  })();
  db.close();
}

describe("census", () => {
  test("a step is what one graph replay ran; the most frequent set is gated, the others reported, eager launches ignored", () => {
    const path = join(dir, "replays.sqlite");
    writeCapture(path, [FIX, FIX, FIX, OTHER, FIX, FIX, FIX], { quantize_q8_1: 5, concat_cont: 2 });
    const census = censusOf(readCapture(path));
    if (typeof census === "string") throw new Error(census);
    expect(census.step).toEqual(FIX);
    expect([census.steps, census.launches]).toEqual([6, 7]);
    expect(census.others).toEqual([{ launches: 1, counts: OTHER }]);
  });
  test("the fold fix's step passes against the table, and the legacy one fails naming the six kernels it moved", () => {
    expect(censusDiff(FIX, FIX)).toEqual([]);
    expect(censusDiff(FIX, LEGACY)).toEqual([
      ["concat_cont", 0, 47],
      ["cpy_scalar", 0, 47],
      ["k_get_rows_float", 1, 48],
      ["l2_norm_f32", 0, 47],
      ["ssm_conv_f32", 0, 47],
      ["ssm_conv_state_update_f32", 48, 1],
    ]);
    expect(Object.values(FIX).reduce((a, b) => a + b, 0)).toBe(1160);
  });
  test("no replay, or two kernel sets replayed equally often, is no step", () => {
    expect(censusOf([])).toContain("no kernel was launched by a cudaGraphLaunch");
    const tie = [
      { launch: 1, kernel: "a" },
      { launch: 2, kernel: "b" },
    ];
    expect(censusOf(tie)).toContain("no step: 2 of 2 graph replays");
  });
  test("the step prints as the table, most launched first, then by name", () => {
    expect(censusTable({ b: 1, a: 1, c: 5 })).toEqual(["c = 5", "a = 1", "b = 1"]);
  });
});

describe("census probe", () => {
  /** nsys that profiles llama-bench and exports `replays` as the capture */
  function nsys(replays: KernelCounts[]) {
    const shell = new FakeShell();
    shell.tools.add("nsys");
    const env: Record<string, string>[] = [];
    shell.on(/^nsys profile /, (_cmd, opts) => {
      env.push(opts?.env ?? {});
      return { code: 0, stdout: "", stderr: "" };
    });
    shell.on(/^nsys export /, (cmd) => {
      writeCapture(cmd[cmd.indexOf("-o") + 1]!, replays);
      return { code: 0, stdout: "", stderr: "" };
    });
    return { shell, env };
  }
  const ctx = (shell: FakeShell, cap = "120") =>
    ({
      head: {
        servedPath: "/packs/served.gguf",
        runtime: { args: ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-fa", "on"] },
      },
      gates: { census: { gen: 8, kernels: { sm120: FIX } } },
      binDir: "/r/local/engine-builds/abc1234-sm120",
      gpu: 1,
      cap,
      shell,
      runDir: dir,
    }) as unknown as ProbeContext;

  test("captures llama-bench at 1 row on the gate card under nsys with graph nodes traced, and passes the table's step", async () => {
    const { shell, env } = nsys([FIX, FIX, OTHER, FIX]);
    const result = await censusProbe.run(ctx(shell));
    expect(result.pass).toBe(true);
    expect(result.summary).toBe(
      "1160 launches per step, 17 kernels as the table (3 of 4 graph replays)",
    );
    expect(result.lines).toEqual([
      "1 of 4 replays are another graph (not gated): scale_f32_vec4 0 -> 96",
    ]);
    expect(shell.calls[0]).toEqual([
      "nsys",
      "profile",
      "-t",
      "cuda",
      "--cuda-graph-trace=node",
      "-f",
      "true",
      "-o",
      join(dir, "census"),
      "/r/local/engine-builds/abc1234-sm120/llama-bench",
      "-m",
      "/packs/served.gguf",
      "-ngl",
      "99",
      "-fa",
      "1",
      "-ctk",
      "q4_0",
      "-ctv",
      "q4_0",
      "-p",
      "0",
      "-n",
      "8",
      "-r",
      "1",
    ]);
    expect(env[0]).toMatchObject({ CUDA_VISIBLE_DEVICES: "1" });
  });
  test("a moved count fails by name, and the lines end with the step as the table to paste", async () => {
    const result = await censusProbe.run(ctx(nsys([LEGACY, LEGACY, OTHER]).shell));
    expect(result.pass).toBe(false);
    expect(result.summary).toBe(
      "6 kernels launch a different number of times per step: 1348 launches against 1160 (2 of 3 graph replays)",
    );
    expect(result.lines.slice(0, 6)).toContain("ssm_conv_state_update_f32: 48 -> 1");
    expect(result.lines).toContain("the step on this build, as [census.kernels.sm120]:");
    expect(result.lines.at(-1)).toBe("ssm_conv_state_update_f32 = 1");
  });
  test("no nsys, or no table for the gate card, fails and says what is missing", async () => {
    const bare = new FakeShell();
    expect((await censusProbe.run(ctx(bare))).summary).toBe("nsys is not on PATH");
    const h100 = await censusProbe.run(ctx(nsys([FIX]).shell, "90"));
    expect([h100.pass, h100.summary]).toEqual([false, "gates.toml has no census for sm_90"]);
  });
});
