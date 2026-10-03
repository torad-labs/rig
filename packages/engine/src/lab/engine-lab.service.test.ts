import { describe, expect, test } from "bun:test";
import { layoutAt, type RunOptions } from "@rig/core";
import { fakePorts } from "@rig/testing";
import { EngineLab, type LabTarget } from "./engine-lab.service.ts";

const TREE = "/r/local/engine-build-trees/t";
const RUNS = "/r/local/engine-lab";

/** ldd answering with the library the env's LD_LIBRARY_PATH puts first */
function setup() {
  const p = fakePorts();
  const lab = new EngineLab(p, layoutAt("/r"));
  p.shell.on(/^ldd /, (_cmd, opts?: RunOptions) => {
    const first = opts?.env?.LD_LIBRARY_PATH?.split(":")[0];
    return {
      code: 0,
      stdout: `\tlibggml-cuda.so.0 => ${first}/libggml-cuda.so.0 (0x7f)\n`,
      stderr: "",
    };
  });
  p.fs.put(`${TREE}/bin/libggml-cuda.so.0`, "tree's library");
  tools(p, `${TREE}/bin`);
  p.fs.put("/libs/new/libggml-cuda.so.0", "relinked library");
  return { p, lab };
}

/** the executables a build carries, as the real directory holds them: the lab names a tool it cannot find */
function tools(p: ReturnType<typeof fakePorts>, dir: string) {
  for (const tool of ["llama-bench", "llama-perplexity", "test-backend-ops"])
    p.fs.put(`${dir}/${tool}`, "elf");
}

const target = (over: Partial<LabTarget> = {}): LabTarget => ({
  tree: TREE,
  env: { GGML_CUDA_GRAPH_MAX: "0" },
  cards: [1, 0],
  ...over,
});

describe("EngineLab.opsTest", () => {
  test("each card on its CUDA index, by default and under the legacy switches, a log a run", async () => {
    const { p, lab } = setup();
    const seen: string[] = [];
    p.shell.on(/^\S*test-backend-ops /, (cmd, opts) => {
      seen.push(
        `${cmd.slice(1).join(" ")} | ${opts?.env?.CUDA_VISIBLE_DEVICES} ${opts?.env?.GGML_CUDA_X_LEGACY ?? "-"}`,
      );
      const failing = cmd.includes("CUDA0") && opts?.env?.GGML_CUDA_X_LEGACY === "1";
      return {
        code: failing ? 1 : 0,
        stdout: failing
          ? "[OP] ERR = 1 > 0.1   OP(n=1): FAIL\n  1/2 tests passed\n"
          : "  2/2 tests passed\n",
        stderr: "",
      };
    });
    const report = await lab.opsTest({
      target: target(),
      ops: ["OP"],
      legacy: { GGML_CUDA_X_LEGACY: "1" },
      run: "t1",
    });
    expect(report.ok).toBe(true);
    if (!report.ok) return;
    expect(seen).toEqual([
      "-o OP -b CUDA0 | 1,0 -",
      "-o OP -b CUDA0 | 1,0 1",
      "-o OP -b CUDA1 | 1,0 -",
      "-o OP -b CUDA1 | 1,0 1",
    ]);
    expect(report.value.failed).toBe(1);
    const failed = report.value.runs.find((run) => run.code !== 0);
    expect(failed).toMatchObject({ card: 1, mode: "legacy", op: "OP" });
    expect(failed?.verdict?.failures).toEqual(["[OP] ERR = 1 > 0.1   OP(n=1): FAIL"]);
    expect(await p.fs.readText(`${RUNS}/t1/tests-1-legacy-OP.log`)).toContain("1/2 tests passed");
  });

  test("a library asked for that the loader does not load fails before anything runs", async () => {
    const { p, lab } = setup();
    p.shell.on(/^ldd /, {
      code: 0,
      stdout: `libggml-cuda.so.0 => ${TREE}/bin/libggml-cuda.so.0\n`,
      stderr: "",
    });
    const report = await lab.opsTest({
      target: target({ lib: "/libs/new" }),
      ops: ["OP"],
      legacy: null,
      run: "t2",
    });
    expect(report.ok).toBe(false);
    if (!report.ok) expect(report.message).toContain(`not /libs/new/libggml-cuda.so.0`);
    expect(p.shell.calls.some((cmd) => cmd[0]?.endsWith("test-backend-ops"))).toBe(false);
  });
});

describe("an installed build", () => {
  test("a tree without bin/ holds its executables and libraries itself (local/engine-builds/<sha7>-sm<cap>)", async () => {
    const { p, lab } = setup();
    p.fs.put("/r/local/engine-builds/abc1234-sm120/libggml-cuda.so.0", "the prebuilt's library");
    tools(p, "/r/local/engine-builds/abc1234-sm120");
    p.shell.on(/^\S*test-backend-ops /, (_cmd, opts) => ({
      code: 0,
      stdout: `  1/1 tests passed ${opts?.env?.LD_LIBRARY_PATH}\n`,
      stderr: "",
    }));
    const report = await lab.opsTest({
      target: target({ tree: "/r/local/engine-builds/abc1234-sm120", cards: [0] }),
      ops: ["OP"],
      legacy: null,
      run: "b1",
    });
    expect(report.ok && report.value.lib.path).toBe(
      "/r/local/engine-builds/abc1234-sm120/libggml-cuda.so.0",
    );
    expect(p.shell.calls.at(-1)?.[0]).toBe("/r/local/engine-builds/abc1234-sm120/test-backend-ops");
  });
});

describe("a build with no llama-perplexity", () => {
  test("the KL legs name the missing tool and what carries it, before ldd is asked about a file that is not there", async () => {
    const { p, lab } = setup();
    const dir = "/r/local/engine-lab/base-7656925";
    p.fs.put(`${dir}/libggml-cuda.so.0`, "an installed build");
    p.fs.put(`${dir}/llama-bench`, "elf");
    const report = await lab.identity({
      target: target({ tree: dir, cards: [0] }),
      model: "/m.gguf",
      text: "/wiki.txt",
      tag: "base",
      ctx: 2048,
      chunks: 8,
      extra: [],
      run: "i1",
    });
    expect(!report.ok && report.message).toBe(
      `${dir}/llama-perplexity is not there: a build holds the targets of the day it was published and none added since (install a newer published build, or rig build --compile)`,
    );
    expect(p.shell.calls.filter((cmd) => cmd[0] === "ldd")).toEqual([]);
  });
});

describe("EngineLab.identity", () => {
  /** llama-perplexity writing its base file: `bytes` */
  const perplexity = (p: ReturnType<typeof fakePorts>, bytes: string) =>
    p.shell.on(/^\S*llama-perplexity /, (cmd) => {
      const at = cmd.indexOf("--kl-divergence-base");
      p.fs.put(cmd[at + 1] as string, bytes);
      return { code: 0, stdout: "", stderr: "Final estimate: PPL = 1.0\n" };
    });

  test("the base file against the reference, byte for byte, with the relinked library ahead of the tree's", async () => {
    const { p, lab } = setup();
    perplexity(p, "abcdef");
    p.fs.put("/refs/ref.kld", "abcdef");
    const report = await lab.identity({
      target: target({ lib: "/libs/new" }),
      model: "/m.gguf",
      text: "/wiki.txt",
      tag: "new",
      ref: "/refs/ref.kld",
      ctx: 2048,
      chunks: 8,
      extra: ["-fa", "on"],
      run: "i1",
    });
    expect(report.ok && report.value.ref).toMatchObject({ identical: true, differingBytes: 0 });
    const call = p.shell.calls.find((cmd) => cmd[0]?.endsWith("llama-perplexity")) ?? [];
    expect(call.slice(1)).toEqual([
      "-m",
      "/m.gguf",
      "-f",
      "/wiki.txt",
      "-c",
      "2048",
      "--chunks",
      "8",
      "--kl-divergence-base",
      `${RUNS}/i1/new.kld`,
      "-fa",
      "on",
    ]);
  });

  test("a file that moved is counted, a byte a difference, a longer tail whole", async () => {
    const { p, lab } = setup();
    perplexity(p, "abXdefgh");
    p.fs.put("/refs/ref.kld", "abcdef");
    const report = await lab.identity({
      target: target(),
      model: "/m.gguf",
      text: "/wiki.txt",
      tag: "mut",
      ref: "/refs/ref.kld",
      ctx: 2048,
      chunks: 8,
      extra: [],
      run: "i2",
    });
    expect(report.ok && report.value.ref).toMatchObject({ identical: false, differingBytes: 3 });
  });
});

describe("EngineLab.ab", () => {
  test("pairs alternate their order, each pair one change, and a later run continues the earlier pairs", async () => {
    const { p, lab } = setup();
    const order: string[] = [];
    p.shell.on(/^\S*llama-bench /, (_cmd, opts) => {
      const side = opts?.env?.SIDE ?? "?";
      order.push(side);
      const rate = side === "b" ? 110 : 100;
      return { code: 0, stdout: `{"samples_ts": [1, ${rate}, ${rate}]}\n`, stderr: "" };
    });
    const req = {
      target: target(),
      model: "/m.gguf",
      a: { env: { SIDE: "a" } },
      b: { lib: "/libs/new", env: { SIDE: "b" } },
      pairs: 2,
      first: 1,
      reps: 3,
      extra: ["-p", "4096"],
      run: "ab1",
    };
    const first = await lab.ab(req);
    expect(order).toEqual(["a", "b", "b", "a"]);
    const changes = first.ok ? first.value.pairs.map((pair) => pair.change) : [];
    expect(changes).toHaveLength(2);
    for (const change of changes) expect(change).toBeCloseTo(0.1, 12);
    const more = await lab.ab({ ...req, pairs: 1, first: 3 });
    expect(order.slice(4)).toEqual(["a", "b"]);
    expect(more.ok && more.value.change.n).toBe(3);
    expect(more.ok && more.value.libs.b.path).toBe("/libs/new/libggml-cuda.so.0");
  });

  test("a side's args reach its own bench and no other: the KV cache type, one engine, one library", async () => {
    const { p, lab } = setup();
    const argvs: string[][] = [];
    // -o jsonl is on the bench argv and not on the ldd one, which also names llama-bench
    p.shell.on(/-o jsonl/, (cmd) => {
      argvs.push(cmd.slice(1));
      const quantized = cmd.includes("q8_0");
      return {
        code: 0,
        stdout: `{"samples_ts": [1, ${quantized ? 90 : 100}, ${quantized ? 90 : 100}]}\n`,
        stderr: "",
      };
    });
    const report = await lab.ab({
      target: target(),
      model: "/m.gguf",
      a: { env: {}, args: ["-ctk", "f16", "-ctv", "f16"] },
      b: { env: {}, args: ["-ctk", "q8_0", "-ctv", "q8_0"] },
      pairs: 2,
      first: 1,
      reps: 3,
      extra: ["-p", "0", "-n", "128", "-d", "65536"],
      run: "ab-args",
    });
    // pair 1 runs a then b, pair 2 b then a; each side's args follow the run's own, never the other side's
    const tail = (argv: string[]) => argv.slice(argv.indexOf("-d")).join(" ");
    expect(argvs.map(tail)).toEqual([
      "-d 65536 -ctk f16 -ctv f16",
      "-d 65536 -ctk q8_0 -ctv q8_0",
      "-d 65536 -ctk q8_0 -ctv q8_0",
      "-d 65536 -ctk f16 -ctv f16",
    ]);
    expect(report.ok).toBe(true);
    if (!report.ok) return;
    expect(report.value.args).toEqual({
      a: ["-ctk", "f16", "-ctv", "f16"],
      b: ["-ctk", "q8_0", "-ctv", "q8_0"],
    });
    for (const pair of report.value.pairs) expect(pair.change).toBeCloseTo(-0.1, 12);
  });

  test("a side with no args runs the run's own and reports none", async () => {
    const { p, lab } = setup();
    const argvs: string[][] = [];
    p.shell.on(/-o jsonl/, (cmd) => {
      argvs.push([...cmd]);
      return { code: 0, stdout: `{"samples_ts": [1, 100, 100]}\n`, stderr: "" };
    });
    const report = await lab.ab({
      target: target(),
      model: "/m.gguf",
      a: { env: {} },
      b: { lib: "/libs/new", env: {} },
      pairs: 1,
      first: 1,
      reps: 3,
      extra: ["-p", "4096"],
      run: "ab-no-args",
    });
    expect(argvs.map((argv) => argv.slice(-2))).toEqual([
      ["-p", "4096"],
      ["-p", "4096"],
    ]);
    expect(report.ok && report.value.args).toEqual({ a: [], b: [] });
  });

  test("refuses a pair whose two sides resolve a different CUDA runtime", async () => {
    const { p, lab } = setup();
    // the shape that cost box 53787584 five KL runs, here as an A/B pair: one side is an engine rig
    // installed, carrying the runtime archives its publisher unpacked, and the other was unpacked from a
    // raw tarball by hand and so finds the system's cuBLAS, a different build. The engines agree; the
    // GEMMs do not.
    p.shell.on(/^ldd /, (_cmd, opts?: RunOptions) => {
      const first = opts?.env?.LD_LIBRARY_PATH?.split(":")[0];
      const cublas =
        first === "/libs/new"
          ? "/usr/lib/x86_64-linux-gnu/libcublas.so.13"
          : `${first}/libcublas.so.13`;
      return {
        code: 0,
        stdout:
          `\tlibggml-cuda.so.0 => ${first}/libggml-cuda.so.0 (0x7f)\n` +
          `\tlibcublas.so.13 => ${cublas} (0x7f)\n`,
        stderr: "",
      };
    });
    p.fs.put(`${TREE}/bin/libcublas.so.13`, "cuBLAS 13.5.1.27, the installed build's own copy");
    p.fs.put("/usr/lib/x86_64-linux-gnu/libcublas.so.13", "cuBLAS from the system, another build");
    let benched = false;
    // -o jsonl is on the bench argv and not on the ldd one, which also names llama-bench
    p.shell.on(/-o jsonl/, () => {
      benched = true;
      return { code: 0, stdout: `{"samples_ts": [1, 100, 100]}\n`, stderr: "" };
    });
    const report = await lab.ab({
      target: target(),
      model: "/m.gguf",
      a: { env: {} },
      b: { lib: "/libs/new", env: {} },
      pairs: 1,
      first: 1,
      reps: 3,
      extra: ["-p", "4096"],
      run: "ab-runtime-drift",
    });
    expect(report.ok).toBe(false);
    if (!report.ok) {
      expect(report.message).toContain("different CUDA runtime");
      expect(report.message).toContain("libcublas.so.13");
    }
    // it refuses BEFORE burning a bench: on a 134 GB pack each pair is minutes of rented card
    expect(benched).toBe(false);
  });

  test("a pair whose sides resolve the same CUDA runtime is not refused", async () => {
    const { p, lab } = setup();
    p.shell.on(/^ldd /, (_cmd, opts?: RunOptions) => {
      const first = opts?.env?.LD_LIBRARY_PATH?.split(":")[0];
      return {
        code: 0,
        stdout:
          `\tlibggml-cuda.so.0 => ${first}/libggml-cuda.so.0 (0x7f)\n` +
          `\tlibcublas.so.13 => /opt/cuda/libcublas.so.13 (0x7f)\n` +
          `\tlibcudart.so.13 => /opt/cuda/libcudart.so.13 (0x7f)\n`,
        stderr: "",
      };
    });
    p.shell.on(/-o jsonl/, { code: 0, stdout: `{"samples_ts": [1, 100, 100]}\n`, stderr: "" });
    const report = await lab.ab({
      target: target(),
      model: "/m.gguf",
      a: { env: {} },
      b: { lib: "/libs/new", env: {} },
      pairs: 1,
      first: 1,
      reps: 3,
      extra: ["-p", "4096"],
      run: "ab-runtime-same",
    });
    expect(report.ok).toBe(true);
  });

  test("two installed builds, each with its own copy of the same runtime, are not refused", async () => {
    const { p, lab } = setup();
    // the ordinary A/B: rig installs every build with the NVIDIA archives unpacked into its OWN directory, so
    // the two sides resolve cuBLAS from two different paths. Same bytes, same GEMM algorithms: a path
    // comparison refused exactly the pair the guard exists to let through.
    p.shell.on(/^ldd /, (_cmd, opts?: RunOptions) => {
      const first = opts?.env?.LD_LIBRARY_PATH?.split(":")[0];
      return {
        code: 0,
        stdout:
          `\tlibggml-cuda.so.0 => ${first}/libggml-cuda.so.0 (0x7f)\n` +
          `\tlibcublas.so.13 => ${first}/libcublas.so.13 (0x7f)\n`,
        stderr: "",
      };
    });
    p.fs.put(`${TREE}/bin/libcublas.so.13`, "cuBLAS 13.5.1.27");
    p.fs.put("/libs/new/libcublas.so.13", "cuBLAS 13.5.1.27");
    p.shell.on(/-o jsonl/, { code: 0, stdout: `{"samples_ts": [1, 100, 100]}\n`, stderr: "" });
    const report = await lab.ab({
      target: target(),
      model: "/m.gguf",
      a: { env: {} },
      b: { lib: "/libs/new", env: {} },
      pairs: 1,
      first: 1,
      reps: 3,
      extra: ["-p", "4096"],
      run: "ab-runtime-copies",
    });
    expect(report.ok).toBe(true);
  });
});

describe("EngineLab.profile", () => {
  test("nsys over llama-bench, its trace read into each kernel's own time", async () => {
    const { p, lab } = setup();
    p.shell.on(/nsys profile/, { code: 0, stdout: "", stderr: "" });
    p.shell.on(/nsys stats/, (cmd) => {
      const base = cmd[cmd.indexOf("--output") + 1];
      p.fs.put(
        `${base}_cuda_gpu_trace.csv`,
        [
          "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name",
          '0,100,1,4,1,1,256,1,1,40,0,0,,,,,GPU (0),1,,7,"void k(int)"',
        ].join("\n"),
      );
      return { code: 0, stdout: "", stderr: "" };
    });
    const report = await lab.profile({
      target: target(),
      model: "/m.gguf",
      tag: "prefill",
      extra: ["-p", "4096", "-n", "0"],
      run: "pr1",
    });
    expect(report.ok && report.value.kernels.map((k) => [k.kernel, k.ownNs])).toEqual([["k", 100]]);
    const nsys = p.shell.calls.find((cmd) => cmd[1] === "profile") ?? [];
    expect(nsys).toContain("--cuda-graph-trace=node");
    expect(await p.fs.exists(`${RUNS}/pr1/prefill.census.json`)).toBe(true);
  });
});
