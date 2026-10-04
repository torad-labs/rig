// An engine change measured on a developer's cmake tree (local/engine-build-trees/<name>/, its executables in bin/):
// its backend ops against the CPU on each card, its output byte for byte against a reference, its KL divergence from
// a base, a paired A/B of two libraries or two sets of switches, the kernels' own time in a profile, and a library
// relinked with chosen sources as HEAD has them (`relink.ts`). Each run's evidence (logs, KLD files, the bench's JSON
// lines, the trace) lands under local/engine-lab/<run>/. The cards are the caller's to hold: `rig engine` runs itself
// under the machine's card lease, so a whole run (every pair of an A/B) holds them once.
import { dirname, join } from "node:path";
import { ExitCode, fail, type Layout, ok, type Ports, type Result } from "@rig/core";
import {
  type ArmRate,
  type KernelShare,
  type KldSummary,
  type OpsVerdict,
  ownTimeCensus,
  type PairedChange,
  pairedChange,
  readBenchRate,
  readGpuTrace,
  readKldSummary,
  readOpsVerdict,
} from "./lab-readings.ts";
import { type RelinkReport, type RelinkRequest, relinkWithHead } from "./relink.ts";

export type LabPorts = Pick<Ports, "shell" | "fs" | "hasher" | "git" | "log">;

/** what a run executes and where */
export interface LabTarget {
  /** a cmake tree, its executables and libraries in bin/, or an installed build holding them itself
   *  (local/engine-builds/<sha7>-sm<cap>/) */
  tree: string;
  /** a directory holding a libggml-cuda.so.0 put ahead of the tree's bin/ on the loader's path (a relinked library) */
  lib?: string;
  /** assignments every run carries (GGML_CUDA_GRAPH_MAX=0, an engine switch) */
  env: Record<string, string>;
  /** nvidia-smi indices, in PCI order: CUDA numbers them CUDA0, CUDA1, ... as listed */
  cards: number[];
}

/** a library, switches and llama-bench arguments one side of an A/B runs with, over the target's and the run's own
 *  (`args` is how two sides of ONE engine differ in a runtime setting: -ctk q8_0 against -ctk f16) */
export interface LabSide {
  lib?: string;
  env: Record<string, string>;
  args?: string[];
}

/** the libggml-cuda a tool loads under a side: the path the loader resolves and the first 12 hex of its sha256,
 *  plus the CUDA runtime it resolves beside it, by soname */
export interface LoadedLib {
  path: string;
  sha: string;
  /** soname -> resolved path, for the libraries in RUNTIME */
  runtime: Record<string, string>;
}

export interface OpsRun {
  card: number;
  mode: "new" | "legacy";
  op: string;
  code: number;
  verdict: OpsVerdict | null;
  log: string;
}
export interface OpsTestReport {
  dir: string;
  lib: LoadedLib;
  runs: OpsRun[];
  /** runs that did not pass every case, or printed no verdict */
  failed: number;
}

export interface IdentityReport {
  dir: string;
  lib: LoadedLib;
  kld: string;
  bytes: number;
  sha256: string;
  log: string;
  /** set when a reference was given */
  ref: { path: string; sha256: string; identical: boolean; differingBytes: number } | null;
}

export interface KldReport {
  dir: string;
  lib: LoadedLib;
  log: string;
  summary: KldSummary;
}

export interface AbPair {
  pair: number;
  a: number;
  b: number;
  /** b/a - 1 */
  change: number;
}
export interface AbReport {
  dir: string;
  libs: { a: LoadedLib; b: LoadedLib };
  /** the arguments each side added to the run's own */
  args: { a: string[]; b: string[] };
  /** every pair in the run's directory, an earlier invocation's included (--first continues a run) */
  pairs: AbPair[];
  /** how each arm's repetitions made its rate */
  stat: ArmRate;
  change: PairedChange;
}

export interface ProfileReport {
  dir: string;
  lib: LoadedLib;
  report: string;
  trace: string;
  ownNs: number;
  nsysNs: number;
  kernels: KernelShare[];
}

const LIB = "libggml-cuda.so.0";

/** The CUDA runtime libraries a pair must share. cuBLAS picks its GEMM algorithm by version, so two
 *  sides on different cuBLAS builds differ in summation order and the pair measures the runtime
 *  rather than the engine. This is easy to do by accident: a raw portable tarball carries no runtime
 *  at all (build/build-publisher.ts unpacks NVIDIA's archives beside the libraries at INSTALL), so an
 *  unpacked tarball on one side and an installed build on the other silently compare two cuBLASes.
 *  Measured on box 53787584 (Oct 1, 2026): that pairing read mean KLD 0.011121 and same top p
 *  96.566 % between two engines whose shared code paths are bit for bit, and 0 once equalised. */
const RUNTIME = /(libcublasLt\.so[.\d]*|libcublas\.so[.\d]*|libcudart\.so[.\d]*) => (\S+)/g;

/** the RUNTIME libraries an ldd answer resolved, by soname */
function runtimeOf(ldd: string): Record<string, string> {
  const found: Record<string, string> = {};
  for (const [, soname, path] of ldd.matchAll(RUNTIME)) found[soname as string] = path as string;
  return found;
}

/** the first RUNTIME soname the two sides resolve to different BYTES, described, else null. Identity is
 *  the bytes and not the path: rig installs every build with its own copy of the runtime in its own
 *  directory, so two legitimate installed builds resolve cuBLAS from two paths. Equal paths are equal
 *  without reading; only differing paths are hashed (cuBLASLt is hundreds of MB, read once a pair). A
 *  soname only one side resolves counts as drift: that is the unpacked-tarball case, where the other
 *  side finds the system's copy instead. */
async function runtimeDrift(
  hasher: Pick<Ports["hasher"], "sha256File">,
  a: Record<string, string>,
  b: Record<string, string>,
): Promise<string | null> {
  for (const soname of [...new Set([...Object.keys(a), ...Object.keys(b)])].sort()) {
    const [x, y] = [a[soname], b[soname]];
    if (x === y) continue;
    if (x === undefined || y === undefined)
      return `${soname}: ${x ?? "(not resolved)"} against ${y ?? "(not resolved)"}`;
    const [hx, hy] = [await hasher.sha256File(x), await hasher.sha256File(y)];
    if (hx !== hy) return `${soname}: ${x} (${hx.slice(0, 12)}) against ${y} (${hy.slice(0, 12)})`;
  }
  return null;
}

/** a target with the directory its executables are in */
type Located = LabTarget & { binDir: string };

export class EngineLab {
  constructor(
    private readonly ports: LabPorts,
    private readonly layout: Layout,
  ) {}

  /** local/engine-lab/<run>/ */
  runDir(run: string): string {
    return join(this.layout.engineLabDir, run);
  }

  /** test-backend-ops -o OP on each card, by default and, with `legacy`, under those switches too */
  async opsTest(req: {
    target: LabTarget;
    ops: string[];
    legacy: Record<string, string> | null;
    filter?: string;
    run: string;
  }): Promise<Result<OpsTestReport>> {
    const { shell, fs, log } = this.ports;
    const dir = await this.open(req.run);
    const target = await this.locate(req.target);
    const lib = await this.loaded(target, {}, "test-backend-ops");
    if (!lib.ok) return lib;
    log.info(`test-backend-ops on libggml-cuda ${lib.value.sha} (${lib.value.path})`);
    const runs: OpsRun[] = [];
    const modes = req.legacy ? (["new", "legacy"] as const) : (["new"] as const);
    for (const [backend, card] of req.target.cards.entries())
      for (const mode of modes)
        for (const op of req.ops) {
          const env = this.env(target, { env: mode === "legacy" ? (req.legacy ?? {}) : {} });
          const argv = [
            this.bin(target, "test-backend-ops"),
            "-o",
            op,
            "-b",
            `CUDA${backend}`,
            ...(req.filter ? ["-p", req.filter] : []),
          ];
          const result = await shell.run(argv, { env, cwd: dir });
          const path = join(dir, `tests-${card}-${mode}-${op}.log`);
          await fs.writeText(path, result.stdout + result.stderr);
          const verdict = readOpsVerdict(result.stdout + result.stderr);
          runs.push({ card, mode, op, code: result.code, verdict, log: path });
          log.info(
            `card ${card} ${mode} ${op}: ${verdict ? `${verdict.passed}/${verdict.total} passed` : `no verdict, exit ${result.code}`}`,
          );
          for (const failure of verdict?.failures ?? []) log.info(`  ${failure}`);
        }
    const failed = runs.filter(
      (run) => run.code !== 0 || !run.verdict || run.verdict.passed < run.verdict.total,
    ).length;
    return ok({ dir, lib: lib.value, runs, failed });
  }

  /** llama-perplexity's KLD base file (its logits over `text`) written as <tag>.kld and compared byte for byte with
   *  `ref`: an engine change that should not move a value proves it here */
  async identity(req: {
    target: LabTarget;
    model: string;
    text: string;
    tag: string;
    ref?: string;
    ctx: number;
    chunks: number;
    extra: string[];
    run: string;
  }): Promise<Result<IdentityReport>> {
    const { fs, hasher, log } = this.ports;
    const dir = await this.open(req.run);
    const target = await this.locate(req.target);
    const lib = await this.loaded(target, {}, "llama-perplexity");
    if (!lib.ok) return lib;
    const kld = join(dir, `${req.tag}.kld`);
    log.info(`KLD base ${req.tag} on libggml-cuda ${lib.value.sha} (${lib.value.path})`);
    const ran = await this.perplexity({ ...req, target }, dir, req.tag, [
      "--kl-divergence-base",
      kld,
    ]);
    if (!ran.ok) return ran;
    const stat = await fs.stat(kld);
    if (!stat) return fail(ExitCode.Failure, `llama-perplexity wrote no ${kld} (log ${ran.value})`);
    const sha256 = await hasher.sha256File(kld);
    let ref: IdentityReport["ref"] = null;
    if (req.ref) {
      if (!(await fs.exists(req.ref))) return fail(ExitCode.Failure, `no reference at ${req.ref}`);
      const differingBytes = await this.differingBytes(req.ref, kld);
      ref = {
        path: req.ref,
        sha256: await hasher.sha256File(req.ref),
        identical: differingBytes === 0,
        differingBytes,
      };
    }
    return ok({
      dir,
      lib: lib.value,
      kld,
      bytes: stat.size,
      sha256,
      log: ran.value,
      ref,
    });
  }

  /** llama-perplexity --kl-divergence against `base` (a KLD base file): the change's distance from it */
  async kld(req: {
    target: LabTarget;
    model: string;
    text: string;
    base: string;
    tag: string;
    ctx: number;
    chunks: number;
    extra: string[];
    run: string;
  }): Promise<Result<KldReport>> {
    const { fs, log } = this.ports;
    const dir = await this.open(req.run);
    if (!(await fs.exists(req.base))) return fail(ExitCode.Failure, `no KLD base at ${req.base}`);
    const target = await this.locate(req.target);
    const lib = await this.loaded(target, {}, "llama-perplexity");
    if (!lib.ok) return lib;
    log.info(`KLD ${req.tag} against ${req.base} on libggml-cuda ${lib.value.sha}`);
    const ran = await this.perplexity({ ...req, target }, dir, req.tag, [
      "--kl-divergence-base",
      req.base,
      "--kl-divergence",
    ]);
    if (!ran.ok) return ran;
    const summary = readKldSummary(await fs.readText(ran.value));
    if (!summary)
      return fail(
        ExitCode.Failure,
        `no KL-divergence summary in ${ran.value}: the run stopped early`,
      );
    return ok({ dir, lib: lib.value, log: ran.value, summary });
  }

  /** llama-bench in back-to-back pairs of a and b, the order alternating pair to pair (a first in odd pairs), each pair
   *  one observation of b/a - 1 */
  async ab(req: {
    target: LabTarget;
    model: string;
    a: LabSide;
    b: LabSide;
    pairs: number;
    first: number;
    reps: number;
    extra: string[];
    run: string;
    /** each arm's rate from its repetitions: their mean unless "median" */
    stat?: ArmRate;
  }): Promise<Result<AbReport>> {
    const { shell, fs, log } = this.ports;
    const dir = await this.open(req.run);
    const sides = { a: req.a, b: req.b };
    const target = await this.locate(req.target);
    const libA = await this.loaded(target, req.a, "llama-bench");
    if (!libA.ok) return libA;
    const libB = await this.loaded(target, req.b, "llama-bench");
    if (!libB.ok) return libB;
    const drift = await runtimeDrift(this.ports.hasher, libA.value.runtime, libB.value.runtime);
    if (drift)
      return fail(
        ExitCode.Failure,
        `a and b resolve a different CUDA runtime (${drift}), so this pair would measure the runtime ` +
          `and not the engine: cuBLAS selects its GEMM algorithm by version. Give both sides the same ` +
          `runtime -- an engine directory unpacked from a raw tarball by hand carries none, where one ` +
          `installed by rig carries the archives its publisher unpacked.`,
      );
    const args = { a: req.a.args ?? [], b: req.b.args ?? [] };
    log.info(
      `a: libggml-cuda ${libA.value.sha} ${envText(req.a.env)} ${argsText(args.a)}`.trimEnd(),
    );
    log.info(
      `b: libggml-cuda ${libB.value.sha} ${envText(req.b.env)} ${argsText(args.b)}`.trimEnd(),
    );
    for (let pair = req.first; pair < req.first + req.pairs; pair++) {
      const order = pair % 2 === 1 ? (["a", "b"] as const) : (["b", "a"] as const);
      const rates: string[] = [];
      for (const side of order) {
        const argv = [
          this.bin(target, "llama-bench"),
          "-m",
          req.model,
          "-r",
          String(req.reps),
          "-o",
          "jsonl",
          ...req.extra,
          ...args[side],
        ];
        const result = await shell.run(argv, { env: this.env(target, sides[side]), cwd: dir });
        if (result.code !== 0)
          return fail(
            ExitCode.Failure,
            `llama-bench (pair ${pair}, ${side}) exit ${result.code}: ${tail(result.stderr)}`,
          );
        await fs.writeText(join(dir, `ab-${pair}-${side}.jsonl`), result.stdout);
        const rate = readBenchRate(result.stdout);
        if (!rate.ok) return rate;
        rates.push(`${side} ${rate.value.samples.map((s) => s.toFixed(1)).join(" ")}`);
      }
      log.info(`pair ${pair}: ${rates.join("; ")}`);
    }
    const stat = req.stat ?? "mean";
    const pairs = await this.pairsIn(dir, stat);
    if (!pairs.ok) return pairs;
    return ok({
      dir,
      libs: { a: libA.value, b: libB.value },
      args,
      pairs: pairs.value,
      stat,
      change: pairedChange(pairs.value.map((pair) => pair.change)),
    });
  }

  /** llama-bench under nsys (CUDA graphs traced by node), each kernel family billed its own time */
  async profile(req: {
    target: LabTarget;
    model: string;
    tag: string;
    extra: string[];
    run: string;
  }): Promise<Result<ProfileReport>> {
    const { shell, fs, log } = this.ports;
    const dir = await this.open(req.run);
    const target = await this.locate(req.target);
    const lib = await this.loaded(target, {}, "llama-bench");
    if (!lib.ok) return lib;
    const base = join(dir, req.tag);
    log.info(`nsys ${req.tag} on libggml-cuda ${lib.value.sha}`);
    const captured = await shell.run(
      [
        "nsys",
        "profile",
        "-f",
        "true",
        "-t",
        "cuda",
        "--cuda-graph-trace=node",
        "-o",
        base,
        this.bin(target, "llama-bench"),
        "-m",
        req.model,
        ...req.extra,
      ],
      { env: this.env(target, {}), cwd: dir },
    );
    await fs.writeText(`${base}.log`, captured.stdout + captured.stderr);
    if (captured.code !== 0)
      return fail(
        ExitCode.Failure,
        `nsys profile exit ${captured.code}: ${tail(captured.stderr)} (log ${base}.log)`,
      );
    const report = `${base}.nsys-rep`;
    const stats = await shell.run(
      ["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv", "--output", base, report],
      { cwd: dir },
    );
    const trace = `${base}_cuda_gpu_trace.csv`;
    if (stats.code !== 0 || !(await fs.exists(trace)))
      return fail(ExitCode.Failure, `nsys stats exit ${stats.code}: ${tail(stats.stderr)}`);
    const runs = readGpuTrace(await fs.readText(trace));
    if (!runs.ok) return runs;
    const census = ownTimeCensus(runs.value);
    await fs.writeText(`${base}.census.json`, `${JSON.stringify(census, null, 2)}\n`);
    return ok({ dir, lib: lib.value, report, trace, ...census });
  }

  /** the tree's libggml-cuda with `head`'s paths as HEAD has them (relink.ts), into <run>/lib/ */
  async relink(req: Omit<RelinkRequest, "out"> & { run: string }): Promise<Result<RelinkReport>> {
    const out = join(await this.open(req.run), "lib");
    return relinkWithHead(this.ports, { tree: req.tree, head: req.head, jobs: req.jobs, out });
  }

  private async open(run: string): Promise<string> {
    const dir = this.runDir(run);
    await this.ports.fs.mkdirp(dir);
    return dir;
  }

  /** the target with its executables' directory: the tree's bin/, or the tree itself where it has none */
  private async locate(target: LabTarget): Promise<Located> {
    const bin = join(target.tree, "bin");
    return { ...target, binDir: (await this.ports.fs.exists(bin)) ? bin : target.tree };
  }

  private bin(target: Located, tool: string): string {
    return join(target.binDir, tool);
  }

  /** the cards in PCI order, the side's library ahead of the tree's bin/ (the executables' RUNPATH yields to it), the
   *  target's switches and then the side's */
  private env(target: Located, side: Partial<LabSide>): Record<string, string> {
    const lib = side.lib ?? target.lib;
    return {
      CUDA_DEVICE_ORDER: "PCI_BUS_ID",
      CUDA_VISIBLE_DEVICES: target.cards.join(","),
      LD_LIBRARY_PATH: [lib, target.binDir].filter(Boolean).join(":"),
      ...target.env,
      ...side.env,
    };
  }

  /** the libggml-cuda `tool` loads under the side, by ldd: a library asked for and not the one loaded fails.
   *  A tool the build does not carry is named first: a published tarball holds the targets of the day it was
   *  published and none added since (engine-sm120-32e695e.tar.gz has no llama-perplexity), and a leg over it
   *  otherwise fails at the run or inside ldd, neither of which says which build is short of what. */
  private async loaded(
    target: Located,
    side: Partial<LabSide>,
    tool: string,
  ): Promise<Result<LoadedLib>> {
    const { shell, hasher } = this.ports;
    const path = this.bin(target, tool);
    if (!(await this.ports.fs.exists(path)))
      return fail(
        ExitCode.Failure,
        `${path} is not there: a build holds the targets of the day it was published and none added since (install a newer published build, or rig build --compile)`,
      );
    const ldd = await shell.run(["ldd", path], { env: this.env(target, side) });
    const loaded = ldd.stdout.match(new RegExp(`${LIB.replaceAll(".", "\\.")} => (\\S+)`))?.[1];
    if (ldd.code !== 0 || !loaded)
      return fail(
        ExitCode.Failure,
        `ldd ${path} finds no ${LIB}: ${tail(ldd.stderr || ldd.stdout)}`,
      );
    const asked = side.lib ?? target.lib;
    if (asked && dirname(loaded) !== asked.replace(/\/+$/, ""))
      return fail(ExitCode.Failure, `${tool} loads ${loaded}, not ${join(asked, LIB)}`);
    return ok({
      path: loaded,
      sha: (await hasher.sha256File(loaded)).slice(0, 12),
      runtime: runtimeOf(ldd.stdout),
    });
  }

  private async perplexity(
    req: {
      target: Located;
      model: string;
      text: string;
      ctx: number;
      chunks: number;
      extra: string[];
    },
    dir: string,
    tag: string,
    mode: string[],
  ): Promise<Result<string>> {
    const { shell, fs } = this.ports;
    const argv = [
      this.bin(req.target, "llama-perplexity"),
      "-m",
      req.model,
      "-f",
      req.text,
      "-c",
      String(req.ctx),
      "--chunks",
      String(req.chunks),
      ...mode,
      ...req.extra,
    ];
    const result = await shell.run(argv, { env: this.env(req.target, {}), cwd: dir });
    const path = join(dir, `${tag}.log`);
    await fs.writeText(path, result.stdout + result.stderr);
    if (result.code !== 0)
      return fail(
        ExitCode.Failure,
        `llama-perplexity exit ${result.code}: ${tail(result.stderr)} (log ${path})`,
      );
    return ok(path);
  }

  /** bytes that differ between two files, the longer one's tail counted whole; equal 64 MiB blocks skipped natively */
  private async differingBytes(a: string, b: string): Promise<number> {
    const { fs } = this.ports;
    const [sa, sb] = [await fs.stat(a), await fs.stat(b)];
    const sizeA = sa?.size ?? 0;
    const sizeB = sb?.size ?? 0;
    const common = Math.min(sizeA, sizeB);
    const block = 64 * 1024 * 1024;
    let differ = Math.abs(sizeA - sizeB);
    for (let at = 0; at < common; at += block) {
      const n = Math.min(block, common - at);
      const [x, y] = [await fs.readRange(a, at, n), await fs.readRange(b, at, n)];
      if (Buffer.compare(x, y) === 0) continue;
      for (let i = 0; i < n; i++) if (x[i] !== y[i]) differ++;
    }
    return differ;
  }

  /** the pairs ab-<n>-a/b.jsonl in `dir`, from 1 up to the first one missing either side */
  private async pairsIn(dir: string, stat: ArmRate): Promise<Result<AbPair[]>> {
    const { fs } = this.ports;
    const pairs: AbPair[] = [];
    for (let pair = 1; ; pair++) {
      const [pa, pb] = [join(dir, `ab-${pair}-a.jsonl`), join(dir, `ab-${pair}-b.jsonl`)];
      if (!(await fs.exists(pa)) || !(await fs.exists(pb))) break;
      const a = readBenchRate(await fs.readText(pa), stat);
      if (!a.ok) return a;
      const b = readBenchRate(await fs.readText(pb), stat);
      if (!b.ok) return b;
      pairs.push({
        pair,
        a: a.value.rate,
        b: b.value.rate,
        change: b.value.rate / a.value.rate - 1,
      });
    }
    return ok(pairs);
  }
}

const envText = (env: Record<string, string>) =>
  Object.entries(env)
    .map(([k, v]) => `${k}=${v}`)
    .join(" ") || "(no switches)";

const argsText = (args: string[]) => (args.length > 0 ? `args ${args.join(" ")}` : "");

const tail = (text: string) => text.trim().split("\n").slice(-3).join(" | ");
