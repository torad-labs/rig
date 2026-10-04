import { join, resolve } from "node:path";
import type { CardLease, Log, Result } from "@rig/core";
import { ExitCode } from "@rig/core";
import type {
  AbReport,
  EngineLab,
  IdentityReport,
  KldReport,
  LabSide,
  LabTarget,
  OpsTestReport,
  ProfileReport,
  RelinkReport,
} from "@rig/engine";
import { type Args, flagBool, flagInt, flagStr, UsageError } from "../cli/args.ts";
import { type Command, printJson } from "../cli/command.ts";

const USAGE =
  "engine test --ops OP,OP [--legacy K=V,...] [--filter CASE] | ident --model GGUF --text FILE --tag NAME [--ref KLD] [--ctx N] [--chunks N] | kld --model GGUF --text FILE --base KLD --tag NAME [--ctx N] [--chunks N] | ab --model GGUF [--a-lib DIR] [--a-env K=V,...] [--a-args 'ARGS'] [--b-lib DIR] [--b-env K=V,...] [--b-args 'ARGS'] [--pairs N] [--first N] [--reps N] [--median] | profile --model GGUF --tag NAME [--top N] | relink --head PATH,... [--jobs N]; each with --tree NAME|DIR [--run NAME] [--json], and all but relink with --gpu 0,1 [--lib DIR] [--env K=V,...] [--eta MIN] [--max-hold MIN] [--held]; the engine tool's own arguments after --   an engine change measured on a cmake tree (local/engine-build-trees/<name>), its evidence in local/engine-lab/<run>/: test-backend-ops per card, by default and under --legacy's switches; ident: the KLD base file byte for byte against --ref; kld: the KL divergence from --base; ab: llama-bench in back-to-back pairs of a and b, the order alternating, with the change's 95 % interval, each arm's rate the mean of its repetitions after the first or with --median their median (--a-args and --b-args are llama-bench arguments one side adds to the ones after --, how two sides of one engine differ in a runtime setting: --b-args '-ctk q8_0 -ctv q8_0'); profile: llama-bench under nsys, each kernel billed its own time; relink: the tree's libggml-cuda with --head's paths as HEAD has them, into <run>/lib (a peer's uncommitted edit left out). A run holds its cards once through the machine's card lease (--held: already under it)";
const FORM = USAGE.split("   ")[0] ?? USAGE;

const COMMON = ["tree", "run", "json"];
const CARDS = ["gpu", "lib", "env", "eta", "max-hold", "held"];
// each run's expected minutes (the card queue's --eta; --eta overrides) and the longest it may hold the cards
const SUBCOMMANDS: Record<
  string,
  { flags: readonly string[]; etaMin: number; maxHoldMin: number }
> = {
  test: { flags: [...COMMON, ...CARDS, "ops", "legacy", "filter"], etaMin: 15, maxHoldMin: 30 },
  ident: {
    flags: [...COMMON, ...CARDS, "model", "text", "tag", "ref", "ctx", "chunks"],
    etaMin: 3,
    maxHoldMin: 20,
  },
  kld: {
    flags: [...COMMON, ...CARDS, "model", "text", "base", "tag", "ctx", "chunks"],
    etaMin: 3,
    maxHoldMin: 30,
  },
  ab: {
    flags: [
      ...COMMON,
      ...CARDS,
      "model",
      "a-lib",
      "a-env",
      "a-args",
      "b-lib",
      "b-env",
      "b-args",
      "pairs",
      "first",
      "reps",
      "median",
    ],
    etaMin: 10,
    maxHoldMin: 120,
  },
  profile: { flags: [...COMMON, ...CARDS, "model", "tag", "top"], etaMin: 3, maxHoldMin: 30 },
  relink: { flags: [...COMMON, "head", "jobs"], etaMin: 0, maxHoldMin: 0 },
};

export interface EngineLabWiring {
  lab: EngineLab;
  lease: CardLease;
  /** how to run this program again: the binary, or bun and main.ts */
  self: readonly string[];
  /** local/engine-build-trees, where --tree NAME lives */
  treesDir: string;
  now: () => number;
  log: Log;
}

export function engineLabCommand(wiring: EngineLabWiring): Command {
  const { lab, lease, self, treesDir, now, log } = wiring;
  return {
    name: "engine",
    usage: USAGE,
    async run(args: Args) {
      const [sub, ...extra] = args.positionals;
      const spec = SUBCOMMANDS[sub ?? ""];
      if (!spec) {
        log.error(`usage: rig ${FORM}`);
        return ExitCode.Usage;
      }
      const other = Object.keys(args.flags).filter((flag) => !spec.flags.includes(flag));
      if (other.length > 0)
        throw new UsageError(
          `engine ${sub} does not take ${other.map((flag) => `--${flag}`).join(", ")}`,
        );
      const tree = treePath(required(args, "tree"), treesDir);
      const run = flagStr(args, "run") ?? `${sub}-${stamp(now())}`;

      if (sub === "relink") {
        const head = list(required(args, "head"));
        const result = await lab.relink({ tree, head, jobs: flagInt(args, "jobs") ?? 8, run });
        return show(log, args, result, describeRelink);
      }

      const cards = cardList(required(args, "gpu"));
      // the whole run under one hold of its cards: this program again, inside the lease, the run's name fixed here
      if (!flagBool(args, "held")) {
        const argv = [
          ...self,
          "engine",
          sub as string,
          ...flagArgv(args, run),
          "--held",
          "--",
          ...extra,
        ];
        const result = await lease.run(
          cards,
          {
            label: `rig engine ${sub} ${run}`,
            etaMin: flagInt(args, "eta") ?? spec.etaMin,
            maxHoldMin: flagInt(args, "max-hold") ?? spec.maxHoldMin,
          },
          argv,
          {
            onLine: (line, stream) =>
              stream === "stdout" ? console.log(line) : console.error(line),
          },
        );
        return result.code;
      }

      const target: LabTarget = {
        tree,
        ...(flagStr(args, "lib") ? { lib: path(args, "lib") } : {}),
        env: envFlag(args, "env"),
        cards,
      };
      switch (sub) {
        case "test": {
          const legacy = flagStr(args, "legacy");
          const result = await lab.opsTest({
            target,
            ops: list(required(args, "ops")),
            legacy: legacy === undefined ? null : envFlag(args, "legacy"),
            ...(flagStr(args, "filter") ? { filter: flagStr(args, "filter") as string } : {}),
            run,
          });
          return show(log, args, result, describeTests, (report) =>
            report.failed > 0 ? ExitCode.Failure : 0,
          );
        }
        case "ident": {
          const result = await lab.identity({
            target,
            model: path(args, "model", true),
            text: path(args, "text", true),
            tag: required(args, "tag"),
            ...(flagStr(args, "ref") ? { ref: path(args, "ref") } : {}),
            ctx: flagInt(args, "ctx") ?? 2048,
            chunks: flagInt(args, "chunks") ?? 8,
            extra,
            run,
          });
          return show(log, args, result, describeIdentity, (report) =>
            report.ref && !report.ref.identical ? ExitCode.Failure : 0,
          );
        }
        case "kld": {
          const result = await lab.kld({
            target,
            model: path(args, "model", true),
            text: path(args, "text", true),
            base: path(args, "base", true),
            tag: required(args, "tag"),
            ctx: flagInt(args, "ctx") ?? 2048,
            chunks: flagInt(args, "chunks") ?? 8,
            extra,
            run,
          });
          return show(log, args, result, describeKld);
        }
        case "ab": {
          const side = (name: "a" | "b"): LabSide => ({
            ...(flagStr(args, `${name}-lib`) ? { lib: path(args, `${name}-lib`) } : {}),
            env: envFlag(args, `${name}-env`),
            ...(flagStr(args, `${name}-args`) ? { args: words(args, `${name}-args`) } : {}),
          });
          const result = await lab.ab({
            target,
            model: path(args, "model", true),
            a: side("a"),
            b: side("b"),
            pairs: positive(args, "pairs") ?? 6,
            first: positive(args, "first") ?? 1,
            reps: positive(args, "reps") ?? 3,
            extra,
            run,
            stat: flagBool(args, "median") ? "median" : "mean",
          });
          return show(log, args, result, describeAb);
        }
        default: {
          const top = positive(args, "top") ?? 15;
          const result = await lab.profile({
            target,
            model: path(args, "model", true),
            tag: required(args, "tag"),
            extra,
            run,
          });
          return show(log, args, result, (report) => describeProfile(report, top));
        }
      }
    },
  };
}

/** a failure logged as its exit code; a report shown as JSON with --json, else as its lines, its exit code `code`'s */
function show<T>(
  log: Log,
  args: Args,
  result: Result<T>,
  describe: (value: T) => string[],
  code: (value: T) => number = () => 0,
): number {
  if (!result.ok) {
    log.error(result.message);
    return result.code;
  }
  if (flagBool(args, "json")) printJson(result.value);
  else for (const line of describe(result.value)) log.info(line);
  return code(result.value);
}

const pct = (x: number) => `${x >= 0 ? "+" : ""}${(100 * x).toFixed(2)} %`;
const ms = (ns: number) => `${(ns / 1e6).toFixed(1)} ms`;

function describeTests(report: OpsTestReport): string[] {
  const failed = report.runs.filter(
    (run) => run.code !== 0 || !run.verdict || run.verdict.passed < run.verdict.total,
  );
  return [
    `${report.runs.length - failed.length}/${report.runs.length} runs passed every case on libggml-cuda ${report.lib.sha} (${report.dir})`,
    ...failed.map(
      (run) =>
        `FAILED card ${run.card} ${run.mode} ${run.op}: ${run.verdict ? `${run.verdict.passed}/${run.verdict.total}` : `exit ${run.code}`} (${run.log})`,
    ),
  ];
}

function describeIdentity(report: IdentityReport): string[] {
  const line = `${report.kld}: ${report.bytes} bytes, sha256 ${report.sha256.slice(0, 12)}, libggml-cuda ${report.lib.sha}`;
  if (!report.ref) return [line];
  return [
    line,
    report.ref.identical
      ? `identical to ${report.ref.path}`
      : `DIFFERS from ${report.ref.path} in ${report.ref.differingBytes} bytes`,
  ];
}

function describeKld(report: KldReport): string[] {
  const s = report.summary;
  return [
    `mean KLD ${s.meanKld} ± ${s.meanKldErr}, same top p ${s.sameTopP} ± ${s.sameTopPErr} %${s.pplRatio === null ? "" : `, PPL ratio ${s.pplRatio}`} (libggml-cuda ${report.lib.sha}, ${report.log})`,
  ];
}

const sideArgs = (args: string[]) => (args.length > 0 ? ` with ${args.join(" ")}` : "");

function describeAb(report: AbReport): string[] {
  const c = report.change;
  return [
    `a: libggml-cuda ${report.libs.a.sha}${sideArgs(report.args.a)}, b: libggml-cuda ${report.libs.b.sha}${sideArgs(report.args.b)} (${report.dir})`,
    ...(report.stat === "median" ? ["each arm the median of its repetitions after the first"] : []),
    ...report.pairs.map(
      (pair) =>
        `pair ${pair.pair}: a ${pair.a.toFixed(1)}, b ${pair.b.toFixed(1)} t/s, ${pct(pair.change)}`,
    ),
    c.interval
      ? `${c.n} pairs: b ${pct(c.mean)} against a, 95 % CI ${pct(c.interval.lo)} to ${pct(c.interval.hi)} (t ${c.interval.t})`
      : `${c.n} pair: b ${pct(c.mean)} against a`,
  ];
}

function describeProfile(report: ProfileReport, top: number): string[] {
  return [
    `own kernel time ${ms(report.ownNs)} (nsys durations ${ms(report.nsysNs)}, ${((100 * (report.nsysNs - report.ownNs)) / report.nsysNs).toFixed(1)} % of them waits) on libggml-cuda ${report.lib.sha}`,
    ...report.kernels
      .slice(0, top)
      .map(
        (k) =>
          `${((100 * k.ownNs) / report.ownNs).toFixed(2).padStart(6)} %  ${ms(k.ownNs).padStart(10)}  ${String(k.launches).padStart(7)}x  ${k.kernel}`,
      ),
    `trace ${report.trace}`,
  ];
}

function describeRelink(report: RelinkReport): string[] {
  return [
    `${report.lib}: sha256 ${report.sha256.slice(0, 12)}, ${report.mirrored.length} objects compiled with HEAD's sources`,
  ];
}

function required(args: Args, name: string): string {
  const value = flagStr(args, name);
  if (!value) throw new UsageError(`engine needs --${name}`);
  return value;
}

/** a path flag, absolute; `exists` is checked by the run that reads it */
const path = (args: Args, name: string, isRequired = false) =>
  resolve(isRequired ? required(args, name) : (flagStr(args, name) as string));

/** a name is a tree under local/engine-build-trees, anything with a slash a directory */
const treePath = (value: string, treesDir: string) =>
  value.includes("/") ? resolve(value) : join(treesDir, value);

const list = (value: string) => value.split(",").filter(Boolean);

/** a flag's value as the arguments of a program: split on whitespace, no quoting (a value holding none to protect) */
const words = (args: Args, name: string) => (flagStr(args, name) as string).trim().split(/\s+/);

function cardList(value: string): number[] {
  if (!/^\d+(,\d+)*$/.test(value))
    throw new UsageError(`--gpu takes card indices (0 or 0,1), not ${JSON.stringify(value)}`);
  const cards = value.split(",").map(Number);
  if (new Set(cards).size !== cards.length)
    throw new UsageError(`--gpu names a card twice: ${value}`);
  return cards;
}

/** K=V,K=V as assignments; none when absent */
function envFlag(args: Args, name: string): Record<string, string> {
  const value = flagStr(args, name);
  if (!value) return {};
  const env: Record<string, string> = {};
  for (const pair of value.split(",")) {
    const eq = pair.indexOf("=");
    if (eq <= 0)
      throw new UsageError(
        `--${name} takes K=V assignments, comma-separated, not ${JSON.stringify(pair)}`,
      );
    env[pair.slice(0, eq)] = pair.slice(eq + 1);
  }
  return env;
}

function positive(args: Args, name: string): number | undefined {
  const value = flagInt(args, name);
  if (value !== undefined && value < 1)
    throw new UsageError(`--${name} takes a whole number above zero`);
  return value;
}

/** the flags as given, the run's name fixed and every path absolute (the run under the lease starts where this one
 *  did, but nothing should hang on that) */
function flagArgv(args: Args, run: string): string[] {
  const paths = new Set(["lib", "model", "text", "ref", "base", "a-lib", "b-lib"]);
  const argv: string[] = [];
  for (const [name, value] of Object.entries(args.flags)) {
    if (name === "run") continue;
    if (value === true) argv.push(`--${name}`);
    else if (typeof value === "string")
      argv.push(
        `--${name}=${paths.has(name) || (name === "tree" && value.includes("/")) ? resolve(value) : value}`,
      );
  }
  return [...argv, `--run=${run}`];
}

/** a run's default name: the UTC minute, 20261001T2104Z */
const stamp = (ms: number) => `${new Date(ms).toISOString().slice(0, 16).replace(/[-:]/g, "")}Z`;
