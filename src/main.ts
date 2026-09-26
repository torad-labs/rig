// The composition root: the one place that knows every feature, every adapter and how they plug
// together. Features never import each other's internals — what one needs from another (`up` from the steps,
// `unit` from serve's plan, `describe` from unit's status) is an interface the consumer declares
// and this file satisfies. Everything below this line is wiring; nothing here decides anything.
import { dirname } from "node:path";
import { version } from "../package.json" with { type: "json" };
import { BuildEngine, buildEngineCommand } from "./features/engine-build/index.ts";
import { gpuRentalCommand, RentGpu } from "./features/gpu-rental/index.ts";
import { BringUpHead, bringUpHeadCommand } from "./features/head-bringup/index.ts";
import { DescribeHead, describeHeadCommand } from "./features/head-description/index.ts";
import { allProbes, RunGates, runGatesCommand } from "./features/head-gating/index.ts";
import { ServeHead, serveHeadCommand, verifyHeadCommand } from "./features/head-serving/index.ts";
import { CheckMachine, checkMachineCommand } from "./features/machine-check/index.ts";
import { DerivePack, derivePackCommand } from "./features/pack-derivation/index.ts";
import { DownloadPack, downloadPackCommand } from "./features/pack-download/index.ts";
import { ManageUnit, manageUnitCommand } from "./features/systemd-unit/index.ts";
import { parseArgs, UsageError, unknownFlags } from "./shared/cli/args.ts";
import type { Command, LoadHead } from "./shared/cli/command.ts";
import { type Engine, loadEngine } from "./shared/engine/engine.ts";
import { listHeads, loadHead } from "./shared/head/head.ts";
import { type Layout, layoutAt } from "./shared/layout.ts";
import { realPorts } from "./shared/platform/index.ts";
import type { Ports } from "./shared/ports/index.ts";
import { ExitCode, fail } from "./shared/result.ts";

/** A compiled binary runs from /$bunfs; a checkout runs this file under bun. */
const compiled = import.meta.dir.startsWith("/$bunfs");

/** The repo root — the directory holding heads/ and engine/: above src/ in a checkout, above
 *  dist/ for the compiled binary; RIG_ROOT overrides both (a binary copied elsewhere). */
const findRoot = () =>
  process.env.RIG_ROOT ??
  (compiled ? dirname(dirname(process.execPath)) : dirname(import.meta.dir));

/** How to invoke this program again (ExecStartPre in a unit): the binary, or bun + this file. */
const selfCommand = () => (compiled ? [process.execPath] : [process.execPath, import.meta.path]);

/** every feature that reads the pin, wired: all the commands but `vast`, and the gates `vast
 *  bench` runs */
function wireEngine(
  ports: Ports,
  layout: Layout,
  engine: Engine,
  head: LoadHead,
): { commands: Command[]; gate: RunGates } {
  const prepare = new CheckMachine(ports, engine);
  const build = new BuildEngine(ports, layout, engine);
  const fetch = new DownloadPack(ports);
  const derive = new DerivePack(ports);
  const serve = new ServeHead(ports, engine);
  const unit = new ManageUnit({ ...ports, planner: serve, self: selfCommand() }, layout);
  const describe = new DescribeHead({ ...ports, unit }, layout, engine);
  const gate = new RunGates(ports, layout, engine, allProbes);
  const up = new BringUpHead({
    ...ports,
    steps: {
      prepare: (options) => prepare.run(options),
      room: (head, options) => prepare.room(head, options),
      fetch: (head) => fetch.run(head),
      build: (options) => build.run(options),
      derive: (head) => derive.run(head),
      installUnit: (head, options) => unit.install(head, options),
    },
  });
  return {
    gate,
    commands: [
      checkMachineCommand(prepare, ports.log),
      buildEngineCommand(build, ports.log),
      downloadPackCommand(fetch, head, ports.log),
      derivePackCommand(derive, head, ports.log),
      verifyHeadCommand(serve, head, ports.log),
      serveHeadCommand(serve, head, ports.log),
      manageUnitCommand(unit, head, ports.log),
      describeHeadCommand(describe, head, () => listHeads(ports.fs, layout), ports.log),
      bringUpHeadCommand(up, head, ports.log),
      runGatesCommand(gate, head, ports.log),
    ],
  };
}

async function main(argv: string[]): Promise<number> {
  if (argv[0] === "--version" || argv[0] === "version") {
    console.log(`rig ${version}`);
    return 0;
  }
  const ports = realPorts();
  const root = findRoot();
  const layout = layoutAt(root);
  const engine = await loadEngine(ports.fs, layout);
  const [name, ...rest] = argv;
  // `vast` alone runs on an engine.toml this binary cannot read: its down, status and idle-check
  // read no pin, and the idle timer runs idle-check from this checkout (RentGpu's constructor).
  if (!engine.ok && name !== "vast") {
    ports.log.error(engine.message);
    return engine.code;
  }
  const head = (name: string) => loadHead(ports.fs, layout, name);

  const wired = engine.ok ? wireEngine(ports, layout, engine.value, head) : null;
  const vast = new RentGpu(
    {
      ...ports,
      gate: {
        run: (head, options) =>
          wired
            ? wired.gate.run(head, options)
            : Promise.resolve(fail(ExitCode.Failure, "no engine to gate with")),
      },
      self: selfCommand(),
      home: process.env.HOME ?? "",
    },
    layout,
    engine,
  );
  const commands: Command[] = [...(wired?.commands ?? []), gpuRentalCommand(vast, head, ports.log)];

  const usage = () => {
    console.error(
      `rig — builds torad-labs/llama.cpp per GPU and brings model heads up (root ${root})\n\nusage: rig <command> [args]\n${commands.map((command) => `  ${command.usage}`).join("\n")}\n\nexit codes: 0 ok, 1 failure (named), 2 busy (a slot is processing), 3 unsupported card, 4 driver older than the toolkit or the prebuilt's CUDA runtime, 64 usage`,
    );
  };
  if (!name || name === "help" || name === "--help" || name === "-h") {
    usage();
    return name ? 0 : 64;
  }
  const cmd = commands.find((command) => command.name === name);
  if (!cmd) {
    console.error(`rig: unknown command ${JSON.stringify(name)}`);
    usage();
    return 64;
  }
  // --help before anything runs: `build --help` would otherwise compile
  if (rest.includes("--help") || rest.includes("-h")) {
    console.error(`usage: rig ${cmd.usage}`);
    return 0;
  }
  try {
    const args = parseArgs(rest);
    if (args.dashed.length > 0)
      throw new UsageError(
        `${cmd.name} does not take ${args.dashed.join(", ")}: rig's flags take two dashes (--gpu N), and llama-server's own go after --`,
      );
    const unknown = unknownFlags(args, cmd.usage);
    if (unknown.length > 0)
      throw new UsageError(
        `${cmd.name} does not take ${unknown.map((flag) => `--${flag}`).join(", ")} (llama-server's own flags go after --)`,
      );
    return await cmd.run(args);
  } catch (error) {
    if (!(error instanceof UsageError)) throw error;
    console.error(`rig: ${error.message}\nusage: rig ${cmd.usage}`);
    return ExitCode.Usage;
  }
}

process.exitCode = await main(process.argv.slice(2));
