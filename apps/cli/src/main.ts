// The composition root: the one place that knows every package, every adapter and how they plug
// together. Packages never import each other's internals — what one needs from another (`up` from the
// steps, `unit` from serve's plan, `describe` from unit's status) is an interface the consumer declares
// and this file satisfies. Everything below this line is wiring; nothing here decides anything.
import { dirname, resolve } from "node:path";
import { realPorts } from "@rig/adapters";
import {
  BUILT_FROM,
  ExitCode,
  fail,
  type Layout,
  layoutAt,
  type Ports,
  type Result,
} from "@rig/core";
import {
  BuildEngine,
  BuildPrebuilt,
  DriverOnlyGate,
  type Engine,
  EngineLab,
  headEngine,
  loadEngine,
  TagRelease,
} from "@rig/engine";
import { allProbes, RunGates } from "@rig/gate";
import { listHeads, loadHead } from "@rig/head";
import { BuildImage, type RegistryCredentials, registryCredentials } from "@rig/image";
import { CheckMachine, PREBUILT_APT_PACKAGES } from "@rig/machine";
import { DerivePack, DownloadPack } from "@rig/pack";
import { GuardBox, PublishTemplate, RentGpu, SweepStopped } from "@rig/rental";
import { BringUpHead, DescribeHead, ManageUnit, ServeHead } from "@rig/serve";
import { version } from "../package.json" with { type: "json" };
import { parseArgs, UsageError, unknownFlags } from "./cli/args.ts";
import type { Command, LoadHead } from "./cli/command.ts";
import { buildEngineCommand } from "./commands/build.command.ts";
import { derivePackCommand } from "./commands/derive.command.ts";
import { describeHeadCommand } from "./commands/describe.command.ts";
import { e2eCommand } from "./commands/e2e.command.ts";
import { engineLabCommand } from "./commands/engine.command.ts";
import { downloadPackCommand } from "./commands/fetch.command.ts";
import { runGatesCommand } from "./commands/gate.command.ts";
import { buildImageCommand, type PublishAfterPush } from "./commands/image.command.ts";
import { checkMachineCommand } from "./commands/prepare.command.ts";
import { serveHeadCommand } from "./commands/serve.command.ts";
import { tagCommand } from "./commands/tag.command.ts";
import { manageUnitCommand } from "./commands/unit.command.ts";
import { bringUpHeadCommand } from "./commands/up.command.ts";
import { gpuRentalCommand, publishTemplate } from "./commands/vast.command.ts";
import { verifyHeadCommand } from "./commands/verify.command.ts";

/** The tree this binary was built from, which tools/build.ts defines at the compile: none run from the source. */
declare const RIG_BUILT_FROM: string | undefined;

/** A compiled binary runs from /$bunfs; a checkout runs this file under bun. */
const compiled = import.meta.dir.startsWith("/$bunfs");

/** The repo root — the directory holding heads/ and engine/: three above this file's src/ in a
 *  checkout (apps/cli/src), above dist/ for the compiled binary; RIG_ROOT overrides both (a binary
 *  copied elsewhere). */
const findRoot = () =>
  process.env.RIG_ROOT ??
  (compiled ? dirname(dirname(process.execPath)) : resolve(import.meta.dir, "../../.."));

/** How to invoke this program again (ExecStartPre in a unit): the binary, or bun + this file. */
const selfCommand = () => (compiled ? [process.execPath] : [process.execPath, import.meta.path]);

/** every feature that reads the pin, wired: all the commands but `vast`, and the gates `vast
 *  bench` runs */
function wireEngine(
  ports: Ports,
  layout: Layout,
  engine: Engine,
  head: LoadHead,
  credentials: () => Promise<RegistryCredentials>,
  publish: PublishAfterPush,
): { commands: Command[]; gate: RunGates } {
  const prepare = new CheckMachine(ports, engine);
  const build = new BuildEngine(ports, layout, engine);
  const prebuilt = new BuildPrebuilt(ports, layout, engine);
  const driverOnly = new DriverOnlyGate(ports, layout);
  const fetch = new DownloadPack(ports);
  const derive = new DerivePack(ports);
  const serve = new ServeHead(ports, engine);
  const unit = new ManageUnit({ ...ports, planner: serve, self: selfCommand() }, layout);
  const describe = new DescribeHead({ ...ports, unit }, layout, engine);
  const gate = new RunGates(ports, layout, engine, allProbes);
  const image = new BuildImage(ports, layout, engine, PREBUILT_APT_PACKAGES, credentials);
  const up = new BringUpHead({
    ...ports,
    steps: {
      unitDevices: (head) => unit.devices(head),
      room: (head, options) => prepare.room(head, options),
      prepare: (options) => prepare.run(options),
      fetch: (head) => fetch.run(head),
      build: (options) => build.run(options),
      derive: (head) => derive.run(head),
      installUnit: (head, options) => unit.install(head, options),
      serve: (head, options) => serve.serve(head, options),
    },
  });
  return {
    gate,
    commands: [
      checkMachineCommand(prepare, ports.log),
      buildEngineCommand(build, prebuilt, ports.log),
      e2eCommand(driverOnly, process.env.RIG_GATE_LOCK, ports.log),
      downloadPackCommand(fetch, head, ports.log),
      derivePackCommand(derive, head, ports.log),
      verifyHeadCommand(serve, head, ports.log),
      serveHeadCommand(serve, head, ports.log),
      manageUnitCommand(unit, head, ports.log),
      describeHeadCommand(describe, head, () => listHeads(ports.fs, layout), ports.log),
      bringUpHeadCommand(up, head, ports.log),
      runGatesCommand(gate, head, ports.log),
      buildImageCommand(image, head, ports.log, publish),
    ],
  };
}

/** every command, wired to `engine`: the pin engine.toml names, or the one a head names for itself */
function commandsFor(
  ports: Ports,
  layout: Layout,
  engine: Result<Engine>,
  head: LoadHead,
): Command[] {
  // the registry's keys from the environment where set, else the keyring, read only by a push or a template
  const credentials = () => registryCredentials(ports.secrets, process.env);
  const template = engine.ok
    ? new PublishTemplate(
        { ...ports, pullKey: async () => (await credentials()).pullKey },
        layout,
        engine.value,
      )
    : null;
  const sweep = new SweepStopped({ ...ports, self: selfCommand() }, layout);
  const publish: PublishAfterPush = template
    ? (head) => publishTemplate(template, sweep, head, { dryRun: false })
    : null;
  const wired = engine.ok
    ? wireEngine(ports, layout, engine.value, head, credentials, publish)
    : null;
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
      vastai: Bun.which("vastai") ?? "vastai",
      home: process.env.HOME ?? "",
    },
    layout,
    engine,
  );
  // on a box rented from a template: vast gives every instance these to stop itself with
  const guard = new GuardBox({
    ...ports,
    box: { id: process.env.CONTAINER_ID, apiKey: process.env.CONTAINER_API_KEY },
  });
  const lab = engineLabCommand({
    lab: new EngineLab(ports, layout),
    lease: ports.cardLease,
    self: selfCommand(),
    treesDir: layout.engineBuildTreesDir,
    now: () => ports.clock.now(),
    log: ports.log,
  });
  return [
    ...(wired?.commands ?? []),
    gpuRentalCommand(vast, template, guard, sweep, head, ports.log),
    lab,
    tagCommand(new TagRelease(ports, layout), ports.log),
  ];
}

async function main(argv: string[]): Promise<number> {
  if (argv[0] === "--version" || argv[0] === "version") {
    console.log(`rig ${version}`);
    return 0;
  }
  // read by `rig vast up` and `lab` before they ship this binary to a box
  if (argv[0] === BUILT_FROM.flag) {
    if (typeof RIG_BUILT_FROM !== "string") {
      console.error(
        "rig: this rig was not compiled by bun run build, so it does not know its tree",
      );
      return 1;
    }
    console.log(RIG_BUILT_FROM);
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
  const commands = commandsFor(ports, layout, engine, head);

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
    // a head with its own pin runs every step on it: the build, the serve, the gates, the rented box
    const named = engine.ok ? cmd.headName?.(args) : undefined;
    const own = named === undefined ? undefined : await head(named);
    if (engine.ok && own?.ok && own.value.engine) {
      const pinned = headEngine(engine.value, own.value.engine, layout);
      if (!pinned.ok) {
        ports.log.error(`${own.value.name}: ${pinned.message}`);
        return pinned.code;
      }
      const onPin = commandsFor(ports, layout, pinned, head).find(
        (command) => command.name === name,
      );
      if (onPin) return await onPin.run(args);
    }
    return await cmd.run(args);
  } catch (error) {
    if (!(error instanceof UsageError)) throw error;
    console.error(`rig: ${error.message}\nusage: rig ${cmd.usage}`);
    return ExitCode.Usage;
  }
}

process.exitCode = await main(process.argv.slice(2));
