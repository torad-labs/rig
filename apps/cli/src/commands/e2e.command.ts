import type { Log } from "@rig/core";
import type { DriverOnlyGate, DriverOnlyGateReport } from "@rig/engine";
import { type Args, flagInt, flagStr, UsageError } from "../cli/args.ts";
import { type Command, reportLine } from "../cli/command.ts";

const USAGE =
  "e2e <pack.gguf> [--prebuilt TARBALL] [--gpu N] [--base IMAGE] [--head NAME] [--json]   the driver-only gate: this checkout packed as release.yml packs it and installed by install.sh in a fresh container (--base, ubuntu:22.04 unless given) with card N (0 unless given) and only the NVIDIA driver, then `rig prepare` must find a prebuilt and no toolkit, `rig build` must install it, and llama-bench must decode the pack on the card; --prebuilt gates a local tarball engine.toml pins by name (before it is published) instead of the pinned URL; --head NAME with that head's source pack also adopts it and derives it as a machine without Torad's private assets would; RIG_GATE_LOCK names a lock file flock holds around the decode";

/** a flag that takes a value: a bare `--prebuilt` is a usage error, not the default */
function flagValue(args: Args, name: string): string | undefined {
  if (args.flags[name] === true) throw new UsageError(`--${name} takes a value`);
  return flagStr(args, name);
}

export function e2eCommand(gate: DriverOnlyGate, lock: string | undefined, log: Log): Command {
  return {
    name: "e2e",
    usage: USAGE,
    async run(args: Args) {
      const [pack, ...extra] = args.positionals;
      if (pack === undefined || extra.length > 0)
        throw new UsageError("e2e takes exactly one pack.gguf");
      const result = await gate.run({
        pack,
        prebuilt: flagValue(args, "prebuilt"),
        gpu: flagInt(args, "gpu") ?? 0,
        base: flagValue(args, "base") ?? "ubuntu:22.04",
        head: flagValue(args, "head"),
        lock,
      });
      return reportLine(log, args, result, describe);
    },
  };
}

function describe(report: DriverOnlyGateReport): string {
  const engine = report.prebuilt ? `the local ${report.prebuilt}` : "the pinned prebuilt";
  return `driver-only gate passed in ${report.base} with ${engine} (${report.log})`;
}
