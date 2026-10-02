import type { Log } from "@rig/core";
import type { BuildEngine, BuildEngineReport } from "@rig/engine";
import { type Args, flagBool, flagInt, flagStr, UsageError } from "../cli/args.ts";
import { type Command, reportLine } from "../cli/command.ts";

const USAGE =
  "build [<head>] [--gpu N | --cap CAP] [--compile] [--portable] [--jobs N] [--from-tarball FILE] [--allow-arch CAP] [--json]   the engine at its pin (the head's own, when it names one), for this card or --cap → local/engine-builds/<sha7>-sm<cap>/ (the published prebuilt where one is pinned; --compile builds from source)";

export function buildEngineCommand(service: BuildEngine, log: Log): Command {
  return {
    name: "build",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    async run(args: Args) {
      const result = await service.run({
        gpu: flagInt(args, "gpu") ?? 0,
        cap: flagCap(args),
        compile: flagBool(args, "compile"),
        portable: flagBool(args, "portable"),
        jobs: flagInt(args, "jobs"),
        fromTarball: flagStr(args, "from-tarball"),
        allowArch: flagStr(args, "allow-arch"),
      });
      return reportLine(log, args, result, describe);
    },
  };
}

/** --cap as a compute capability with the dot removed (120); anything else is a usage error */
function flagCap(args: Args): string | undefined {
  const cap = flagStr(args, "cap");
  if (cap !== undefined && !/^\d{2,3}$/.test(cap))
    throw new UsageError(
      `--cap takes a compute capability with the dot removed, like 120, not ${JSON.stringify(cap)}`,
    );
  return cap;
}

function describe(report: BuildEngineReport): string {
  const state = report.alreadyBuilt ? "already built" : "ready";
  const tarball = report.tarball ? `; portable tarball ${report.tarball}` : "";
  return `${state}: ${report.dir} (${report.marker})${tarball}`;
}
