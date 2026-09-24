import { type Args, flagBool, flagInt, flagStr } from "../../shared/cli/args.ts";
import { type Command, reportLine } from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import type { BuildEngine, BuildEngineReport } from "./engine-build.service.ts";

const USAGE =
  "build [--gpu N] [--compile] [--portable] [--jobs N] [--from-tarball FILE] [--allow-arch CAP] [--json]   the engine at its pin, for this card → local/engine-builds/<sha7>-sm<cap>/ (the published prebuilt where one is pinned; --compile builds from source)";

export function buildEngineCommand(service: BuildEngine, log: Log): Command {
  return {
    name: "build",
    usage: USAGE,
    async run(args: Args) {
      const result = await service.run({
        gpu: flagInt(args, "gpu") ?? 0,
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

function describe(report: BuildEngineReport): string {
  const state = report.alreadyBuilt ? "already built" : "ready";
  const tarball = report.tarball ? `; portable tarball ${report.tarball}` : "";
  return `${state}: ${report.dir} (${report.marker})${tarball}`;
}
