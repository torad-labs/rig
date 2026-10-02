import type { Log } from "@rig/core";
import type {
  BuildEngine,
  BuildEngineReport,
  BuildPrebuilt,
  BuildPrebuiltReport,
} from "@rig/engine";
import { type Args, flagBool, flagInt, flagStr, UsageError } from "../cli/args.ts";
import { type Command, reportLine } from "../cli/command.ts";

const USAGE =
  "build [<head>] [--gpu N | --cap CAP] [--compile] [--portable] [--jobs N] [--from-tarball FILE [--off-pin NAME]] [--allow-arch CAP] [--prebuilt [--sha FORK_SHA]] [--json]   the engine at its pin (the head's own, when it names one), for this card or --cap → local/engine-builds/<sha7>-sm<cap>/ (the published prebuilt where one is pinned; --compile builds from source); --off-pin installs a tarball of another commit for the lab → local/engine-lab/<NAME>/, with the pin's CUDA runtime beside it; --prebuilt builds the release's prebuilt of engine.toml's pin (or --sha's commit) with --portable in tools/prebuilt's image, on its glibc 2.35 floor, capped at 14 GiB and 6 CPUs (--jobs 6 unless given) → local/prebuilt/engine-builds/";

/** the flags of a build on this machine, which a build in the release image does not take */
const HOST_ONLY = ["cap", "compile", "portable", "from-tarball", "off-pin", "allow-arch"] as const;

export function buildEngineCommand(
  service: BuildEngine,
  prebuilt: BuildPrebuilt,
  log: Log,
): Command {
  return {
    name: "build",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    async run(args: Args) {
      if (flagBool(args, "prebuilt")) {
        const head = args.positionals[0];
        if (head !== undefined)
          throw new UsageError(
            `--prebuilt builds engine.toml's pin or --sha's commit, not ${head}'s own: name the commit with --sha`,
          );
        const host = HOST_ONLY.find((name) => args.flags[name] !== undefined);
        if (host)
          throw new UsageError(
            `--prebuilt builds in the release image and takes --gpu, --jobs and --sha, not --${host}`,
          );
        const result = await prebuilt.run({
          gpu: flagInt(args, "gpu") ?? 0,
          jobs: flagInt(args, "jobs") ?? 6,
          sha: flagStr(args, "sha"),
        });
        return reportLine(log, args, result, describePrebuilt);
      }
      if (flagStr(args, "sha") !== undefined)
        throw new UsageError("--sha names the fork commit --prebuilt builds");
      const result = await service.run({
        gpu: flagInt(args, "gpu") ?? 0,
        cap: flagCap(args),
        compile: flagBool(args, "compile"),
        portable: flagBool(args, "portable"),
        jobs: flagInt(args, "jobs"),
        fromTarball: flagStr(args, "from-tarball"),
        offPin: flagStr(args, "off-pin"),
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

function describePrebuilt(report: BuildPrebuiltReport): string {
  return `prebuilt of torad-labs/llama.cpp @ ${report.sha.slice(0, 7)} built in ${report.image}: ${report.tarball}`;
}
