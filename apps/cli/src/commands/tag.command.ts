import type { Log } from "@rig/core";
import type { TagRelease, TagReleaseReport } from "@rig/engine";
import { type Args, UsageError } from "../cli/args.ts";
import { type Command, reportLine } from "../cli/command.ts";

const USAGE =
  "tag [--json]   the release tag v<version> (apps/cli/package.json) on this checkout's HEAD, pushed to origin; refused unless HEAD is origin/main with no change git sees and `rig e2e` without --prebuilt passed on that commit in this checkout (a machine with only the driver installing the published pin)";

export function tagCommand(tag: TagRelease, log: Log): Command {
  return {
    name: "tag",
    usage: USAGE,
    async run(args: Args) {
      if (args.positionals.length > 0) throw new UsageError("tag takes no arguments");
      return reportLine(log, args, await tag.run(), describe);
    },
  };
}

function describe(report: TagReleaseReport): string {
  return `${report.tag} pushed on ${report.commit.slice(0, 7)} (${report.receipt})`;
}
