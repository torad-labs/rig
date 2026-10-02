import type { Log } from "@rig/core";
import type { DerivePack, DerivePackReport } from "@rig/pack";
import type { Args } from "../cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "derive <head> [--json]   the served pack from the source pack, byte-for-byte against its pinned sha256";

export function derivePackCommand(service: DerivePack, loadHead: LoadHead, log: Log): Command {
  return {
    name: "derive",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    run(args: Args) {
      return withHead(args.positionals[0], "derive <head>", loadHead, log, async (head) =>
        reportLine(log, args, await service.run(head), describe),
      );
    },
  };
}

function describe(report: DerivePackReport): string {
  const flipped = report.flipped ? ` (${report.flipped} digits flipped)` : "";
  return `${report.state}: ${report.path}${flipped}`;
}
