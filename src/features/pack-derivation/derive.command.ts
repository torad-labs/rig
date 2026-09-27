import type { Args } from "../../shared/cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import type { DerivePack, DerivePackReport } from "./pack-derivation.service.ts";

const USAGE =
  "derive <head> [--json]   the served pack from the source pack, byte-for-byte against its pinned sha256";

export function derivePackCommand(service: DerivePack, loadHead: LoadHead, log: Log): Command {
  return {
    name: "derive",
    usage: USAGE,
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
