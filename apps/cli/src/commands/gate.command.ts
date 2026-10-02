import type { Log } from "@rig/core";
import type { RunGates } from "@rig/gate";
import { type Args, flagInt, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "gate <head> [--only a,b] [--live [URL]] [--gpu N] [--json]   the head's probes on the gate card (and, with --live, against the running head); evidence in local/gate-runs/<head>/<run>/";

export function runGatesCommand(service: RunGates, loadHead: LoadHead, log: Log): Command {
  return {
    name: "gate",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    run(args: Args) {
      return withHead(args.positionals[0], "gate <head>", loadHead, log, async (head) => {
        const live = args.flags.live;
        const result = await service.run(head, {
          only: flagStr(args, "only")
            ?.split(",")
            .map((name) => name.trim())
            .filter(Boolean),
          live: live === true ? true : typeof live === "string" ? live : undefined,
          gpu: flagInt(args, "gpu"),
        });
        return reportLine(
          log,
          args,
          result,
          (report) => `PASS ${report.probes.length} probe(s): ${report.dir}`,
        );
      });
    },
  };
}
