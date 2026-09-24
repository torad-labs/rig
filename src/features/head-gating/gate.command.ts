import { type Args, flagInt, flagStr } from "../../shared/cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import type { RunGates } from "./head-gating.service.ts";

const USAGE =
  "gate <head> [--only a,b] [--live [URL]] [--gpu N] [--json]   the head's probes on the gate card (and, with --live, against the running head); evidence in local/gate-runs/<head>/<run>/";

export function runGatesCommand(service: RunGates, loadHead: LoadHead, log: Log): Command {
  return {
    name: "gate",
    usage: USAGE,
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
