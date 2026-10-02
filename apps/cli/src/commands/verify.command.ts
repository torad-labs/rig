import type { Log } from "@rig/core";
import type { ServeHead } from "@rig/serve";
import { type Args, flagDevices, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "verify <head> [--gpu N[,M…]|auto] [--pack PATH] [--json]   the build for the cards is complete, the named pack (or, without --pack, this machine's own resolution) is a pinned pack's bytes, every asset exists (a rendered unit's ExecStartPre carries --pack, fixed at install)";

export function verifyHeadCommand(service: ServeHead, loadHead: LoadHead, log: Log): Command {
  return {
    name: "verify",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    run(args: Args) {
      return withHead(args.positionals[0], "verify <head>", loadHead, log, async (head) => {
        const result = await service.verify(
          head,
          flagDevices(args, "gpu") ?? "auto",
          flagStr(args, "pack"),
        );
        return reportLine(
          log,
          args,
          result,
          (report) =>
            `verified ${head.name}: ${report.binDir} (sm_${report.cap}), ${report.servedPath}`,
        );
      });
    },
  };
}
