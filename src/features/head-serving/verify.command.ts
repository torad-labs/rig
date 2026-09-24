import { type Args, flagInt, flagStr } from "../../shared/cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import type { ServeHead } from "./head-serving.service.ts";

const USAGE =
  "verify <head> [--gpu N] [--pack PATH] [--json]   the build for this card is complete, the named pack (or, without --pack, this machine's own resolution) is a pinned pack's bytes, every asset exists (a rendered unit's ExecStartPre carries --pack, fixed at install)";

export function verifyHeadCommand(service: ServeHead, loadHead: LoadHead, log: Log): Command {
  return {
    name: "verify",
    usage: USAGE,
    run(args: Args) {
      return withHead(args.positionals[0], "verify <head>", loadHead, log, async (head) => {
        const result = await service.verify(
          head,
          flagInt(args, "gpu") ?? head.gpu,
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
