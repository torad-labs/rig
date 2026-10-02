import type { Log } from "@rig/core";
import type { DownloadPack } from "@rig/pack";
import { type Args, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "fetch <head> [--from FILE] [--json]   the source pack, by sha256, from Hugging Face or adopted from FILE";

export function downloadPackCommand(service: DownloadPack, loadHead: LoadHead, log: Log): Command {
  return {
    name: "fetch",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    run(args: Args) {
      return withHead(args.positionals[0], "fetch <head>", loadHead, log, async (head) => {
        const result = await service.run(head, { from: flagStr(args, "from") });
        return reportLine(log, args, result, (report) => `${report.state}: ${report.path}`);
      });
    },
  };
}
