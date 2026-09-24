import { type Args, flagBool, flagInt, flagStr } from "../../shared/cli/args.ts";
import { type Command, type LoadHead, reportJson, withHead } from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import type { BringUpHead } from "./head-bringup.service.ts";

const USAGE =
  "up <head> [--gpu N] [--restart] [--cache-ram MiB] [--allow-arch CAP] [--json]   prepare, fetch, build, derive, unit, start — the whole bring-up";

export function bringUpHeadCommand(service: BringUpHead, loadHead: LoadHead, log: Log): Command {
  return {
    name: "up",
    usage: USAGE,
    run(args: Args) {
      return withHead(args.positionals[0], "up <head>", loadHead, log, async (head) => {
        const result = await service.run(head, {
          gpu: flagInt(args, "gpu") ?? head.gpu,
          restart: flagBool(args, "restart"),
          cacheRam: flagInt(args, "cache-ram"),
          allowArch: flagStr(args, "allow-arch"),
        });
        return reportJson(log, args, result);
      });
    },
  };
}
