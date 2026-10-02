import type { Log } from "@rig/core";
import type { BringUpHead } from "@rig/serve";
import { type Args, flagBool, flagDevices, flagInt, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, report, reportJson, withHead } from "../cli/command.ts";

const USAGE =
  "up <head> [--gpu N[,M…]|auto] [--restart] [--foreground] [--cache-ram MiB] [--slots N] [--allow-arch CAP] [--json]   prepare, fetch, build, derive, unit, start — the whole bring-up (--foreground: no unit, the server in this process, for a container)";

export function bringUpHeadCommand(service: BringUpHead, loadHead: LoadHead, log: Log): Command {
  return {
    name: "up",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    run(args: Args) {
      return withHead(args.positionals[0], "up <head>", loadHead, log, async (head) => {
        const options = {
          devices: flagDevices(args, "gpu"),
          restart: flagBool(args, "restart"),
          cacheRam: flagInt(args, "cache-ram"),
          slots: flagInt(args, "slots"),
          allowArch: flagStr(args, "allow-arch"),
        };
        if (flagBool(args, "foreground")) {
          const served = await service.foreground(head, options);
          return served.ok ? served.value : report(log, served, () => {});
        }
        return reportJson(log, args, await service.run(head, options));
      });
    },
  };
}
