import type { Log } from "@rig/core";
import { ExitCode } from "@rig/core";
import { renderedFlag } from "@rig/head";
import type { ServeHead } from "@rig/serve";
import { type Args, flagBool, flagDevices, flagInt } from "../cli/args.ts";
import { type Command, type LoadHead, printJson, report, withHead } from "../cli/command.ts";

const USAGE =
  "serve <head> [--gpu N[,M…]|auto] [--cache-ram MiB] [--slots N] [--ctx N] [--plan] [--json] [-- extra llama-server args]   verify, then run llama-server in the foreground";

export function serveHeadCommand(service: ServeHead, loadHead: LoadHead, log: Log): Command {
  return {
    name: "serve",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    run(args: Args) {
      const [name, ...extra] = args.positionals;
      return withHead(name, "serve <head>", loadHead, log, async (head) => {
        // llama-server takes a flag's last occurrence, so one of serve's own after -- would replace what the profile check
        // charged and the engine's [caches] allowed (a K/V pair with no CUDA kernel, a -c the card cannot hold)
        const clash = extra.filter(renderedFlag);
        if (clash.length > 0) {
          log.error(
            `REFUSING: ${clash.join(", ")} after -- would override what serve renders from ${head.name} and its profile; use --ctx, --slots or --cache-ram, or the head's [cache] and profiles`,
          );
          return ExitCode.Usage;
        }
        const options = {
          devices: flagDevices(args, "gpu") ?? "auto",
          cacheRam: flagInt(args, "cache-ram"),
          slots: flagInt(args, "slots"),
          ctx: flagInt(args, "ctx"),
        };
        if (flagBool(args, "plan")) {
          const plan = await service.plan(head, options);
          return report(log, plan, (value) => {
            if (flagBool(args, "json")) printJson(value);
            else console.log([...value.argv, ...extra].join(" "));
          });
        }
        const served = await service.serve(head, options, extra);
        return served.ok ? served.value : report(log, served, () => {});
      });
    },
  };
}
