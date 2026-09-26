import { type Args, flagBool, flagInt } from "../../shared/cli/args.ts";
import { type Command, type LoadHead, printJson, withHead } from "../../shared/cli/command.ts";
import { renderedFlag } from "../../shared/head/head-config.ts";
import type { Log } from "../../shared/ports/index.ts";
import { ExitCode } from "../../shared/result.ts";
import type { ServeHead } from "./head-serving.service.ts";

const USAGE =
  "serve <head> [--gpu N] [--cache-ram MiB] [--slots N] [--ctx N] [--plan] [--json] [-- extra llama-server args]   verify, then run llama-server in the foreground";

export function serveHeadCommand(service: ServeHead, loadHead: LoadHead, log: Log): Command {
  return {
    name: "serve",
    usage: USAGE,
    run(args: Args) {
      const [name, ...extra] = args.positionals;
      return withHead(name, "serve <head>", loadHead, log, async (head) => {
        // llama-server takes a flag's last occurrence, so one of serve's own after -- would replace what the tier check
        // charged and the engine's [caches] allowed (a K/V pair with no CUDA kernel, a -c the card cannot hold)
        const clash = extra.filter(renderedFlag);
        if (clash.length > 0) {
          log.error(
            `REFUSING: ${clash.join(", ")} after -- would override what serve renders from ${head.name} and its tier; use --ctx, --slots or --cache-ram, or the head's [cache] and tiers`,
          );
          return ExitCode.Usage;
        }
        const gpu = flagInt(args, "gpu") ?? head.gpu;
        const plan = await service.plan(head, {
          gpu,
          cacheRam: flagInt(args, "cache-ram"),
          slots: flagInt(args, "slots"),
          ctx: flagInt(args, "ctx"),
        });
        if (!plan.ok) {
          log.error(plan.message);
          return plan.code;
        }
        if (flagBool(args, "plan")) {
          if (flagBool(args, "json")) printJson(plan.value);
          else console.log([...plan.value.argv, ...extra].join(" "));
          return 0;
        }
        const verified = await service.verify(head, gpu);
        if (!verified.ok) {
          log.error(verified.message);
          return verified.code;
        }
        return service.run(head, plan.value, extra);
      });
    },
  };
}
