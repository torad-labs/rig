import type { Log } from "@rig/core";
import { ExitCode } from "@rig/core";
import type { ManageUnit, UnitStatus } from "@rig/serve";
import { type Args, flagBool, flagDevices, flagInt } from "../cli/args.ts";
import {
  type Command,
  type LoadHead,
  printJson,
  report,
  reportLine,
  withHead,
} from "../cli/command.ts";

const USAGE =
  "unit install|render|status|uninstall <head> [--gpu N[,M…]|auto] [--cache-ram MiB] [--slots N] [--json]   the systemd user unit that keeps the head running";
const FORM = USAGE.split("   ")[0] ?? USAGE;

export function manageUnitCommand(service: ManageUnit, loadHead: LoadHead, log: Log): Command {
  return {
    name: "unit",
    usage: USAGE,
    headName: (args) => args.positionals[1],
    run(args: Args) {
      const [subcommand, name] = args.positionals;
      if (!subcommand) {
        log.error(`usage: rig ${FORM}`);
        return Promise.resolve(ExitCode.Usage);
      }
      return withHead(name, FORM, loadHead, log, async (head) => {
        const options = {
          devices: flagDevices(args, "gpu"),
          cacheRam: flagInt(args, "cache-ram"),
          slots: flagInt(args, "slots"),
        };
        switch (subcommand) {
          case "install":
            return reportLine(
              log,
              args,
              await service.install(head, options),
              (installed) => `${installed.state}: ${installed.path}`,
            );
          case "render":
            return report(log, await service.render(head, options), (rendered) =>
              process.stdout.write(rendered.text),
            );
          case "status": {
            const status = await service.status(head);
            if (flagBool(args, "json")) printJson(status);
            else log.info(describeStatus(status));
            return 0;
          }
          case "uninstall":
            return reportLine(log, args, await service.uninstall(head), (removed) =>
              removed.removed ? `removed ${removed.unit}` : `${removed.unit} was not installed`,
            );
          default:
            log.error(`unknown subcommand ${JSON.stringify(subcommand)}; usage: rig ${FORM}`);
            return ExitCode.Usage;
        }
      });
    },
  };
}

function describeStatus(status: UnitStatus): string {
  const installed = status.installed ? "installed" : "not installed";
  const active = status.active
    ? `active (pid ${status.mainPid}, oom_score_adj ${status.oomScoreAdj ?? "?"})`
    : "inactive";
  return `${status.unit}: ${installed}, ${active}`;
}
