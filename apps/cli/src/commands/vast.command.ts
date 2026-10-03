import type { Log, Result } from "@rig/core";
import { ExitCode } from "@rig/core";
import type { Head } from "@rig/head";
import type {
  BenchReport,
  GuardBox,
  PublishTemplate,
  RentGpu,
  StatusReport,
  SweepStopped,
  TemplateReport,
} from "@rig/rental";
import { type Args, flagBool, flagInt, flagNumber, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, reportJson, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "vast up <head> --gpu CLASS [--gpus N] [--max-price DPH] [--min-down-mbps MBPS] [--geo GEO] [--allow-arch CAP] [--disk-gb GB] [--idle-minutes MIN] [--private] [--dry-run] | up <head> --template [--hours H] [--budget USD] [--disk-gb GB] [--max-price DPH] [--idle-minutes MIN] [--dry-run] | lab --gpu CLASS [--gpus N] [--max-price DPH] [--min-down-mbps MBPS] [--geo GEO] [--allow-arch CAP] [--disk-gb GB] [--idle-minutes MIN] [--vm] [--image IMAGE] [--dry-run] | down [--all] | status | idle-check | sweep | bench <head> | template <head> [--disk-gb GB] [--idle-minutes MIN] [--max-hours H] [--dry-run] | guard <head> [--idle-minutes MIN] [--max-hours H] [--stop-when FILE], each with [--json]   a rented card as a head, reached through an ssh tunnel (its public pack; --private ships the private [derive] assets); up --template: a box from the head's published template, the offers ranked by the session's cost all in for --hours (1 when not given), its own on-start bringing the head up, and none over --budget dollars all in rented; --min-down-mbps (up, lab): no offer whose host downloads slower is rented, and with none above it the command fails naming the fastest offer and its price (up --template floors at 800 and ranks by download time already); lab: a rented card with no head, rig and the engine pin shipped to it for `rig engine` measurements over ssh (nothing is fetched, derived or served, and the idle reaper reads the card alone); --vm asks the market for a host that rents a full virtual machine and brings it up from vast's KVM image, the one kind of box that runs docker (`rig e2e`, `rig image`), and --image names another image for either kind; sweep: destroy rig's boxes stopped for vast.toml's stopped_hours (hourly from rig-vast-sweep.timer); template: the head's pushed image as a vast template a box comes up from with no script, and the sweep's timer armed; guard: on such a box, stop it after the idle budget, after --max-hours whatever it reads, or at once when --stop-when's file exists";
const FORM = USAGE.split("   ")[0] ?? USAGE;
/** the flags each subcommand takes: main holds a command's flags to its whole usage line, which names every
 *  subcommand's, so `down --dry-run` would pass there and destroy the box; each subcommand refuses any other */
export const SUBCOMMAND_FLAGS: Record<string, readonly string[]> = {
  up: [
    "gpu",
    "gpus",
    "max-price",
    "min-down-mbps",
    "geo",
    "allow-arch",
    "disk-gb",
    "idle-minutes",
    "private",
    "dry-run",
    "template",
    "hours",
    "budget",
    "json",
  ],
  lab: [
    "gpu",
    "gpus",
    "max-price",
    "min-down-mbps",
    "geo",
    "allow-arch",
    "disk-gb",
    "idle-minutes",
    "vm",
    "image",
    "dry-run",
    "json",
  ],
  down: ["all", "json"],
  status: ["json"],
  "idle-check": ["json"],
  sweep: ["json"],
  bench: ["json"],
  template: ["disk-gb", "idle-minutes", "max-hours", "dry-run", "json"],
  guard: ["idle-minutes", "max-hours", "stop-when", "json"],
};

/** the idle rule's budget on a box rented from a template: one hour with nothing served */
export const TEMPLATE_IDLE_MINUTES = 60;
/** a template box's lifetime: stopped after a day whatever its readings say, so a reading that is wrong never bills for
 *  good (a stop keeps the disk; `vastai start instance` resumes it) */
export const TEMPLATE_MAX_HOURS = 24;

/** the head's template saved on its last pushed image, and the sweep's timer armed: a box rented from it in vast's
 *  console has no idle timer here, so once its guard stops it, the sweep is what destroys it */
export async function publishTemplate(
  template: PublishTemplate,
  sweep: SweepStopped,
  head: Head,
  options: {
    idleMinutes?: number | undefined;
    maxHours?: number | undefined;
    diskGb?: number | undefined;
    dryRun: boolean;
  },
): Promise<Result<TemplateReport>> {
  const result = await template.run(head, {
    idleMinutes: options.idleMinutes ?? TEMPLATE_IDLE_MINUTES,
    maxHours: options.maxHours ?? TEMPLATE_MAX_HOURS,
    diskGb: options.diskGb,
    dryRun: options.dryRun,
  });
  if (result.ok && !options.dryRun) await sweep.arm();
  return result;
}

export function gpuRentalCommand(
  service: RentGpu,
  template: PublishTemplate | null,
  guard: GuardBox,
  sweep: SweepStopped,
  loadHead: LoadHead,
  log: Log,
): Command {
  return {
    name: "vast",
    usage: USAGE,
    headName: (args) => args.positionals[1],
    async run(args: Args) {
      const [subcommand, name] = args.positionals;
      const takes = SUBCOMMAND_FLAGS[subcommand ?? ""];
      const other = takes ? Object.keys(args.flags).filter((flag) => !takes.includes(flag)) : [];
      if (other.length > 0) {
        log.error(
          `vast ${subcommand} does not take ${other.map((flag) => `--${flag}`).join(", ")}; usage: rig ${FORM}`,
        );
        return ExitCode.Usage;
      }
      switch (subcommand) {
        case "up":
          return withHead(name, "vast up <head>", loadHead, log, async (head) => {
            if (flagBool(args, "template")) {
              if (args.flags["min-down-mbps"] !== undefined) {
                log.error(
                  "usage: up --template does not take --min-down-mbps: it ranks offers by the pack's download time and floors the host at 800 Mb/s",
                );
                return ExitCode.Usage;
              }
              const hours = flagNumber(args, "hours") ?? 1;
              const budget = flagNumber(args, "budget");
              const idleMinutes = positiveInt(args, "idle-minutes");
              if (!(hours > 0) || idleMinutes === null || (budget !== undefined && !(budget > 0))) {
                log.error(
                  "usage: --hours and --budget take a number above zero, --idle-minutes a whole one",
                );
                return ExitCode.Usage;
              }
              const result = await service.upFromTemplate(head, {
                hours,
                budget,
                diskGb: flagNumber(args, "disk-gb"),
                maxDph: flagNumber(args, "max-price"),
                dryRun: flagBool(args, "dry-run"),
                idleMinutes,
              });
              return reportJson(log, args, result);
            }
            const gpu = flagStr(args, "gpu");
            if (!gpu) {
              log.error(
                "usage: rig vast up <head> --gpu CLASS (as vast names it: H100_SXM, RTX_5090, …)",
              );
              return ExitCode.Usage;
            }
            const gpus = positiveInt(args, "gpus");
            const diskGb = positiveInt(args, "disk-gb");
            const idleMinutes = positiveInt(args, "idle-minutes");
            const minDownMbps = positiveInt(args, "min-down-mbps");
            if (gpus === null || diskGb === null || idleMinutes === null || minDownMbps === null) {
              log.error(
                "usage: --gpus, --disk-gb, --idle-minutes and --min-down-mbps take a whole number above zero",
              );
              return ExitCode.Usage;
            }
            const result = await service.up(head, {
              gpu,
              gpus,
              maxDph: flagNumber(args, "max-price"),
              minDownMbps,
              geo: flagStr(args, "geo"),
              dryRun: flagBool(args, "dry-run"),
              allowArch: flagStr(args, "allow-arch"),
              diskGb,
              idleMinutes,
              private: flagBool(args, "private"),
            });
            return reportJson(log, args, result);
          });
        case "lab": {
          const gpu = flagStr(args, "gpu");
          if (!gpu) {
            log.error("usage: rig vast lab --gpu CLASS (as vast names it: RTX_5090, H100_SXM, …)");
            return ExitCode.Usage;
          }
          const gpus = positiveInt(args, "gpus");
          const diskGb = positiveInt(args, "disk-gb");
          const idleMinutes = positiveInt(args, "idle-minutes");
          const minDownMbps = positiveInt(args, "min-down-mbps");
          if (gpus === null || diskGb === null || idleMinutes === null || minDownMbps === null) {
            log.error(
              "usage: --gpus, --disk-gb, --idle-minutes and --min-down-mbps take a whole number above zero",
            );
            return ExitCode.Usage;
          }
          const result = await service.lab({
            gpu,
            gpus,
            maxDph: flagNumber(args, "max-price"),
            minDownMbps,
            geo: flagStr(args, "geo"),
            dryRun: flagBool(args, "dry-run"),
            allowArch: flagStr(args, "allow-arch"),
            diskGb,
            idleMinutes,
            vm: flagBool(args, "vm"),
            image: flagStr(args, "image"),
          });
          return reportJson(log, args, result);
        }
        case "down":
          return reportJson(log, args, await service.down({ all: flagBool(args, "all") }));
        case "status":
          return reportLine(log, args, await service.status(), describeStatus);
        case "idle-check":
          return reportJson(log, args, await service.idleCheck());
        case "sweep":
          return reportJson(log, args, await sweep.run());
        case "bench":
          return withHead(name, "vast bench <head>", loadHead, log, async (head) =>
            reportLine(log, args, await service.bench(head), describeBench),
          );
        case "template":
          return withHead(name, "vast template <head>", loadHead, log, async (head) => {
            if (!template) {
              log.error("vast template reads the engine pin, and engine.toml did not load");
              return ExitCode.Failure;
            }
            const diskGb = positiveInt(args, "disk-gb");
            const idleMinutes = positiveInt(args, "idle-minutes");
            const maxHours = positiveInt(args, "max-hours");
            if (diskGb === null || idleMinutes === null || maxHours === null) {
              log.error(
                "usage: --disk-gb, --idle-minutes and --max-hours take a whole number above zero",
              );
              return ExitCode.Usage;
            }
            const result = await publishTemplate(template, sweep, head, {
              idleMinutes,
              maxHours,
              diskGb,
              dryRun: flagBool(args, "dry-run"),
            });
            return reportLine(log, args, result, (t) =>
              t.hashId
                ? `template ${t.name}: ${t.hashId}`
                : `template ${t.name} (dry run): ${t.image}, ${t.diskGb} GB\n${t.onstart}`,
            );
          });
        case "guard":
          return withHead(name, "vast guard <head>", loadHead, log, async (head) => {
            const idleMinutes = positiveInt(args, "idle-minutes");
            const maxHours = positiveInt(args, "max-hours");
            if (idleMinutes === null || maxHours === null) {
              log.error("usage: --idle-minutes and --max-hours take a whole number above zero");
              return ExitCode.Usage;
            }
            const result = await guard.run(head, {
              idleMinutes: idleMinutes ?? TEMPLATE_IDLE_MINUTES,
              maxHours,
              stopWhen: flagStr(args, "stop-when"),
            });
            return reportJson(log, args, result);
          });
        default:
          log.error(`usage: rig ${FORM}`);
          return ExitCode.Usage;
      }
    },
  };
}

/** the flag as a whole number above zero, undefined when absent, null when zero or below (a value
 *  that is no number at all is flagInt's UsageError): a typo must not fall back to vast.toml's
 *  value on a box that is about to bill */
function positiveInt(args: Args, name: string): number | undefined | null {
  if (args.flags[name] === undefined) return undefined;
  const value = flagInt(args, name);
  return value !== undefined && value > 0 ? value : null;
}

function describeStatus(status: StatusReport): string {
  const tunnel = `tunnel ${status.tunnelActive ? "active" : "down"}`;
  if (!status.box) return `no box; ${tunnel}`;
  const listed =
    status.listed === "unread" ? "vast UNREAD" : status.listed ? status.status : "NOT LISTED";
  const server = `server ${status.healthy ? "healthy" : "unreachable"}`;
  const timer = `idle timer ${status.idleTimer === "re-armed" ? "WAS NOT RUNNING, re-armed" : status.idleTimer}`;
  const check = status.idleCheck === "failed" ? ", its last check FAILED" : "";
  return `box ${status.box.instanceId} ${status.box.gpu} $${status.box.dph}/h: ${listed}, ${status.hours} h (~$${status.cost}); ${tunnel}, ${server}, ${timer}${check}`;
}

function describeBench(bench: BenchReport): string {
  return `${bench.pass ? "PASS" : "FAIL"}: box evidence ${bench.remote}, live evidence ${bench.local}`;
}
