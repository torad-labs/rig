import type { Log, Result } from "@rig/core";
import { ExitCode } from "@rig/core";
import type { Head } from "@rig/head";
import type {
  BenchReport,
  BoxStatus,
  GuardBox,
  PublishTemplate,
  RentGpu,
  StatusReport,
  SweepStopped,
  TemplateReport,
} from "@rig/rental";
import { MAX_HOURS } from "@rig/rental";
import { type Args, flagBool, flagInt, flagNumber, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, reportJson, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "vast up <head> --gpu CLASS [--gpus N] [--max-price DPH] [--min-down-mbps MBPS] [--geo GEO] [--allow-arch CAP] [--disk-gb GB] [--idle-minutes MIN] [--max-hours H] [--private] [--dry-run] | up <head> --template [--hours H] [--budget USD] [--disk-gb GB] [--max-price DPH] [--idle-minutes MIN] [--max-hours H] [--dry-run] | lab --gpu CLASS [--gpus N] [--max-price DPH] [--min-down-mbps MBPS] [--geo GEO] [--allow-arch CAP] [--disk-gb GB] [--idle-minutes MIN] [--max-hours H] [--vm] [--image IMAGE] [--pack HEAD [--hours H]] [--dry-run] | down [--box ID] [--all] | status [--box ID] | idle-check [--box ID] | sweep | bench <head> [--box ID] | template <head> [--disk-gb GB] [--idle-minutes MIN] [--max-hours H] [--dry-run] | guard <head> [--idle-minutes MIN] [--max-hours H] [--stop-when FILE], each with [--json]   a rented card as a head, reached through an ssh tunnel (its public pack; --private ships the private [derive] assets); rig holds any number of boxes, each with its own idle reaper and its own hard stop, a persistent timer armed at create that destroys it --max-hours after (vast.toml's max_hours when not given) whatever it reads; down, status, idle-check and bench act on the box --box names, or the only one held (down --all: every one, and rig's boxes the market lists that none of them is); up --template: a box from the head's published template, the offers ranked by the session's cost all in for --hours (1 when not given), its own on-start bringing the head up, and none over --budget dollars all in rented; --min-down-mbps (up, lab): no offer whose host downloads slower is rented, and with none above it the command fails naming the fastest offer and its price (up --template floors at 800 and ranks by download time already); a host's download is the rate rig measured on a box rented from it, else its region's, else what it declares; --pack (lab): the offers ranked by the session all in for --hours (1 when not given), the pack's download at each host's price per GB and rate included, the disk sized for the pack; lab: a rented card with no head, rig and the engine pin shipped to it for `rig engine` measurements over ssh (nothing is fetched, derived or served, and the idle reaper reads the card alone); --vm asks the market for a host that rents a full virtual machine and brings it up from vast's KVM image, the one kind of box that runs docker (`rig e2e`, `rig image`), and --image names another image for either kind; sweep: destroy rig's boxes stopped for vast.toml's stopped_hours (hourly from rig-vast-sweep.timer); template: the head's pushed image as a vast template a box comes up from with no script, and the sweep's timer armed; guard: on such a box, stop it after the idle budget, after --max-hours whatever it reads, or at once when --stop-when's file exists";
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
    "max-hours",
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
    "max-hours",
    "vm",
    "image",
    "pack",
    "hours",
    "dry-run",
    "json",
  ],
  down: ["box", "all", "json"],
  status: ["box", "json"],
  "idle-check": ["box", "json"],
  sweep: ["json"],
  bench: ["box", "json"],
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
              const maxHours = hardStopHours(args, log);
              if (maxHours === null) return ExitCode.Usage;
              const result = await service.upFromTemplate(head, {
                hours,
                budget,
                diskGb: flagNumber(args, "disk-gb"),
                maxDph: flagNumber(args, "max-price"),
                dryRun: flagBool(args, "dry-run"),
                idleMinutes,
                maxHours,
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
            const maxHours = hardStopHours(args, log);
            if (maxHours === null) return ExitCode.Usage;
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
              maxHours,
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
          const maxHours = hardStopHours(args, log);
          if (maxHours === null) return ExitCode.Usage;
          const pack = flagStr(args, "pack");
          const hours = flagNumber(args, "hours");
          if (hours !== undefined && (!pack || !(hours > 0))) {
            log.error(
              "usage: --hours prices the session --pack names, and takes a number above zero",
            );
            return ExitCode.Usage;
          }
          const lab = (head?: Head) =>
            service.lab({
              gpu,
              gpus,
              maxDph: flagNumber(args, "max-price"),
              minDownMbps,
              geo: flagStr(args, "geo"),
              dryRun: flagBool(args, "dry-run"),
              allowArch: flagStr(args, "allow-arch"),
              diskGb,
              idleMinutes,
              maxHours,
              vm: flagBool(args, "vm"),
              image: flagStr(args, "image"),
              pack: head,
              hours,
            });
          if (pack === undefined) return reportJson(log, args, await lab());
          return withHead(pack, "vast lab --pack <head>", loadHead, log, async (head) =>
            reportJson(log, args, await lab(head)),
          );
        }
        case "down": {
          const box = boxId(args, log);
          if (box === null) return ExitCode.Usage;
          return reportJson(log, args, await service.down({ all: flagBool(args, "all"), box }));
        }
        case "status": {
          const box = boxId(args, log);
          if (box === null) return ExitCode.Usage;
          return reportLine(log, args, await service.status({ box }), describeStatus);
        }
        case "idle-check": {
          const box = boxId(args, log);
          if (box === null) return ExitCode.Usage;
          return reportJson(log, args, await service.idleCheck({ box }));
        }
        case "sweep":
          return reportJson(log, args, await sweep.run());
        case "bench": {
          const box = boxId(args, log);
          if (box === null) return ExitCode.Usage;
          return withHead(name, "vast bench <head>", loadHead, log, async (head) =>
            reportLine(log, args, await service.bench(head, { box }), describeBench),
          );
        }
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

/** `--box`: a whole number above zero, undefined when absent, null (said) when it is not one */
function boxId(args: Args, log: Log): number | undefined | null {
  const box = positiveInt(args, "box");
  if (box === null) log.error("usage: --box takes a box's instance id");
  return box;
}

/** `--max-hours` for a rental: its hard stop's hours, in MAX_HOURS's range; undefined when absent, null (said) when
 *  out of it */
function hardStopHours(args: Args, log: Log): number | undefined | null {
  const hours = flagNumber(args, "max-hours");
  if (hours === undefined || (hours >= MAX_HOURS.floor && hours <= MAX_HOURS.ceiling)) return hours;
  log.error(`usage: --max-hours takes ${MAX_HOURS.floor} to ${MAX_HOURS.ceiling} hours`);
  return null;
}

/** one line a box */
function describeStatus(status: StatusReport): string {
  if (status.boxes.length === 0) return "no box";
  return status.boxes.map(describeBox).join("\n");
}

function describeBox(status: BoxStatus): string {
  const listed =
    status.listed === "unread" ? "vast UNREAD" : status.listed ? status.status : "NOT LISTED";
  const serves = status.box.head || status.box.legacy;
  const tunnel = serves ? `tunnel ${status.tunnelActive ? "active" : "down"}, ` : "";
  const server = serves ? `server ${status.healthy ? "healthy" : "unreachable"}, ` : "";
  const timer = `idle timer ${status.idleTimer === "re-armed" ? "WAS NOT RUNNING, re-armed" : status.idleTimer}`;
  const check = status.idleCheck === "failed" ? ", its last check FAILED" : "";
  const stop = status.hardStop
    ? `, hard stop ${new Date(status.hardStop.at).toISOString()}${status.hardStop.timer === "active" ? "" : status.hardStop.timer === "re-armed" ? " (WAS NOT RUNNING, re-armed)" : " (NOT RUNNING)"}`
    : ", no hard stop";
  return `box ${status.box.instanceId} ${status.box.gpu} $${status.box.dph}/h: ${listed}, ${status.hours} h (~$${status.cost}); ${tunnel}${server}${timer}${check}${stop}`;
}

function describeBench(bench: BenchReport): string {
  return `${bench.pass ? "PASS" : "FAIL"}: box evidence ${bench.remote}, live evidence ${bench.local}`;
}
