import { type Args, flagBool, flagInt, flagNumber, flagStr } from "../../shared/cli/args.ts";
import {
  type Command,
  type LoadHead,
  reportJson,
  reportLine,
  withHead,
} from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import { ExitCode } from "../../shared/result.ts";
import type { BenchReport, RentGpu, StatusReport } from "./gpu-rental.service.ts";

const USAGE =
  "vast up <head> --gpu CLASS [--gpus N] [--max-price DPH] [--geo GEO] [--allow-arch CAP] [--disk-gb GB] [--idle-minutes MIN] [--private] [--dry-run] | down [--all] | status | idle-check | bench <head>, each with [--json]   a rented card as a head, reached through an ssh tunnel (its public pack; --private ships the private [derive] assets)";
const FORM = USAGE.split("   ")[0] ?? USAGE;
/** the flags each subcommand takes: main holds a command's flags to its whole usage line, which names every
 *  subcommand's, so `down --dry-run` would pass there and destroy the box; each subcommand refuses any other */
export const SUBCOMMAND_FLAGS: Record<string, readonly string[]> = {
  up: [
    "gpu",
    "gpus",
    "max-price",
    "geo",
    "allow-arch",
    "disk-gb",
    "idle-minutes",
    "private",
    "dry-run",
    "json",
  ],
  down: ["all", "json"],
  status: ["json"],
  "idle-check": ["json"],
  bench: ["json"],
};

export function gpuRentalCommand(service: RentGpu, loadHead: LoadHead, log: Log): Command {
  return {
    name: "vast",
    usage: USAGE,
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
            if (gpus === null || diskGb === null || idleMinutes === null) {
              log.error(
                "usage: --gpus, --disk-gb and --idle-minutes take a whole number above zero",
              );
              return ExitCode.Usage;
            }
            const result = await service.up(head, {
              gpu,
              gpus,
              maxDph: flagNumber(args, "max-price"),
              geo: flagStr(args, "geo"),
              dryRun: flagBool(args, "dry-run"),
              allowArch: flagStr(args, "allow-arch"),
              diskGb,
              idleMinutes,
              private: flagBool(args, "private"),
            });
            return reportJson(log, args, result);
          });
        case "down":
          return reportJson(log, args, await service.down({ all: flagBool(args, "all") }));
        case "status":
          return reportLine(log, args, await service.status(), describeStatus);
        case "idle-check":
          return reportJson(log, args, await service.idleCheck());
        case "bench":
          return withHead(name, "vast bench <head>", loadHead, log, async (head) =>
            reportLine(log, args, await service.bench(head), describeBench),
          );
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
