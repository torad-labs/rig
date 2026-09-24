import { type Args, flagBool, flagInt, flagStr } from "../../shared/cli/args.ts";
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
  "vast up <head> --gpu CLASS [--gpus N] [--max-price DPH] [--geo GEO] [--allow-arch CAP] [--disk-gb GB] [--idle-minutes MIN] [--dry-run] | down [--all] | status | idle-check | bench <head>   a rented card as a head, reached through an ssh tunnel";
const FORM = USAGE.split("   ")[0] ?? USAGE;

export function gpuRentalCommand(service: RentGpu, loadHead: LoadHead, log: Log): Command {
  return {
    name: "vast",
    usage: USAGE,
    async run(args: Args) {
      const [subcommand, name] = args.positionals;
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
            const gpus = flagStr(args, "gpus");
            const maxPrice = flagStr(args, "max-price");
            const diskGb = positiveInt(args, "disk-gb");
            const idleMinutes = positiveInt(args, "idle-minutes");
            if (diskGb === null || idleMinutes === null) {
              log.error("usage: --disk-gb and --idle-minutes take a whole number above zero");
              return ExitCode.Usage;
            }
            const result = await service.up(head, {
              gpu,
              gpus: gpus !== undefined ? Number(gpus) : undefined,
              maxDph: maxPrice !== undefined ? Number(maxPrice) : undefined,
              geo: flagStr(args, "geo"),
              dryRun: flagBool(args, "dry-run"),
              allowArch: flagStr(args, "allow-arch"),
              diskGb,
              idleMinutes,
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

/** the flag as a whole number above zero, undefined when absent, null when malformed: a typo
 *  must not fall back to vast.toml's value on a box that is about to bill */
function positiveInt(args: Args, name: string): number | undefined | null {
  if (args.flags[name] === undefined) return undefined;
  const value = flagInt(args, name);
  return value !== undefined && value > 0 ? value : null;
}

function describeStatus(status: StatusReport): string {
  const tunnel = `tunnel ${status.tunnelActive ? "active" : "down"}`;
  if (!status.box) return `no box; ${tunnel}`;
  const listed = status.listed ? status.status : "NOT LISTED";
  const server = `server ${status.healthy ? "healthy" : "unreachable"}`;
  return `box ${status.box.instanceId} ${status.box.gpu} $${status.box.dph}/h: ${listed}, ${status.hours} h (~$${status.cost}); ${tunnel}, ${server}`;
}

function describeBench(bench: BenchReport): string {
  return `${bench.pass ? "PASS" : "FAIL"}: box evidence ${bench.remote}, live evidence ${bench.local}`;
}
