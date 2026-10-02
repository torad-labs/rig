import type { Log } from "@rig/core";
import type { CheckMachine, CheckMachineReport } from "@rig/machine";
import { type Args, flagInt, flagStr } from "../cli/args.ts";
import { type Command, reportLine } from "../cli/command.ts";

const USAGE =
  "prepare [<head>] [--gpu N] [--allow-arch CAP] [--json]   tools, card, driver (for the head's own pin, when it names one) — exit 1 missing tool, 3 unsupported card, 4 driver too old";

export function checkMachineCommand(service: CheckMachine, log: Log): Command {
  return {
    name: "prepare",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    async run(args: Args) {
      const result = await service.run({
        gpu: flagInt(args, "gpu") ?? 0,
        allowArch: flagStr(args, "allow-arch"),
      });
      return reportLine(log, args, result, describe);
    },
  };
}

function describe(report: CheckMachineReport): string {
  const card = report.card;
  const unmeasured = report.supported ? "" : " — UNMEASURED card, allowed for a benchmark";
  const engine = report.built
    ? "the engine is installed with its CUDA runtime (no toolkit needed)"
    : report.prebuilt
      ? "the engine installs prebuilt (no toolkit needed)"
      : `toolkit ${report.toolkitCuda}`;
  return `prepared: ${card.name}, ${card.memoryMiB} MiB, sm_${card.computeCap}, driver ${card.driver} (CUDA ${report.driverCuda}); ${engine}${unmeasured}`;
}
