import { type Args, flagInt, flagStr } from "../../shared/cli/args.ts";
import { type Command, reportLine } from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import type { CheckMachine, CheckMachineReport } from "./machine-check.service.ts";

const USAGE =
  "prepare [--gpu N] [--allow-arch CAP] [--json]   tools, card, driver — exit 1 missing tool, 3 unsupported card, 4 driver too old";

export function checkMachineCommand(service: CheckMachine, log: Log): Command {
  return {
    name: "prepare",
    usage: USAGE,
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
  const engine = report.prebuilt
    ? "the engine installs prebuilt (no toolkit needed)"
    : `toolkit ${report.toolkitCuda}`;
  return `prepared: ${card.name}, ${card.memoryMiB} MiB, sm_${card.computeCap}, driver ${card.driver} (CUDA ${report.driverCuda}); ${engine}${unmeasured}`;
}
