// Which probes a run executes. Named with --only, the run is exactly those; otherwise every
// probe on the gate card runs, and the ones against the live head too when --live is given.
// A probe that cannot measure this head (no draft head to compare) is skipped and said; asked
// for by name it is a usage error, as is a live probe without a live head.
import type { Head } from "../../shared/head/head.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import type { Probe } from "./probes/probe.ts";

export interface SkippedProbe {
  probe: string;
  reason: string;
}

export interface ProbeSelection {
  selected: Probe[];
  skipped: SkippedProbe[];
}

export interface SelectionRequest {
  only: string[] | undefined;
  liveUrl: string | null;
}

export function selectProbes(
  probes: Probe[],
  head: Head,
  request: SelectionRequest,
): Result<ProbeSelection> {
  const unknown = request.only?.find((name) => !probes.some((probe) => probe.name === name));
  if (unknown !== undefined) {
    const known = probes.map((probe) => probe.name).join(", ");
    return fail(ExitCode.Usage, `no probe named ${JSON.stringify(unknown)}; probes: ${known}`);
  }

  const named = probes.filter((probe) => {
    if (request.only) return request.only.includes(probe.name);
    return probe.needs !== "live" || request.liveUrl !== null;
  });

  const selected: Probe[] = [];
  const skipped: SkippedProbe[] = [];
  for (const probe of named) {
    const reason = probe.applies?.(head) ?? null;
    if (reason === null) {
      selected.push(probe);
    } else if (request.only) {
      return fail(ExitCode.Usage, `${probe.name} does not apply to ${head.name}: ${reason}`);
    } else {
      skipped.push({ probe: probe.name, reason });
    }
  }

  const liveWithoutHead = selected.find((probe) => probe.needs === "live");
  if (liveWithoutHead && request.liveUrl === null) {
    const message = `${liveWithoutHead.name} runs against the live head: add --live [URL]`;
    return fail(ExitCode.Usage, message);
  }
  if (selected.length === 0) return fail(ExitCode.Usage, "no probe selected");
  return ok({ selected, skipped });
}
