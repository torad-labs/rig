// CardLease: one seam between rig and the machine. A port names a capability, never a tool.
import type { RunOptions, RunResult } from "./shell.ts";

/** what a hold of the cards declares to the machine's queue */
export interface LeaseTerms {
  /** who holds them and for what, as the queue shows it */
  label: string;
  /** how long the run is expected to take, in minutes: the queue fits shorter jobs into the wait by it */
  etaMin: number;
  /** the longest the cards are held, in minutes */
  maxHoldMin: number;
}

export interface CardLease {
  /** `cmd` run while `cards` (by nvidia-smi index) are held for it alone, so no other job this machine schedules runs
   *  on them meanwhile, under `terms`; on a machine with nothing to hold them through, `cmd` as is */
  run(
    cards: readonly number[],
    terms: LeaseTerms,
    cmd: readonly string[],
    opts?: RunOptions,
  ): Promise<RunResult>;
}
