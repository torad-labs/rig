import type { CardLease, LeaseTerms, RunOptions, RunResult, Shell } from "@rig/core";

/** CardLease through gpu-lease where it is installed (the machine's card lease: the run waits its turn for the cards and
 *  holds them while it runs, its queue told the run's expected time). Its --card takes one index or `both` of a two-card machine, so another set is refused by
 *  name. Without gpu-lease nothing on the machine schedules the cards, and the command runs as it is. */
export class GpuLeaseCardLease implements CardLease {
  constructor(private readonly shell: Shell) {}
  async run(
    cards: readonly number[],
    terms: LeaseTerms,
    cmd: readonly string[],
    opts?: RunOptions,
  ): Promise<RunResult> {
    if (!(await this.shell.which("gpu-lease"))) return this.shell.run(cmd, opts);
    const set = [...new Set(cards)].sort((a, b) => a - b);
    const card = set.length === 1 ? String(set[0]) : set.join(",") === "0,1" ? "both" : null;
    if (card === null)
      return {
        code: 1,
        stdout: "",
        stderr: `gpu-lease holds one card or both of 0 and 1, not ${cards.join(",")}`,
      };
    return this.shell.run(
      [
        "gpu-lease",
        "run",
        "--card",
        card,
        "--eta",
        `${terms.etaMin}m`,
        "--max-hold",
        `${terms.maxHoldMin}m`,
        "--label",
        terms.label,
        "--",
        ...cmd,
      ],
      opts,
    );
  }
}
