// The evidence of one run: local/gate-runs/<head>/<stamp>/ holding <probe>.json for each
// probe's data, gate.log as the reader saw it, and summary.json for evidence.md to cite. The
// verdict is the conjunction of the probes' verdicts; a "measured" probe records and judges
// nothing.
import { join } from "node:path";
import type { CacheFormats, DraftCache } from "../../shared/head/cache-formats.ts";
import type { Clock, FileSystem } from "../../shared/ports/index.ts";
import { compactStamp } from "../../shared/stamp.ts";
import type { SkippedProbe } from "./probe-selection.ts";
import type { ProbeResult } from "./probes/probe.ts";

export type ProbeVerdict = Pick<ProbeResult, "name" | "pass" | "summary">;

export interface RunGatesReport {
  run: string;
  dir: string;
  gpu: number | null;
  probes: ProbeVerdict[];
  /** probes this head does not apply to (no draft head to compare …): named here, not only on
   *  the console a run's own log line scrolls past */
  skipped: SkippedProbe[];
  pass: boolean;
}

export interface RunProvenance {
  head: string;
  /** the pin and the pack the card legs ran; null for a run with no card legs, which measured only the live head */
  engine: string | null;
  served: string | null;
  /** the live head's URL when live probes ran: its engine and pack are its own, not this checkout's */
  live: string | null;
  gpu: number | null;
  /** the formats the card's server legs ran and the tier they came from (null: below every tier, the head's own); both
   *  null for a run with no card legs. A llama-bench leg runs no -cts and no K bias, so it records its own in its
   *  probe's json (benchCache) */
  cache: CacheFormats | null;
  tier: number | null;
  /** the draft head's own K/V (-ctkd/-ctvd) in the card's drafted legs: null without card legs or a draft head */
  draft_cache: Pick<DraftCache, "k" | "v"> | null;
}

export class GateRun {
  private readonly results: ProbeResult[] = [];
  private readonly log: string[] = [];

  private constructor(
    private readonly fs: FileSystem,
    readonly id: string,
    readonly dir: string,
  ) {}

  static async open(fs: FileSystem, clock: Clock, gateRunsDir: string, head: string) {
    const id = compactStamp(clock.now());
    const dir = join(gateRunsDir, head, id);
    await fs.mkdirp(dir);
    return new GateRun(fs, id, dir);
  }

  /** keeps the probe's data and returns the lines a reader sees, verdict last */
  async record(result: ProbeResult): Promise<string[]> {
    const verdict = result.pass === "measured" ? "MEASURED" : result.pass ? "PASS" : "FAIL";
    const lines = [
      ...result.lines.map((line) => `  ${line}`),
      `  ${verdict} ${result.name}: ${result.summary}`,
    ];
    this.results.push(result);
    this.log.push(`== ${result.name}`, ...lines);
    await this.fs.writeText(join(this.dir, `${result.name}.json`), toJson(result.data));
    return lines;
  }

  get failed(): string[] {
    return this.results.filter((result) => result.pass === false).map((result) => result.name);
  }

  async close(provenance: RunProvenance, skipped: SkippedProbe[]): Promise<RunGatesReport> {
    // a card probe that did not run is in every record of the run, gate.log included
    for (const { probe, reason } of skipped) this.log.push(`== ${probe}: SKIPPED, ${reason}`);
    const report: RunGatesReport = {
      run: this.id,
      dir: this.dir,
      gpu: provenance.gpu,
      probes: this.results.map(({ name, pass, summary }) => ({ name, pass, summary })),
      skipped,
      pass: this.failed.length === 0,
    };
    const summary = {
      head: provenance.head,
      engine: provenance.engine,
      served: provenance.served,
      live: provenance.live,
      cache: provenance.cache,
      tier: provenance.tier,
      draft_cache: provenance.draft_cache,
      ...report,
    };
    await this.fs.writeText(join(this.dir, "summary.json"), toJson(summary));
    await this.fs.writeText(join(this.dir, "gate.log"), `${this.log.join("\n")}\n`);
    return report;
  }
}

const toJson = (value: unknown) => JSON.stringify(value, null, 2);
