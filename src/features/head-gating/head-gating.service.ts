// Runs a head's probes and keeps the evidence. The probes are chosen (probe-selection), the gate
// card is claimed before anything starts (gate-card), each probe on the card starts its own
// legs on the gate port (gate-server), each live probe speaks to the running head, and every
// result lands in a run directory (gate-run) with a summary.json a reader or evidence.md cites.
// A failed probe fails the run with exit 1 and its lines say what moved.
import { type Engine, engineSource } from "../../shared/engine/engine.ts";
import { engineCorpus } from "../../shared/engine/engine-corpus.ts";
import { tierCache } from "../../shared/head/cache-formats.ts";
import type { Head } from "../../shared/head/head.ts";
import type { Layout } from "../../shared/layout.ts";
import type {
  Clock,
  FileSystem,
  Git,
  Gpu,
  Hasher,
  Http,
  Log,
  Shell,
} from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import { claimGateCard, type GateCard } from "./gate-card.ts";
import type { GateLegs, LegOptions } from "./gate-legs.ts";
import { GateRun, type RunGatesReport, type RunProvenance } from "./gate-run.ts";
import { GateServer } from "./gate-server.ts";
import { loadGates } from "./gates-config.ts";
import type { HeadClient } from "./head-client.ts";
import { LlamaClient } from "./llama-client.ts";
import { selectProbes } from "./probe-selection.ts";
import type { Probe, ProbeContext } from "./probes/probe.ts";

export type { RunGatesReport } from "./gate-run.ts";

export interface RunGatesDeps {
  shell: Shell;
  fs: FileSystem;
  git: Git;
  http: Http;
  gpu: Gpu;
  hasher: Hasher;
  clock: Clock;
  log: Log;
}

export interface RunGatesOptions {
  /** probe names; without it every card probe runs, and the live ones when `live` is set */
  only?: string[] | undefined;
  /** `true` for the head's own port, a URL for a head elsewhere (a tunnel to a rented box) */
  live?: string | boolean | undefined;
  /** the card the legs run on; gates.toml's when absent */
  gpu?: number | undefined;
}

export class RunGates {
  constructor(
    private readonly deps: RunGatesDeps,
    private readonly layout: Layout,
    private readonly engine: Engine,
    private readonly probes: Probe[],
  ) {}

  async run(head: Head, options: RunGatesOptions = {}): Promise<Result<RunGatesReport>> {
    const gates = await loadGates(this.deps.fs, head);
    if (!gates.ok) return gates;

    const liveUrl = liveUrlOf(head, options.live);
    const selection = selectProbes(this.probes, head, { only: options.only, liveUrl });
    if (!selection.ok) return selection;
    for (const { probe, reason } of selection.value.skipped) {
      this.deps.log.info(`-- ${probe}: skipped, ${reason}`);
    }
    const probes = selection.value.selected;

    const gpu = options.gpu ?? gates.value.gate.gpu;
    const onCard = probes.some((probe) => probe.needs !== "live");
    const card = onCard ? await claimGateCard(this.deps, this.engine, head, gpu) : ok(null);
    if (!card.ok) return card;
    if (card.value) {
      const { cache, tier } = card.value;
      const formats = `K/V ${cache.k}/${cache.v}, state ${cache.s}`;
      if (tier === null)
        this.deps.log.warn(
          `GPU ${gpu} is below every tier of ${head.name}: its legs run the head's own cache formats (${formats}), which serve would never run on it`,
        );
      else this.deps.log.info(`GPU ${gpu}: the ${tier} MiB tier's cache formats, ${formats}`);
    }

    const live = liveUrl ? await this.liveHead(liveUrl) : ok(null);
    if (!live.ok) return live;

    const run = await GateRun.open(
      this.deps.fs,
      this.deps.clock,
      this.layout.gateRunsDir,
      head.name,
    );
    const context: Omit<ProbeContext, "corpus"> = {
      head,
      gates: gates.value,
      server: this.legsOn(card.value, head, gates.value.gate.port, run.dir),
      live: live.value,
      binDir: card.value?.binDir ?? "",
      gpu,
      cap: card.value?.cap ?? "",
      cache: card.value?.cache ?? tierCache(head, {}),
      fs: this.deps.fs,
      shell: this.deps.shell,
      log: this.deps.log,
      runDir: run.dir,
      gateRunsDir: this.layout.gateRunsDir,
    };
    const corpus = () => this.corpus(gates.value.corpus.chars);

    for (const probe of probes) {
      this.deps.log.info(`== ${probe.name}`);
      const result = await probe.run({ ...context, corpus });
      for (const line of await run.record(result)) this.deps.log.info(line);
    }

    const report = await run.close(
      runProvenance(head, this.engine, card.value, liveUrl),
      selection.value.skipped,
    );
    if (!report.pass) {
      return fail(ExitCode.Failure, `${run.failed.join(", ")} FAILED — ${run.dir}/gate.log`);
    }
    return ok(report);
  }

  private async liveHead(url: string): Promise<Result<HeadClient>> {
    const client = new LlamaClient(this.deps.http, url);
    if (!(await client.healthy())) return fail(ExitCode.Failure, `no healthy head at ${url}`);
    return ok(client);
  }

  private legsOn(card: GateCard | null, head: Head, port: number, runDir: string): GateLegs {
    if (card === null) return noGateLegs;
    return new GateServer(this.deps, head, card.binDir, card.gpu, port, runDir, card.cache);
  }

  /** the probes' long text, from the engine tree the build used: the submodule at the pin, or
   *  the fetched copy on a box */
  private async corpus(chars: number): Promise<string> {
    const source = await engineSource(this.deps.fs, this.deps.git, this.layout, this.engine);
    if (!source.ok) throw new Error(source.message);
    return engineCorpus(this.deps.fs, source.value, chars);
  }
}

/** --live alone means the head's own port; a URL names a head elsewhere; absent, no live probe */
/** what a run measured: this checkout's pin and pack only where card legs ran them. A live-only run measured the head
 *  at its URL, whose engine and pack are whatever that machine serves (a rented box serves its public pack) */
export function runProvenance(
  head: Pick<Head, "name" | "served" | "speculative">,
  engine: Pick<Engine, "sha7">,
  card: GateCard | null,
  live: string | null,
): RunProvenance {
  const draft = card && head.speculative ? head.speculative.cache : null;
  return {
    head: head.name,
    engine: card ? engine.sha7 : null,
    served: card ? head.served.sha256 : null,
    live,
    gpu: card?.gpu ?? null,
    cache: card?.cache ?? null,
    tier: card?.tier ?? null,
    draft_cache: draft ? { k: draft.k, v: draft.v } : null,
  };
}

function liveUrlOf(head: Head, live: string | boolean | undefined): string | null {
  if (live === true) return `http://127.0.0.1:${head.port}`;
  return live || null;
}

/** what a live-only run hands the probes: no card was claimed, so no leg can start */
const noGateLegs: GateLegs = {
  leg(options: LegOptions): Promise<never> {
    return Promise.reject(
      new Error(`no gate card in this run, so the ${options.label} leg cannot start`),
    );
  },
};
