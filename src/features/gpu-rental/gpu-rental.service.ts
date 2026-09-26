// A rented card as a head. `up` rents the cheapest box that fits, ships rig and the head to it,
// brings the head up there with rig's own steps (prepare, fetch, build, derive, serve), and
// opens the tunnel and the idle timer here. `down` destroys the box and re-reads the listing (a
// destroy that did not answer is still billing). `idleCheck` is the cost control, run by the
// timer. `bench` runs the head's gates on the box and the live probes through the tunnel, the
// evidence pulled back. The box gets no credential: the server binds loopback and ssh is the
// only way in. Nor a private [derive] asset unless `--private` asks: a rented box is someone
// else's machine, so by default it derives and serves the head's public pack. Remote command lines
// live in rented-box.ts; the tunnel's http in head-endpoint.ts.
import { basename, join } from "node:path";
import type { Engine } from "../../shared/engine/engine.ts";
import { cacheRefusal, tierCache } from "../../shared/head/cache-formats.ts";
import type { Head } from "../../shared/head/head.ts";
import { deriveAsset, tierSpeculates } from "../../shared/head/head-config.ts";
import { HeadEndpoint, type Serving } from "../../shared/head/head-endpoint.ts";
import { pickTier } from "../../shared/head/tier.ts";
import type { Layout } from "../../shared/layout.ts";
import type {
  Clock,
  FileSystem,
  Http,
  Instance,
  Log,
  Offer,
  Rental,
  Shell,
  Ssh,
  Systemd,
} from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import { type BoxState, billedCost, billedHours, RentalState } from "./box-state.ts";
import { loadVastConfig, offerQuery, type VastConfig } from "./rental-config.ts";
import { RentedBox } from "./rented-box.ts";
import {
  IDLE_SERVICE,
  IDLE_TIMER,
  renderIdleService,
  renderIdleTimer,
  renderTunnelUnit,
  TUNNEL_UNIT,
} from "./units.ts";

/** what the live probes need from head-gating, handed in by main.ts: this feature never imports it */
export interface LiveGate {
  run(
    head: Head,
    options: { live: string; only: string[] },
  ): Promise<Result<{ dir: string; pass: boolean }>>;
}

export interface RentGpuDeps {
  fs: FileSystem;
  shell: Shell;
  http: Http;
  rental: Rental;
  ssh: Ssh;
  systemd: Systemd;
  clock: Clock;
  log: Log;
  gate: LiveGate;
  /** how to invoke rig again, for the idle timer's unit */
  self: readonly string[];
  home: string;
}

export interface RentOptions {
  /** the card class as vast names it: RTX_5090, H100_SXM … */
  gpu: string;
  gpus?: number | undefined;
  maxDph?: number | undefined;
  geo?: string | undefined;
  /** query the market and pick, create nothing */
  dryRun?: boolean;
  /** rent a card this engine is not measured on, for a benchmark */
  allowArch?: string | undefined;
  /** the box's disk, over vast.toml's disk_gb: a job that keeps data on the box needs more */
  diskGb?: number | undefined;
  /** this box's idle budget, over vast.toml's idle_minutes: a job the server's counters cannot
   *  see (training beside it) needs longer, and the budget is still the cost cap */
  idleMinutes?: number | undefined;
  /** ship the head's private [derive] assets too, so the box serves this machine's pack */
  private?: boolean;
}

export interface RentReport {
  kind: "up";
  instanceId: number;
  gpu: string;
  cap: string;
  dph: number;
  sshHost: string;
  sshPort: number;
  localUrl: string;
  serving: Serving;
}

export interface DryRunReport {
  kind: "dry-run";
  pick: Offer;
  query: string;
}

export interface DownReport {
  destroyed: number[];
  hours?: number;
  cost?: number;
}

export interface StatusReport {
  box: BoxState | null;
  /** "unread" when the market could not be read: no evidence the box is gone */
  listed: boolean | "unread";
  status?: string;
  hours?: number;
  cost?: number;
  tunnelActive: boolean;
  healthy: boolean;
  /** whether the cost control runs; a box listed (or unread) without it has it re-armed by status */
  idleTimer: "active" | "inactive" | "re-armed" | "none";
  /** how the timer's last check ended: an active timer whose checks fail controls nothing */
  idleCheck?: "ok" | "failed" | "unread";
}

export interface IdleReport {
  action: "no-box" | "active" | "changed" | "idle" | "destroyed";
  idleMinutes?: number;
}

export interface BenchReport {
  /** the box's gate run, pulled here */
  remote: string;
  /** the live probes' run through the tunnel */
  local: string;
  pass: boolean;
}

interface Pick {
  offer: Offer;
  query: string;
  label: string;
}

/** a box with its ssh endpoint known */
type ReachableBox = BoxState & { sshHost: string; sshPort: number };

type EngineSource =
  | { kind: "prebuilt" }
  | { kind: "cached"; tarball: string }
  | { kind: "compile" };

const SERVER_HEALTHY_TIMEOUT_MS = 600_000;
/** the card's utilization at which the box is in use whatever its server says: an idle
 *  llama-server with its model loaded reads 0 */
const GPU_BUSY_PCT = 10;

export class RentGpu {
  private readonly state: RentalState;

  /** [loaded] is engine.toml as this binary read it, failed or not: down, status and idle-check
   *  never read the pin, so a checkout whose engine.toml this binary cannot read stops only up and
   *  bench, never the cost control of a box that bills (2026-09-25, 8:59-11:09 PM CT: fourteen
   *  idle checks exited 1 on `miscompilers: Invalid key` while box 52647843 billed) */
  constructor(
    private readonly deps: RentGpuDeps,
    private readonly layout: Layout,
    private readonly loaded: Result<Engine>,
  ) {
    this.state = new RentalState(deps.fs, layout);
  }

  /** the pin; up and bench return the load failure before anything reads it */
  private get engine(): Engine {
    if (!this.loaded.ok) throw new Error(this.loaded.message);
    return this.loaded.value;
  }

  async up(head: Head, options: RentOptions): Promise<Result<RentReport | DryRunReport>> {
    if (!this.loaded.ok) return this.loaded;
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;

    const ready = await this.preflight(options);
    if (!ready.ok) return ready;

    const pick = await this.pickOffer(config.value, options);
    if (!pick.ok) return pick;
    const fits = this.servesOn(head, pick.value.offer);
    if (!fits.ok) return fits;
    if (options.dryRun) {
      this.deps.log.info("dry run: no box created");
      return ok({ kind: "dry-run", pick: pick.value.offer, query: pick.value.query });
    }

    const box = await this.createBox(config.value, head, pick.value.offer, options);
    if (!box.ok) return box;
    const remote = this.remote(config.value, box.value);

    const shipped = await this.shipPayload(remote, head, options.private ?? false);
    if (!shipped.ok) return shipped;
    const engine = await this.engineSource(remote, pick.value.offer);

    const broughtUp = await this.bringUp(
      remote,
      box.value,
      head,
      pick.value.offer,
      engine,
      options,
    );
    if (!broughtUp.ok) return broughtUp;
    await remote.startServer(head.name);

    const endpoint = await this.openTunnel(
      config.value,
      box.value,
      head,
      idleBudget(config.value, box.value),
    );
    if ((await endpoint.waitHealthy(SERVER_HEALTHY_TIMEOUT_MS)) !== "healthy") {
      const tail = await remote.serverLogTail(30);
      const message = `the server did not become healthy through the tunnel (box ${box.value.instanceId} left running):\n${tail}`;
      return fail(ExitCode.Failure, message);
    }
    await this.deps.fs.remove(this.state.idleFile);
    await this.armIdleTimer();

    const serving = await endpoint.serving();
    // the answer must be the pinned pack this local head resolves to, the way head-bringup
    // already checks for a local start: a mismatch here is either a leftover process on the
    // port or the box's own derive resolving differently than this machine did, and either way
    // READY would be false — not a cosmetic mismatch to warn past on a fresh, still-billing box
    const expected = boxServedFile(head, options.private ?? false);
    if (serving.model !== expected) {
      const message = `the box serves ${serving.model}, not the pinned ${expected} (box ${box.value.instanceId} left running for inspection)`;
      return fail(ExitCode.Failure, message);
    }
    const offer = pick.value.offer;
    const undrived = head.undrived ? ` UNDRIVED: ${head.undrived}` : "";
    this.deps.log.info(
      `READY: ${pick.value.label} box ${box.value.instanceId} at $${offer.dph.toFixed(3)}/h — ${serving.model}, ${serving.slots} slots, via ${endpoint.url}; idle timer armed (${idleExposure(idleBudget(config.value, box.value), offer.dph)})${undrived}`,
    );
    return ok({
      kind: "up",
      instanceId: box.value.instanceId,
      gpu: offer.gpu,
      cap: offer.computeCap,
      dph: offer.dph,
      sshHost: box.value.sshHost,
      sshPort: box.value.sshPort,
      localUrl: endpoint.url,
      serving,
    });
  }

  async down(options: { all?: boolean } = {}): Promise<Result<DownReport>> {
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;

    // The timer and the tunnel stop only once the box is confirmed gone: a destroy that fails
    // (vast 5xx or 429, the box still listed) leaves a box billing, and its idle check must run
    // again rather than have been disabled by the attempt.
    const destroyed: number[] = [];
    let billed: { hours: number; cost: number } | undefined;
    const box = await this.state.box();
    if (box) {
      const gone = await this.destroy(box);
      if (!gone.ok) return gone;
      billed = gone.value;
      destroyed.push(box.instanceId);
    } else {
      this.deps.log.info(`no box in ${this.state.dir}`);
    }
    await this.stopLocalUnits();

    if (options.all) {
      for (const instance of await this.forgottenBoxes(config.value, destroyed)) {
        await this.deps.rental.destroy(instance.id);
        destroyed.push(instance.id);
        this.deps.log.info(`destroyed forgotten ${instance.label} box ${instance.id}`);
      }
    }
    return ok({ destroyed, ...billed });
  }

  async status(): Promise<Result<StatusReport>> {
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;
    const box = await this.state.box();
    const tunnelActive = await this.deps.systemd.isActive(TUNNEL_UNIT);
    const healthy = await this.endpoint(config.value).healthy();
    if (!box) return ok({ box: null, listed: false, tunnelActive, healthy, idleTimer: "none" });

    const listing = await this.listing(box);
    const hours = billedHours(box, this.deps.clock.now());
    // A box billing with its cost control dead is the one state status must not only report:
    // 2026-09-24 the timer went inactive at 09:08 with no stop in the journal, and box 52390478
    // billed idle until a person noticed (8.3 h, ~$11.63). A market that cannot be read is no
    // evidence the box is gone, so the timer is re-armed then too.
    const timerActive = await this.deps.systemd.isActive(IDLE_TIMER);
    let idleTimer: StatusReport["idleTimer"] = timerActive ? "active" : "inactive";
    if (listing !== null && !timerActive) {
      await this.armIdleTimer();
      const billing = listing === "unread" ? "may be billing" : "is billing";
      this.deps.log.warn(
        `box ${box.instanceId} ${billing} and ${IDLE_TIMER} was not running: re-armed it (idle budget ${idleBudget(config.value, box)} min)`,
      );
      idleTimer = "re-armed";
    }
    // An active timer whose checks fail is as dead as a stopped one (2026-09-25: fourteen checks
    // exited 1 while the timer read active), so the last check's result is part of the answer.
    const lastCheck = await this.deps.systemd.lastResult(IDLE_SERVICE);
    const idleCheck: StatusReport["idleCheck"] =
      lastCheck === null ? "unread" : lastCheck === "success" ? "ok" : "failed";
    if (idleCheck === "failed") {
      this.deps.log.warn(
        `${IDLE_SERVICE}'s last run ended ${lastCheck}: the box's cost control is not running; see journalctl --user -u ${IDLE_SERVICE}`,
      );
    }
    return ok({
      box,
      listed: listing === "unread" ? "unread" : listing !== null,
      ...(listing !== null && listing !== "unread" ? { status: listing.status } : {}),
      hours,
      cost: billedCost(hours, box.dph),
      tunnelActive,
      healthy,
      idleTimer,
      idleCheck,
    });
  }

  /** the market's listing of the box: null when it lists no such box, "unread" when the market
   *  could not be read (a 429, an expired key, no CLI), which is no evidence either way */
  private async listing(box: BoxState): Promise<Instance | null | "unread"> {
    try {
      return await this.deps.rental.show(box.instanceId);
    } catch (error) {
      this.deps.log.warn(`vast could not be read: ${(error as Error).message}`);
      return "unread";
    }
  }

  /** the timer's check: the server's token counters through the tunnel, unchanged for
   *  idle_minutes, and the card idle as the market reads it, means nobody is using the box, so it
   *  goes down. A server that does not answer is no evidence of idleness: a box running other
   *  work on its card (a gate, a build, a training run) never serves, and was destroyed with
   *  that work still running (2026-09-24, 12.4 h in, the GPU at 100 %). */
  async idleCheck(): Promise<Result<IdleReport>> {
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;
    const box = await this.state.box();
    if (!box) return ok({ action: "no-box" });

    const now = this.deps.clock.now();
    const activity = await this.endpoint(config.value).activity();
    const listing = await this.listing(box);
    const gpuUtil = listing === "unread" ? undefined : listing?.gpuUtil;
    // The card's reading is the other half of the evidence: vast unreadable, or listing the box
    // running with no sample (its gpu_util is number | null), leaves a busy card looking like an
    // idle one. Only a box it lists as not running, or no longer lists, needs no reading.
    const cardUnread =
      listing === "unread"
        ? "vast could not be read"
        : listing?.status === "running" && gpuUtil === undefined
          ? "vast listed no GPU reading"
          : undefined;
    // Neither the server nor the card readable is no evidence at all: `vast bench` stops the
    // server for its gates, and a market outage over the whole budget would destroy the box
    // mid-gate. The check counts nothing, the clock keeps its last change, and the run fails, so
    // the unit's last result (what `vast status` reads) shows a cost control that saw nothing.
    if (cardUnread && activity.key === "unreachable") {
      const message =
        listing === "unread"
          ? `neither the server nor vast answered: box ${box.instanceId} not counted, its idle clock unchanged`
          : `the server did not answer and ${cardUnread}: box ${box.instanceId} not counted, its idle clock unchanged`;
      return fail(ExitCode.Failure, message);
    }
    const gpuBusy = gpuUtil !== undefined && gpuUtil >= GPU_BUSY_PCT;
    const last = await this.state.idle();
    if (activity.busy > 0 || gpuBusy || activity.key !== last?.key) {
      await this.state.saveIdle({ key: activity.key, ts: now });
      return ok({ action: activity.busy > 0 || gpuBusy ? "active" : "changed" });
    }
    // an idle server beside an unread card counts nothing either: a busy card is never idle,
    // whatever the server says
    if (cardUnread) {
      const message = `the server is idle but ${cardUnread}: box ${box.instanceId} not counted, its idle clock unchanged`;
      return fail(ExitCode.Failure, message);
    }

    const idleMinutes = Math.floor((now - last.ts) / 60_000);
    if (idleMinutes < idleBudget(config.value, box)) return ok({ action: "idle", idleMinutes });
    const card = gpuUtil === undefined ? "no GPU reading" : `GPU ${gpuUtil} %`;
    this.deps.log.info(
      `idle for ${idleMinutes} min (${activity.key}, ${card}) — destroying the box`,
    );
    const down = await this.down();
    if (!down.ok) return down;
    return ok({ action: "destroyed", idleMinutes });
  }

  /** the head's gates on the box (its one card, the server stopped meanwhile), the run pulled
   *  back, then the live probes through the tunnel */
  async bench(head: Head): Promise<Result<BenchReport>> {
    if (!this.loaded.ok) return this.loaded;
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;
    const box = await this.state.box();
    if (!box) return fail(ExitCode.Failure, "no box; run: rig vast up");
    const remote = this.remote(config.value, box);

    const pulledRuns = join(this.state.pulledRunsDir, pulledRunName(box));
    await this.deps.fs.mkdirp(pulledRuns);
    await remote.stopServer();
    const gate = await remote.rig(`gate ${head.name} --gpu 0 --json`);
    const remoteRunDir = gateRunDir(gate.stdout);
    if (remoteRunDir) await this.pullGateRun(remote, remoteRunDir, pulledRuns);
    await remote.startServer(head.name);

    const endpoint = this.endpoint(config.value);
    if ((await endpoint.waitHealthy(SERVER_HEALTHY_TIMEOUT_MS)) !== "healthy") {
      const message = `the server did not come back healthy after the gates (box ${box.instanceId})`;
      return fail(ExitCode.Failure, message);
    }
    if (gate.code !== 0) {
      this.deps.log.error(lastLines(gate.stderr, 10));
      return fail(ExitCode.Failure, `gates FAILED on the box (exit ${gate.code}) — ${pulledRuns}`);
    }

    const live = await this.deps.gate.run(head, {
      live: endpoint.url,
      only: ["sessions", "concurrency"],
    });
    if (!live.ok) return live;
    return ok({ remote: pulledRuns, local: live.value.dir, pass: live.value.pass });
  }

  // --- up, step by step ---

  /** nothing rented yet, the binary to ship exists, the key vast will get exists, funds cover
   *  a box; a dry run is a market query and may run beside a live box */
  private async preflight(options: RentOptions): Promise<Result<void>> {
    if (!options.dryRun && (await this.state.box())) {
      const message = `a box already exists (${this.state.instanceFile}); run: rig vast down`;
      return fail(ExitCode.Failure, message);
    }
    const rigBinary = join(this.layout.root, "dist", "rig");
    if (!options.dryRun && !(await this.deps.fs.exists(rigBinary))) {
      const message = `${rigBinary} is missing — the box runs the compiled rig (run: bun run build)`;
      return fail(ExitCode.Failure, message);
    }
    const pubkeyPath = join(this.deps.home, ".ssh", "id_ed25519.pub");
    if (!(await this.deps.fs.exists(pubkeyPath))) {
      return fail(ExitCode.Failure, `no ${pubkeyPath} to register with vast`);
    }

    const funds = await this.deps.rental.funds();
    this.deps.log.info(`vast funds: $${funds.toFixed(2)}`);
    if (!options.dryRun && funds < 1) {
      return fail(ExitCode.Failure, `funds $${funds.toFixed(2)} — top up before creating a box`);
    }

    const pubkey = (await this.deps.fs.readText(pubkeyPath)).trim();
    if (!(await this.deps.rental.hasSshKey(pubkey))) {
      await this.deps.rental.registerSshKey(pubkey);
      this.deps.log.info(`registered ${pubkeyPath} with vast`);
    }
    return ok(undefined);
  }

  /** the tier serve would give the offered card and the cache formats it would run there, refused before the rental:
   *  on the box they are refused only by serve, after the fetch, the build and the derive, on a card paid by the hour */
  private servesOn(head: Head, offer: Offer): Result<void> {
    const tier = pickTier(head, offer.gpuRamMiB);
    if (!tier.ok)
      return fail(tier.code, `REFUSING to rent ${offer.gpu}: ${head.name} ${tier.message}`);
    const draft = tierSpeculates(head, tier.value) ? head.speculative?.cache : undefined;
    const refusal = cacheRefusal(this.engine, tierCache(head, tier.value), draft);
    if (refusal)
      return fail(
        ExitCode.Unsupported,
        `REFUSING to rent ${offer.gpu}: on its tier (${tier.value.min_vram_mib} MiB) ${refusal}`,
      );
    return ok(undefined);
  }

  /** the market queried, the offers kept beside the box's state, the cheapest one picked;
   *  a card this engine is not measured on needs --allow-arch */
  private async pickOffer(config: VastConfig, options: RentOptions): Promise<Result<Pick>> {
    const query = offerQuery(config, options.gpu, {
      maxDph: options.maxDph,
      geo: options.geo,
      diskGb: options.diskGb ?? config.rental.disk_gb,
      gpus: options.gpus,
    });
    const offers = await this.deps.rental.searchOffers(query);
    await this.deps.fs.mkdirp(this.state.dir);
    await this.deps.fs.writeText(this.state.path("offers.json"), JSON.stringify(offers, null, 2));
    if (offers.length === 0) return fail(ExitCode.Failure, `no offer matches: ${query}`);
    for (const offer of offers.slice(0, 5)) this.deps.log.info(`  ${describeOffer(offer)}`);

    const offer = offers[0]!;
    if (!this.engine.supports(offer.computeCap) && !options.allowArch) {
      const measured = this.engine.archs.map((arch) => `sm_${arch.cap}`).join(", ");
      const message = `${offer.gpu} is sm_${offer.computeCap}, not a card this engine is measured on (${measured}); --allow-arch ${offer.computeCap} rents it for a benchmark`;
      return fail(ExitCode.Unsupported, message);
    }
    const label = offer.gpus > 1 ? `${offer.gpus}× ${offer.gpu}` : offer.gpu;
    this.deps.log.info(
      `pick: offer ${offer.id} ${label} at $${offer.dph.toFixed(3)}/h, ${offer.geo}, sm_${offer.computeCap}`,
    );
    return ok({ offer, query, label });
  }

  /** the instance created and recorded, then waited for: running, with an ssh endpoint that
   *  answers; a box that never gets there is destroyed, not left billing */
  private async createBox(
    config: VastConfig,
    head: Head,
    offer: Offer,
    options: RentOptions,
  ): Promise<Result<ReachableBox>> {
    const instanceId = await this.deps.rental.create(offer.id, {
      image: config.rental.image,
      diskGb: options.diskGb ?? config.rental.disk_gb,
      label: config.rental.label,
    });
    const box: BoxState = {
      instanceId,
      offerId: offer.id,
      gpu: offer.gpu,
      gpus: offer.gpus,
      cap: offer.computeCap,
      dph: offer.dph,
      geo: offer.geo,
      createdAt: this.deps.clock.now(),
      head: head.name,
      ...(options.idleMinutes !== undefined ? { idleMinutes: options.idleMinutes } : {}),
    };
    await this.state.saveBox(box);
    this.deps.log.info(
      `instance ${instanceId} created; waiting for it to run (image pull + vast's sshd install)`,
    );

    const running = await this.waitRunning(instanceId);
    if (!running) {
      await this.abandon(instanceId, "never reached running");
      return fail(ExitCode.Failure, `instance ${instanceId} never reached running; destroyed`);
    }
    if (!running.sshHost || !running.sshPort) {
      await this.abandon(instanceId, "no ssh endpoint");
      return fail(ExitCode.Failure, `instance ${instanceId} has no ssh endpoint; destroyed`);
    }
    const reachable: ReachableBox = { ...box, sshHost: running.sshHost, sshPort: running.sshPort };
    await this.state.saveBox(reachable);
    await this.deps.fs.remove(this.state.knownHosts);

    if (!(await this.waitSsh(this.remote(config, reachable)))) {
      await this.abandon(instanceId, "ssh never answered");
      const message = `ssh never answered at ${reachable.sshHost}:${reachable.sshPort}; destroyed ${instanceId}`;
      return fail(ExitCode.Failure, message);
    }
    this.deps.log.info(`ssh up: root@${reachable.sshHost}:${reachable.sshPort}`);
    return ok(reachable);
  }

  /** rig, the head and the engine pin, as one tarball; the head's private [derive] assets only
   *  with --private */
  private async shipPayload(
    remote: RentedBox,
    head: Head,
    shipPrivate: boolean,
  ): Promise<Result<void>> {
    const payload = this.state.path("payload.tar.gz");
    const held = shipPrivate ? [] : privateAssets(head);
    if (held.length > 0) {
      this.deps.log.info(
        `kept here: ${held.join(", ")} (private); the box serves ${boxServedFile(head, false)} — --private ships them`,
      );
    }
    const tar = await this.deps.shell.run(
      [
        "tar",
        "-C",
        this.layout.root,
        ...held.map((path) => `--exclude=heads/${head.name}/${path}`),
        "-czf",
        payload,
        "dist/rig",
        `heads/${head.name}`,
        "engine/engine.toml",
      ],
      { timeoutMs: 300_000 },
    );
    if (tar.code !== 0) return fail(ExitCode.Failure, `tar: ${tar.stderr.trim()}`);
    await remote.receivePayload(payload);
    return ok(undefined);
  }

  /** how the box gets its engine: the pin's prebuilt for this sm, installed as on any machine
   *  (prepare installs no compiler for a card a prebuilt covers); else a build of this sm cached
   *  from an earlier box, shipped; else a compile on the box */
  private async engineSource(remote: RentedBox, offer: Offer): Promise<EngineSource> {
    if (this.engine.prebuiltFor(offer.computeCap)) return { kind: "prebuilt" };
    const tarball = await this.shipCachedBuild(remote, offer);
    return tarball ? { kind: "cached", tarball } : { kind: "compile" };
  }

  /** a build for this card's sm from an earlier box skips the box's compile; its name, or null */
  private async shipCachedBuild(remote: RentedBox, offer: Offer): Promise<string | null> {
    const name = this.buildTarballName(offer);
    const cached = join(this.state.cachedBuildsDir, name);
    if (!(await this.deps.fs.exists(cached))) return null;
    await remote.receiveBuild(cached, name);
    this.deps.log.info(`pushed cached build ${name}`);
    return name;
  }

  /** rig's own steps on the box: prepare, fetch (the 7 GB download overlaps the build), build
   *  (or unpack the cached build), derive. A step that fails leaves the box running for
   *  inspection and says so. */
  private async bringUp(
    remote: RentedBox,
    box: ReachableBox,
    head: Head,
    offer: Offer,
    engine: EngineSource,
    options: RentOptions,
  ): Promise<Result<void>> {
    const allowArch = options.allowArch ? ` --allow-arch ${options.allowArch}` : "";
    const step = (label: string, args: string) => this.remoteStep(remote, box, label, args);

    if (!(await step("prepare", `prepare --gpu 0${allowArch}`))) {
      return fail(ExitCode.Failure, `prepare failed on box ${box.instanceId}`);
    }
    const fetching = remote.rig(`fetch ${head.name}`);
    const built =
      engine.kind === "prebuilt"
        ? await step("build", `build --gpu 0${allowArch}`)
        : engine.kind === "cached"
          ? await step(
              "build",
              `build --gpu 0 --from-tarball ${remote.buildTarball(engine.tarball)}${allowArch}`,
            )
          : await this.buildOnBox(remote, offer, step, allowArch);
    const fetched = await fetching;
    if (!built) return fail(ExitCode.Failure, `build failed on box ${box.instanceId}`);
    if (fetched.code !== 0) {
      const message = `fetch failed on box ${box.instanceId}: ${lastLines(fetched.stderr, 5)}`;
      return fail(ExitCode.Failure, message);
    }
    if (!(await this.deriveStep(remote, box, head))) {
      return fail(ExitCode.Failure, `derive failed on box ${box.instanceId}`);
    }
    return ok(undefined);
  }

  /** a portable build on the box, cached here for the next box with this sm */
  private async buildOnBox(
    remote: RentedBox,
    offer: Offer,
    step: (label: string, args: string) => Promise<boolean>,
    allowArch: string,
  ): Promise<boolean> {
    const started = this.deps.clock.now();
    this.deps.log.info(
      `no cached build for sm_${offer.computeCap} — building on the box (5–20 min)`,
    );
    const built = await step("build", `build --gpu 0 --portable${allowArch}`);
    if (!built) return false;

    const name = this.buildTarballName(offer);
    await this.deps.fs.mkdirp(this.state.cachedBuildsDir);
    await remote.sendBuild(name, join(this.state.cachedBuildsDir, name));
    const seconds = Math.round((this.deps.clock.now() - started) / 1000);
    this.deps.log.info(`built in ${seconds} s; cached ${name} in ${this.state.cachedBuildsDir}`);
    return true;
  }

  private async remoteStep(remote: RentedBox, box: BoxState, label: string, args: string) {
    const result = await remote.rig(args);
    if (result.code === 0) return true;
    this.deps.log.error(
      `${label} failed on the box (exit ${result.code}); box ${box.instanceId} left running for inspection:\n${lastLines(result.stderr, 20)}`,
    );
    return false;
  }

  /** derive on the box, `--json` so its own report reaches this console: every other step's
   *  stdout is discarded on success (`remoteStep` above), which would otherwise swallow the
   *  box's own undrived notice — printed on the box's stderr, three hops from the operator (this
   *  machine's log, never forwarded here) — as if nothing happened */
  private async deriveStep(remote: RentedBox, box: BoxState, head: Head): Promise<boolean> {
    const result = await remote.rig(`derive ${head.name} --json`);
    if (result.code !== 0) {
      this.deps.log.error(
        `derive failed on the box (exit ${result.code}); box ${box.instanceId} left running for inspection:\n${lastLines(result.stderr, 20)}`,
      );
      return false;
    }
    const report = parseJson<{ state?: string; reason?: string }>(result.stdout);
    if (report?.reason) {
      this.deps.log.warn(`box ${box.instanceId}: ${report.reason}`);
    }
    return true;
  }

  /** the tunnel unit to the box's ssh endpoint and the idle timer, installed and started */
  private async openTunnel(
    config: VastConfig,
    box: ReachableBox,
    head: Head,
    idleMinutes: number,
  ): Promise<HeadEndpoint> {
    await this.deps.fs.writeText(
      this.state.tunnelEnv,
      `HOST=${box.sshHost}\nPORT=${box.sshPort}\n`,
    );
    await this.installUnits(config, head, idleMinutes);
    await this.deps.systemd.restart(TUNNEL_UNIT);
    return this.endpoint(config);
  }

  private async installUnits(config: VastConfig, head: Head, idleMinutes: number): Promise<void> {
    const dir = this.deps.systemd.unitDir();
    await this.deps.fs.mkdirp(dir);
    const units: Record<string, string> = {
      [TUNNEL_UNIT]: renderTunnelUnit({
        tunnelEnv: this.state.tunnelEnv,
        knownHosts: this.state.knownHosts,
        localPort: config.rental.local_port,
        remotePort: head.port,
      }),
      [IDLE_SERVICE]: renderIdleService({
        self: this.deps.self,
        idleMinutes,
      }),
      [IDLE_TIMER]: renderIdleTimer(),
    };
    let changed = false;
    for (const [name, text] of Object.entries(units)) {
      const path = join(dir, name);
      const current = (await this.deps.fs.exists(path)) ? await this.deps.fs.readText(path) : null;
      if (current === text) continue;
      await this.deps.fs.replaceText(path, text);
      changed = true;
    }
    if (changed) await this.deps.systemd.daemonReload();
  }

  // --- down ---

  private async stopLocalUnits(): Promise<void> {
    try {
      await this.deps.systemd.disable(IDLE_TIMER);
    } catch {
      /* not installed yet */
    }
    for (const unit of [IDLE_TIMER, TUNNEL_UNIT]) {
      try {
        await this.deps.systemd.stop(unit);
      } catch {
        /* not installed yet */
      }
    }
  }

  /** the idle timer enabled as well as started, so a restart of the user manager or of this
   *  machine arms it again while the box bills; `down` disables it */
  private async armIdleTimer(): Promise<void> {
    await this.deps.systemd.enable(IDLE_TIMER);
    await this.deps.systemd.restart(IDLE_TIMER);
  }

  /** destroyed and confirmed gone from the listing, whatever the destroy call answered: a box
   *  still listed is still billing, and one the market no longer lists (destroyed from vast's
   *  console, or reclaimed by its host) is gone even when the call refuses it */
  private async destroy(box: BoxState): Promise<Result<{ hours: number; cost: number }>> {
    let refused = "";
    try {
      await this.deps.rental.destroy(box.instanceId);
    } catch (error) {
      refused = ` (the destroy failed: ${(error as Error).message})`;
    }
    await this.deps.clock.sleep(5000);
    const stillListed = (await this.deps.rental.list()).some((i) => i.id === box.instanceId);
    if (stillListed) {
      const message = `box ${box.instanceId} is STILL listed after destroy${refused} — it is billing, and its idle timer stays armed; run: vastai destroy instance ${box.instanceId} -y`;
      return fail(ExitCode.Failure, message);
    }
    const hours = billedHours(box, this.deps.clock.now());
    const cost = billedCost(hours, box.dph);
    this.deps.log.info(
      `destroyed box ${box.instanceId} after ${hours} h (~$${cost} at $${box.dph}/h)`,
    );
    await this.state.clear();
    return ok({ hours, cost });
  }

  /** boxes with our label that the state file no longer knows about */
  private async forgottenBoxes(config: VastConfig, known: number[]): Promise<Instance[]> {
    const listed = await this.deps.rental.list();
    return listed.filter((i) => i.label === config.rental.label && !known.includes(i.id));
  }

  // --- the box and the tunnel ---

  private remote(config: VastConfig, box: BoxState): RentedBox {
    return new RentedBox(this.deps.ssh, this.state.target(box), config.rental.remote_dir);
  }

  private endpoint(config: VastConfig): HeadEndpoint {
    const url = `http://127.0.0.1:${config.rental.local_port}`;
    return new HeadEndpoint(this.deps.http, this.deps.clock, url);
  }

  private buildTarballName(offer: Offer): string {
    return `engine-sm${offer.computeCap}-${this.engine.sha7}.tar.gz`;
  }

  private async pullGateRun(remote: RentedBox, remoteRunDir: string, into: string): Promise<void> {
    const tarball = join(into, "gate-run.tar.gz");
    await remote.sendGateRun(remoteRunDir, tarball);
    await this.deps.shell.run(["tar", "-C", into, "-xzf", tarball]);
    await this.deps.fs.remove(tarball);
  }

  private async waitRunning(instanceId: number): Promise<Instance | null> {
    for (let attempt = 0; attempt < 90; attempt++) {
      const instance = await this.deps.rental.show(instanceId);
      if (instance?.status === "running") return instance;
      await this.deps.clock.sleep(10_000);
    }
    return null;
  }

  private async waitSsh(remote: RentedBox): Promise<boolean> {
    for (let attempt = 0; attempt < 60; attempt++) {
      if (await remote.reachable(20_000)) return true;
      await this.deps.clock.sleep(5000);
    }
    return false;
  }

  private async abandon(instanceId: number, why: string): Promise<void> {
    this.deps.log.error(`instance ${instanceId} ${why}; destroying it`);
    await this.deps.rental.destroy(instanceId);
    await this.state.clear();
  }
}

/** the idle minutes that destroy this box: its own budget from `up`, else vast.toml's */
function idleBudget(config: VastConfig, box: BoxState): number {
  return box.idleMinutes ?? config.rental.idle_minutes;
}

/** the budget and what it lets a box nobody uses cost before it goes (box 52390478 was rented with
 *  2,880 minutes for a training run and billed idle for hours after its session ended, 2026-09-24) */
function idleExposure(minutes: number, dph: number): string {
  return `${minutes} min: up to ~$${billedCost(minutes / 60, dph).toFixed(2)} idle before it is destroyed`;
}

/** one market row as the log shows it */
function describeOffer(offer: Offer): string {
  const ram = `${(offer.gpuRamMiB / 1024).toFixed(0)}GB`;
  const cpu = offer.cpu.slice(0, 26).padEnd(26);
  return `$${offer.dph.toFixed(3)}/h  ${offer.gpu} ${ram} bw=${offer.bandwidth.toFixed(0)}GB/s  ${cpu} ram=${offer.ramGiB}GB  down=${offer.downMbps.toFixed(0)}Mb/s  rel=${offer.reliability.toFixed(3)}  cuda=${offer.cudaMaxGood}  ${offer.geo}  id=${offer.id}`;
}

/** local/rented-box/pulled-runs/<card>-<instance>: h100-sxm-1000 */
function pulledRunName(box: BoxState): string {
  return `${box.gpu.replace(/\s+/g, "-").toLowerCase()}-${box.instanceId}`;
}

/** the run directory `rig gate --json` printed on the box, or null when it printed none */
function gateRunDir(stdout: string): string | null {
  try {
    return (JSON.parse(stdout) as { dir: string }).dir;
  } catch {
    return null;
  }
}

/** a `--json` report a remote `rig` command printed, or null when it printed none */
function parseJson<T>(stdout: string): T | null {
  try {
    return JSON.parse(stdout) as T;
  } catch {
    return null;
  }
}

const lastLines = (text: string, count: number) => text.trim().split("\n").slice(-count).join("\n");

/** the head's private [derive] assets (no url), head-relative: what a box gets only with --private */
function privateAssets(head: Head): string[] {
  return (head.derive ?? []).map(deriveAsset).flatMap((asset) => (asset.url ? [] : [asset.path]));
}

/** the pack the box serves: this machine's with its private assets, else what a machine without
 *  them serves (head.ts): the [public] pack when head.toml declares one, else the source pack */
function boxServedFile(head: Head, shipPrivate: boolean): string {
  if (shipPrivate || privateAssets(head).length === 0) return basename(head.servedPath);
  return basename(head.declaredPublic?.path ?? head.sourcePath);
}
