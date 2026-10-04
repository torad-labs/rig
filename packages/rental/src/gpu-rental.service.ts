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
import type {
  Clock,
  FileSystem,
  Git,
  Http,
  Instance,
  Layout,
  Log,
  Offer,
  Rental,
  Shell,
  Ssh,
  Systemd,
} from "@rig/core";
import { BUILT_FROM, ExitCode, fail, ok, type Result } from "@rig/core";
import type { Engine } from "@rig/engine";
import {
  cacheRefusal,
  describeProfile,
  type Head,
  HeadEndpoint,
  pickProfile,
  privateAssets,
  profileCache,
  profileSpeculates,
  type Serving,
} from "@rig/head";
import { publishedImage } from "@rig/image";
import { type BoxState, billedCost, billedHours, RentalState } from "./box-state.ts";
import { headDiskGb, headPackBytes } from "./box-template.ts";
import { DownloadRates, describeRate, downloadRate } from "./download-rates.ts";
import { loadVastConfig, MAX_HOURS, offerQuery, type VastConfig } from "./rental-config.ts";

export { MAX_HOURS } from "./rental-config.ts";

import { RentedBox } from "./rented-box.ts";
import { type OfferEstimate, rankOffers, templateQuery } from "./template-offers.ts";
import {
  IDLE_CHECK_MINUTES,
  renderIdleService,
  renderIdleTimer,
  renderStopService,
  renderStopTimer,
  renderTunnelUnit,
} from "./units.ts";

/** the card's utilization, as vast lists it, at which a box `vast up` rented is in use whatever its server says: an idle
 *  llama-server with its model loaded reads 0. A template box's guard reads its container's CPU instead (box-guard.ts) */
const GPU_BUSY_PCT = 10;
/** a box receiving this much on average over a check's window is working, whatever its card and server read: a pack
 *  being pulled (134 GB at 251 Mb/s is 70 minutes of an idle card), an image pushed to it, a clone. 512 KB/s is 4 Mb/s;
 *  an idle box's ssh session and a log tail are a few KB/s. */
const DOWNLOAD_BUSY_KBPS = 512;
/** how far back a read of the box's sampler looks: two checks' worth, so one tick the timer missed is still covered */
const CARD_WINDOW_SECONDS = 2 * IDLE_CHECK_MINUTES * 60;
/** what the box needs beyond the pack: the engine, the logs, and the derive step's room */
const PACK_SLACK_BYTES = 5e9;

/** what the live probes need from head-gating, handed in by main.ts: this feature never imports it */
export interface LiveGate {
  run(
    head: Head,
    options: { live: string; only: string[] },
  ): Promise<Result<{ dir: string; pass: boolean }>>;
}

/** what `rig --built-from` prints when bun run build compiled it: a tree id, -dirty after it when a source was not the
 *  tree's */
const STAMP = new RegExp(`^([0-9a-f]{40}|[0-9a-f]{64})(${BUILT_FROM.dirty})?$`);

export interface RentGpuDeps {
  fs: FileSystem;
  shell: Shell;
  /** HEAD's tree, which the dist/rig a box is shipped must have been built from */
  git: Git;
  http: Http;
  rental: Rental;
  ssh: Ssh;
  systemd: Systemd;
  clock: Clock;
  log: Log;
  gate: LiveGate;
  /** how to invoke rig again, for the idle timer's unit */
  self: readonly string[];
  /** the vast CLI by its absolute path, for the hard stop's lines that run no rig code (units.ts) */
  vastai: string;
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
  /** the host's download speed under which an offer is never rented, in Mb/s: a pack of 134 GB takes about 70
   *  minutes at 251 Mb/s and 3.5 minutes at 5 Gb/s. With no offer above it the rent fails and names the fastest. */
  minDownMbps?: number | undefined;
  /** this box's idle budget, over vast.toml's idle_minutes: a job the server's counters cannot
   *  see (training beside it) needs longer, and the budget is still the cost cap */
  idleMinutes?: number | undefined;
  /** this box's hard stop, in hours after its create, over vast.toml's max_hours */
  maxHours?: number | undefined;
  /** ship the head's private [derive] assets too, so the box serves this machine's pack */
  private?: boolean;
}

/** the image a `--vm` box comes up from: a VM's image comes from vast's own KVM repository, fully qualified
 *  (docs.vast.ai/guides/instances/virtual-machines), and a plain one so the gate's driver-only checks hold */
export const VM_IMAGE = "docker.io/vastai/kvm:ubuntu_terminal";

/** How long rig waits for ssh once vast lists a box running, in tries of up to 25 s (a refused connection costs 5). A
 *  container answers within the first few minutes; a VM boots an operating system inside the container, vast's docs say
 *  its boot is "slower" without a figure, and the first one rig rented was still refusing connections after the 5
 *  minutes a container is given (instance 53930104, Oct 2: running by 22:46Z, "Connection refused" until 22:51Z). */
const SSH_ATTEMPTS = 60;
export const VM_SSH_ATTEMPTS = 240;

/** What a KVM box is created with, because some KVM hosts' sshd refuses the key vast writes: the file ends up owned by
 *  a uid the VM has no user for, and sshd's StrictModes says no (vast-ai/vast-cli#336, open, repaired by this on-start
 *  on two reporters' instances; a chmod alone is not enough, the file needs the chown). On-start runs after vast
 *  writes the key, so it is the one place the repair holds; on a healthy host it changes nothing. It is a guard
 *  against that fault, not what made our boxes unreachable: the VMs of Oct 2 were refused by vast's ssh proxy,
 *  and the host's own address (`Instance.directSsh`) answered with this key (box 53930876). */
export const VM_ONSTART =
  "mkdir -p /root/.ssh; chown root:root /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod g-w,o-w /root";

/** `vast lab`: a card to measure on, with no head: the same offer query and idle budget as `up`, nothing private to
 *  hold back (a head's assets are not shipped at all) */
export type LabOptions = Omit<RentOptions, "private"> & {
  /** the head whose pack the box will pull: the offers are ranked by what the session costs all in, the pack's download
   *  at each host's price per GB and its measured rate included, and the disk is sized to hold the pack */
  pack?: Head | undefined;
  /** the session's hours after the pack is pulled, priced into the ranking with --pack (1 when not given) */
  hours?: number | undefined;
  /** rent a full virtual machine rather than a container, the one kind of box that runs docker: `rig e2e` and
   *  `rig image` both drive it, and a container box's dockerd cannot create its iptables chain or mount its
   *  overlayfs (measured on box 53914526, 2026-10-02). Takes VM_IMAGE unless `image` names another. */
  vm?: boolean;
  /** the instance image over vast.toml's, for a box whose job needs another (a driver-only one for the gate) */
  image?: string | undefined;
};

/** `vast up <head> --template`: a box from the head's published template, which brings the head up on its own */
export interface TemplateRentOptions {
  /** the session's hours after the first token, priced into the pick */
  hours: number;
  maxDph?: number | undefined;
  /** the most the session may cost all in, its hours and the pack's download: an offer over it is never rented */
  budget?: number | undefined;
  /** a disk larger than the head's own, for work beside its pack (more packs, a KLD base); never smaller */
  diskGb?: number | undefined;
  dryRun?: boolean;
  idleMinutes?: number | undefined;
  /** the box's hard stop, in hours after its create, over vast.toml's max_hours */
  maxHours?: number | undefined;
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
  /** a template box: its price all in and its time to serve, as estimated when it was picked */
  estimate?: { dollars: number; minutesToServe: number; hours: number };
}

export interface LabReport {
  kind: "lab";
  instanceId: number;
  gpu: string;
  cap: string;
  dph: number;
  geo: string;
  sshHost: string;
  sshPort: number;
  /** minutes with the card idle before the reaper destroys the box, and what that costs at most */
  idleMinutes: number;
}

export interface DryRunReport {
  kind: "dry-run";
  pick: Offer;
  query: string;
  /** a template box: the first offers, the cheapest all in first */
  ranked?: OfferEstimate[];
}

export interface DownReport {
  destroyed: number[];
  hours?: number;
  cost?: number;
}

/** every box rig holds, the oldest first */
export interface StatusReport {
  boxes: BoxStatus[];
}

export interface BoxStatus {
  box: BoxState;
  /** "unread" when the market could not be read: no evidence the box is gone */
  listed: boolean | "unread";
  status?: string;
  hours: number;
  cost: number;
  tunnelActive: boolean;
  healthy: boolean;
  /** whether the cost control runs; a box listed (or unread) without it has it re-armed by status */
  idleTimer: "active" | "inactive" | "re-armed";
  /** how the timer's last check ended: an active timer whose checks fail controls nothing */
  idleCheck: "ok" | "failed" | "unread";
  /** when the box is destroyed whatever it reads, and whether its timer runs (re-armed like the idle timer's); null
   *  for a legacy box, created without one */
  hardStop: { at: number; timer: "active" | "inactive" | "re-armed" } | null;
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
  /** with a pack to price: the first offers, the cheapest all in first */
  ranked?: OfferEstimate[];
}

/** a box with its ssh endpoint known */
type ReachableBox = BoxState & { sshHost: string; sshPort: number };

type EngineSource =
  | { kind: "prebuilt" }
  | { kind: "cached"; tarball: string }
  | { kind: "compile" };

const SERVER_HEALTHY_TIMEOUT_MS = 600_000;
/** a template box's whole bring-up, from running to its first token: the pack's download at the 800 Mbit/s floor is 22
 *  minutes of it */
const TEMPLATE_BOOT_TIMEOUT_MS = 45 * 60_000;
const TEMPLATE_POLL_MS = 30_000;

export class RentGpu {
  private readonly state: RentalState;
  private readonly rates: DownloadRates;

  /** [loaded] is engine.toml as this binary read it, failed or not: down, status and idle-check
   *  never read the pin, so a checkout whose engine.toml this binary cannot read stops only up and
   *  bench, never the cost control of a box that bills (2026-09-25, 8:59-11:09 PM CT: fourteen
   *  idle checks exited 1 on `miscompilers: Invalid key` while box 52647843 billed) */
  constructor(
    private readonly deps: RentGpuDeps,
    private readonly layout: Layout,
    private readonly loaded: Result<Engine>,
  ) {
    this.state = new RentalState(deps.fs, layout, (message) => deps.log.warn(message));
    this.rates = new DownloadRates(deps.fs, this.state.downloadRatesFile);
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

    const endpoint = await this.openTunnel(config.value, box.value, head);
    if ((await endpoint.waitHealthy(SERVER_HEALTHY_TIMEOUT_MS)) !== "healthy") {
      const tail = await remote.serverLogTail(30);
      const message = `the server did not become healthy through the tunnel (box ${box.value.instanceId} left running):\n${tail}`;
      return fail(ExitCode.Failure, message);
    }
    await this.deps.fs.remove(this.state.files(box.value).idle);
    await this.armIdleTimer(box.value);

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

  /** a card rented to measure on and nothing else: rig and the engine pin shipped, no head brought up (no fetch, no
   *  derive, no server, no tunnel). What is measured goes on afterwards (an engine build, a pack) and runs with
   *  `rig engine` over ssh. The idle timer is armed at create like any box's; with no server to answer it, its check
   *  reads the card alone, and a card under GPU_BUSY_PCT for the whole budget is destroyed. */
  async lab(options: LabOptions): Promise<Result<LabReport | DryRunReport>> {
    if (!this.loaded.ok) return this.loaded;
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;

    const { pack } = options;
    const packGb = pack ? headDiskGb(pack) : 0;
    if (pack && options.diskGb !== undefined && options.diskGb < packGb)
      return fail(
        ExitCode.Usage,
        `--disk-gb ${options.diskGb} is smaller than ${pack.name}'s own ${packGb} GB: its pack would not fit`,
      );
    const diskGb = pack
      ? (options.diskGb ?? Math.max(config.value.rental.disk_gb, packGb))
      : options.diskGb;

    const ready = await this.preflight(options);
    if (!ready.ok) return ready;

    const pick = await this.pickOffer(config.value, {
      ...options,
      diskGb,
      ...(pack ? { priced: { bytes: headPackBytes(pack), hours: options.hours ?? 1 } } : {}),
    });
    if (!pick.ok) return pick;
    if (options.dryRun) {
      this.deps.log.info("dry run: no box created");
      const { offer, query, ranked } = pick.value;
      return ok({ kind: "dry-run", pick: offer, query, ...(ranked ? { ranked } : {}) });
    }

    const image = options.vm ? (options.image ?? VM_IMAGE) : options.image;
    const box = await this.createBox(config.value, null, pick.value.offer, {
      ...options,
      diskGb,
      image,
      ...(options.vm ? { onstart: VM_ONSTART, vm: true } : {}),
    });
    if (!box.ok) return box;
    const shipped = await this.shipPayload(this.remote(config.value, box.value), null, false);
    if (!shipped.ok) return shipped;

    const offer = pick.value.offer;
    const idleMinutes = idleBudget(config.value, box.value);
    const kind = options.vm ? "VM" : "box";
    this.deps.log.info(
      `READY: ${pick.value.label} ${kind} ${box.value.instanceId} at $${offer.dph.toFixed(3)}/h, ${offer.geo} — rig and the engine pin on it, no head; ssh root@${box.value.sshHost} -p ${box.value.sshPort}; idle timer armed (${idleExposure(idleMinutes, offer.dph)})`,
    );
    return ok({
      kind: "lab",
      instanceId: box.value.instanceId,
      gpu: offer.gpu,
      cap: offer.computeCap,
      dph: offer.dph,
      geo: offer.geo,
      sshHost: box.value.sshHost,
      sshPort: box.value.sshPort,
      idleMinutes,
    });
  }

  /** a box from the head's published template: the market asked with the template's filters, the offers ranked by what
   *  the session costs all in (the pack's download billed per GB included), funds checked against the pick, the box
   *  created from the template with this rig's label, and waited on while its on-start fetches, verifies and serves the
   *  head; then the tunnel and the idle timer, as for any box. A box whose supervisor gave up, or that never serves, is
   *  destroyed after its log is copied here: nothing is left billing for a person to notice. */
  async upFromTemplate(
    head: Head,
    options: TemplateRentOptions,
  ): Promise<Result<RentReport | DryRunReport>> {
    if (!this.loaded.ok) return this.loaded;
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;
    const published = await publishedImage(this.deps.fs, this.layout, head.name);
    if (!published.ok) return published;
    const { record, dir } = published.value;
    const templateFile = join(dir, "template.json");
    if (!(await this.deps.fs.exists(templateFile)))
      return fail(
        ExitCode.Failure,
        `no template for ${head.name}: run rig vast template ${head.name}`,
      );
    const template = JSON.parse(await this.deps.fs.readText(templateFile)) as {
      hashId: string;
      image: string;
    };
    if (template.image !== record.image)
      return fail(
        ExitCode.Failure,
        `the template runs ${template.image}, the last image pushed is ${record.image}: run rig vast template ${head.name}`,
      );

    const ready = await this.preflight({ dryRun: options.dryRun ?? false, binary: false });
    if (!ready.ok) return ready;

    const headGb = headDiskGb(head);
    if (options.diskGb !== undefined && options.diskGb < headGb)
      return fail(
        ExitCode.Usage,
        `--disk-gb ${options.diskGb} is smaller than ${head.name}'s own ${headGb} GB: its pack would not fit`,
      );
    const diskGb = options.diskGb ?? headGb;
    const packBytes = headPackBytes(head);
    const query = templateQuery(head, {
      cap: record.cap,
      cuda: this.engine.cuda?.version,
      diskGb,
      maxDph: options.maxDph,
    });
    const offers = await this.deps.rental.searchOffers(query, diskGb);
    await this.deps.fs.mkdirp(this.state.dir);
    await this.deps.fs.writeText(this.state.path("offers.json"), JSON.stringify(offers, null, 2));
    if (offers.length === 0) return fail(ExitCode.Failure, `no offer matches: ${query}`);
    const pulls = await this.rates.all();
    const ranked = rankOffers(offers, packBytes, options.hours, (offer) =>
      downloadRate(offer, pulls),
    );
    for (const each of ranked.slice(0, 5)) this.deps.log.info(`  ${describeEstimate(each)}`);
    const { budget } = options;
    const pick = ranked.find((each) => budget === undefined || each.dollars <= budget);
    if (!pick)
      return fail(
        ExitCode.Failure,
        `no offer within the budget of $${budget?.toFixed(2)} all in for ${options.hours} h: the cheapest is ~$${ranked[0]!.dollars.toFixed(2)} (offer ${ranked[0]!.offer.id}); nothing rented`,
      );
    const fits = this.servesOn(head, pick.offer);
    if (!fits.ok) return fits;
    if (options.dryRun) {
      this.deps.log.info("dry run: no box created");
      return ok({ kind: "dry-run", pick: pick.offer, query, ranked: ranked.slice(0, 5) });
    }
    const funds = await this.deps.rental.funds();
    if (funds < pick.dollars)
      return fail(
        ExitCode.Failure,
        `funds $${funds.toFixed(2)} cover less than the pick's ${options.hours} h all in, ~$${pick.dollars.toFixed(2)}: top up first`,
      );

    const box = await this.createBox(config.value, head, pick.offer, {
      source: { templateHash: template.hashId },
      diskGb,
      idleMinutes: options.idleMinutes,
      maxHours: options.maxHours,
    });
    if (!box.ok) return box;
    const remote = this.remote(config.value, box.value);
    const endpoint = await this.openTunnel(config.value, box.value, head);
    const served = await this.awaitTemplateBoot(remote, box.value, endpoint);
    if (!served.ok) return served;
    await this.deps.fs.remove(this.state.files(box.value).idle);
    await this.armIdleTimer(box.value);

    const serving = await endpoint.serving();
    const expected = boxServedFile(head, false);
    if (serving.model !== expected) {
      const message = `the box serves ${serving.model}, not the pinned ${expected} (box ${box.value.instanceId} left running for inspection)`;
      return fail(ExitCode.Failure, message);
    }
    const { offer } = pick;
    const label = offer.gpus > 1 ? `${offer.gpus}× ${offer.gpu}` : offer.gpu;
    this.deps.log.info(
      `READY: ${label} box ${box.value.instanceId} at $${offer.dph.toFixed(3)}/h from template ${template.hashId.slice(0, 8)} — ${serving.model}, ${serving.slots} slots, via ${endpoint.url}; idle timer armed (${idleExposure(idleBudget(config.value, box.value), offer.dph)})`,
    );
    this.deps.log.info(
      `clients: ANTHROPIC_BASE_URL=${endpoint.url} ANTHROPIC_AUTH_TOKEN=rig (Claude Code); OpenAI base ${endpoint.url}/v1`,
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
      estimate: {
        dollars: Math.round(pick.dollars * 100) / 100,
        minutesToServe: Math.round(pick.minutesToServe),
        hours: options.hours,
      },
    });
  }

  /** the box's own bring-up, followed until the head answers through the tunnel: rig's last line on the box said each
   *  time it changes; a supervisor that gave up, or a boot past its budget, has the box's log copied here and the box
   *  destroyed */
  private async awaitTemplateBoot(
    remote: RentedBox,
    box: ReachableBox,
    endpoint: HeadEndpoint,
  ): Promise<Result<void>> {
    const deadline = this.deps.clock.now() + TEMPLATE_BOOT_TIMEOUT_MS;
    let said = "";
    for (;;) {
      if (await endpoint.healthy()) return ok(undefined);
      const boot = await remote.templateBoot();
      if (boot.last && boot.last !== said) {
        said = boot.last;
        this.deps.log.info(`box ${box.instanceId}: ${boot.last}`);
      }
      const why = boot.failed
        ? "its supervisor gave up on rig up"
        : this.deps.clock.now() >= deadline
          ? `it did not serve within ${TEMPLATE_BOOT_TIMEOUT_MS / 60_000} min`
          : null;
      if (why) {
        const kept = this.state.path(`box-${box.instanceId}-up.log`);
        await remote.pullTemplateLog("up.log", kept).catch(() => {});
        const gone = await this.destroy(box);
        const fate = gone.ok ? "destroyed" : `NOT confirmed destroyed: ${gone.message}`;
        return fail(ExitCode.Failure, `box ${box.instanceId}: ${why}; its log is ${kept}; ${fate}`);
      }
      await this.deps.clock.sleep(TEMPLATE_POLL_MS);
    }
  }

  /** `box` names one box; with none, the only box rig holds, and `all` every box it holds and every box of rig's label
   *  the market lists that it does not (a box whose state was lost). Each box's units stop only once it is confirmed
   *  gone: a destroy that fails (vast 5xx or 429, the box still listed) leaves a box billing, and its idle check and
   *  hard stop must run again rather than have been disabled by the attempt. */
  async down(
    options: { all?: boolean; box?: number | undefined } = {},
  ): Promise<Result<DownReport>> {
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;

    if (options.all && options.box !== undefined)
      return fail(
        ExitCode.Usage,
        "down takes --box or --all, not both: --all destroys every box rig holds",
      );
    const held = await this.state.boxes();
    let boxes: BoxState[];
    if (options.all) boxes = held;
    else {
      const named = this.named(held, options.box);
      if (!named.ok) return named;
      boxes = named.value ? [named.value] : [];
    }
    if (held.length === 0) this.deps.log.info(`no box in ${this.state.dir}`);
    const destroyed: number[] = [];
    let hours = 0;
    let cost = 0;
    // every box is tried, whatever vast answered for the one before it
    const refused: string[] = [];
    for (const box of boxes) {
      const gone = await this.destroy(box);
      if (!gone.ok) {
        refused.push(gone.message);
        continue;
      }
      hours += gone.value.hours;
      cost += gone.value.cost;
      destroyed.push(box.instanceId);
    }
    const billed = destroyed.length > 0 ? { hours, cost: Math.round(cost * 100) / 100 } : {};

    if (options.all) {
      const known = held.map((box) => box.instanceId);
      for (const instance of await this.forgottenBoxes(config.value, known)) {
        await this.deps.rental.destroy(instance.id);
        destroyed.push(instance.id);
        this.deps.log.info(`destroyed forgotten ${instance.label} box ${instance.id}`);
      }
    }
    if (refused.length > 0) return fail(ExitCode.Failure, refused.join("\n"));
    return ok({ destroyed, ...billed });
  }

  /** every box rig holds, or the one `box` names */
  async status(options: { box?: number | undefined } = {}): Promise<Result<StatusReport>> {
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;
    const held = await this.state.boxes();
    let boxes = held;
    if (options.box !== undefined) {
      const named = this.named(held, options.box);
      if (!named.ok) return named;
      boxes = named.value ? [named.value] : [];
    }
    const statuses: BoxStatus[] = [];
    for (const box of boxes) statuses.push(await this.boxStatus(config.value, box));
    return ok({ boxes: statuses });
  }

  private async boxStatus(config: VastConfig, box: BoxState): Promise<BoxStatus> {
    const files = this.state.files(box);
    const tunnelActive = await this.deps.systemd.isActive(files.tunnelUnit);
    const endpoint = this.endpoint(config, box);
    const healthy = endpoint ? await endpoint.healthy() : false;
    const listing = await this.listing(box);
    const hours = billedHours(box, this.deps.clock.now());
    // A box billing with its cost control dead is the one state status must not only report:
    // 2026-09-24 the timer went inactive at 09:08 with no stop in the journal, and box 52390478
    // billed idle until a person noticed (8.3 h, ~$11.63). A market that cannot be read is no
    // evidence the box is gone, so the timer is re-armed then too. The hard stop's timer the same.
    const billing = listing === "unread" ? "may be billing" : "is billing";
    const timerActive = await this.deps.systemd.isActive(files.idleTimer);
    let idleTimer: BoxStatus["idleTimer"] = timerActive ? "active" : "inactive";
    if (listing !== null && !timerActive) {
      await this.armIdleTimer(box);
      this.deps.log.warn(
        `box ${box.instanceId} ${billing} and ${files.idleTimer} was not running: re-armed it (idle budget ${idleBudget(config, box)} min)`,
      );
      idleTimer = "re-armed";
    }
    let hardStop: BoxStatus["hardStop"] = null;
    if (files.stopTimer && box.stopAt !== undefined) {
      const stopActive = await this.deps.systemd.isActive(files.stopTimer);
      hardStop = { at: box.stopAt, timer: stopActive ? "active" : "inactive" };
      if (listing !== null && !stopActive) {
        await this.armStopTimer(box);
        this.deps.log.warn(
          `box ${box.instanceId} ${billing} and ${files.stopTimer} was not running: re-armed it`,
        );
        hardStop = { at: box.stopAt, timer: "re-armed" };
      }
    }
    // An active timer whose checks fail is as dead as a stopped one (2026-09-25: fourteen checks
    // exited 1 while the timer read active), so the last check's result is part of the answer.
    const lastCheck = await this.deps.systemd.lastResult(files.idleService);
    const idleCheck: BoxStatus["idleCheck"] =
      lastCheck === null ? "unread" : lastCheck === "success" ? "ok" : "failed";
    if (idleCheck === "failed") {
      this.deps.log.warn(
        `${files.idleService}'s last run ended ${lastCheck}: box ${box.instanceId}'s cost control is not running; see journalctl --user -u ${files.idleService}`,
      );
    }
    return {
      box,
      listed: listing === "unread" ? "unread" : listing !== null,
      ...(listing !== null && listing !== "unread" ? { status: listing.status } : {}),
      hours,
      cost: billedCost(hours, box.dph),
      tunnelActive,
      healthy,
      idleTimer,
      idleCheck,
      hardStop,
    };
  }

  /** the box `id` names among those held, or the only one held when it names none (null when none is held); several
   *  held and none named is a usage error: a command that acts on a box never guesses which */
  private named(held: BoxState[], id: number | undefined): Result<BoxState | null> {
    if (id !== undefined) {
      const box = held.find((each) => each.instanceId === id);
      return box ? ok(box) : fail(ExitCode.Failure, `no box ${id} held here (${this.state.dir})`);
    }
    if (held.length <= 1) return ok(held[0] ?? null);
    return fail(
      ExitCode.Usage,
      `${held.length} boxes held: name one with --box (${held.map((box) => box.instanceId).join(", ")})`,
    );
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
  async idleCheck(options: { box?: number | undefined } = {}): Promise<Result<IdleReport>> {
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;
    const held = await this.state.boxes();
    // a box's own timer names it; one destroyed since is no box, not a failure of the unit that outlived it
    const named =
      options.box !== undefined
        ? ok(held.find((each) => each.instanceId === options.box) ?? null)
        : this.named(held, undefined);
    if (!named.ok) return named;
    const box = named.value;
    if (!box) return ok({ action: "no-box" });

    const now = this.deps.clock.now();
    // a box that serves nothing has no tunnel, and must never read another box's server through its port
    const endpoint = this.endpoint(config.value, box);
    const activity = (await endpoint?.activity()) ?? { key: "unreachable", busy: 0 };
    const listing = await this.listing(box);
    let gpuUtil = listing === "unread" ? undefined : listing?.gpuUtil;
    // The card is read on the box itself, at the address rig recorded for it, for a box vast lists as running: vast
    // lists no sample at all for a VM, and its sample for a container is one instant, as the box's own is. The box keeps
    // a sampler, so the read carries the peak since the last read and work that ran between two reads is seen. A box that
    // does not answer, with nothing from vast to go on, is idle by the same clock (else a VM that lost its network bills
    // for ever); one that answers without a number stays a card nobody read, below.
    let readOnBox: "box" | "unreachable" | undefined;
    let seenOnBox:
      | {
          now: number;
          window: number | null;
          downloadKBps: number | null;
          pull: { kibPerSecond: number; seconds: number } | null;
        }
      | undefined;
    if (listing && listing !== "unread" && listing.status === "running") {
      if (box.sshHost && box.sshPort) {
        const seen = await this.remote(config.value, box).cardUtilization(
          20_000,
          CARD_WINDOW_SECONDS,
          DOWNLOAD_BUSY_KBPS,
        );
        if (seen === "unreachable") {
          if (gpuUtil === undefined) {
            gpuUtil = 0;
            readOnBox = "unreachable";
          }
        } else if (seen !== null) {
          seenOnBox = seen;
          gpuUtil = Math.max(gpuUtil ?? 0, seen.now, seen.window ?? 0);
          readOnBox = "box";
        }
      }
    }
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
    const downloadKBps = seenOnBox?.downloadKBps ?? null;
    const downloading = downloadKBps !== null && downloadKBps >= DOWNLOAD_BUSY_KBPS;
    if (seenOnBox?.pull) await this.recordPull(box, seenOnBox.pull, now);
    const last = await this.state.idle(box);
    const gpu =
      readOnBox === "unreachable"
        ? "box unreachable over ssh"
        : gpuUtil === undefined
          ? "no GPU reading"
          : seenOnBox && seenOnBox.window !== null && seenOnBox.window > seenOnBox.now
            ? `GPU ${seenOnBox.now} % now, peak ${seenOnBox.window} % in the last ${CARD_WINDOW_SECONDS / 60} min, read on the box`
            : `GPU ${gpuUtil} %${readOnBox === "box" ? " read on the box" : ""}`;
    const card = downloadKBps === null ? gpu : `${gpu}, download ${rate(downloadKBps)}`;
    // Every check says what it saw. The reaper's decision is a run of readings, and a destroy that cannot be traced to
    // them cannot be told from a reaper that read wrong (2026-10-02: "idle for 51 min", six checks before it silent).
    const said = (verdict: string) =>
      this.deps.log.info(
        `idle-check box ${box.instanceId}: server ${activity.key}, ${card}: ${verdict}`,
      );
    if (activity.busy > 0 || gpuBusy || downloading || activity.key !== last?.key) {
      await this.state.saveIdle(box, { key: activity.key, ts: now });
      const verdict = activity.busy > 0 || gpuBusy || downloading ? "active" : "changed";
      said(verdict);
      return ok({ action: verdict });
    }
    // an idle server beside an unread card counts nothing either: a busy card is never idle,
    // whatever the server says
    if (cardUnread) {
      const message = `the server is idle but ${cardUnread}: box ${box.instanceId} not counted, its idle clock unchanged`;
      return fail(ExitCode.Failure, message);
    }

    const idleMinutes = Math.floor((now - last.ts) / 60_000);
    const budget = idleBudget(config.value, box);
    if (idleMinutes < budget) {
      said(`idle ${idleMinutes} of ${budget} min`);
      return ok({ action: "idle", idleMinutes });
    }
    this.deps.log.info(
      `idle for ${idleMinutes} min (${activity.key}, ${card}) — destroying the box`,
    );
    const down = await this.down({ box: box.instanceId });
    if (!down.ok) return down;
    return ok({ action: "destroyed", idleMinutes });
  }

  /** the head's gates on the box (its one card, the server stopped meanwhile), the run pulled
   *  back, then the live probes through the tunnel */
  async bench(
    head: Head,
    options: { box?: number | undefined } = {},
  ): Promise<Result<BenchReport>> {
    if (!this.loaded.ok) return this.loaded;
    const config = await loadVastConfig(this.deps.fs, this.layout);
    if (!config.ok) return config;
    const held = await this.state.boxes();
    // with several boxes and none named, the one serving this head
    const serving = held.filter((each) => each.head === head.name);
    const named = this.named(
      options.box === undefined && serving.length === 1 ? serving : held,
      options.box,
    );
    if (!named.ok) return named;
    const box = named.value;
    if (!box) return fail(ExitCode.Failure, "no box; run: rig vast up");
    const endpoint = this.endpoint(config.value, box);
    if (!endpoint) return fail(ExitCode.Failure, `box ${box.instanceId} serves no head`);
    const remote = this.remote(config.value, box);

    const pulledRuns = join(this.state.pulledRunsDir, pulledRunName(box));
    await this.deps.fs.mkdirp(pulledRuns);
    await remote.stopServer();
    const gate = await remote.rig(`gate ${head.name} --gpu 0 --json`);
    const remoteRunDir = gateRunDir(gate.stdout);
    if (remoteRunDir) await this.pullGateRun(remote, remoteRunDir, pulledRuns);
    await remote.startServer(head.name);

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
  private async preflight(options: {
    dryRun?: boolean | undefined;
    /** the box runs a compiled rig shipped from here; a template box runs its image's */
    binary?: boolean;
  }): Promise<Result<void>> {
    // a box rented by a rig that held one keeps that rig's units, which act on the only box: it stays alone until down
    const legacy = await this.state.legacyBox();
    if (!options.dryRun && legacy) {
      const message = `box ${legacy.instanceId} was rented by a rig that held one box (${this.state.legacyInstanceFile}), and none is rented beside it; run: rig vast down --box ${legacy.instanceId}`;
      return fail(ExitCode.Failure, message);
    }
    const rigBinary = join(this.layout.root, "dist", "rig");
    if (options.binary !== false && !options.dryRun && !(await this.deps.fs.exists(rigBinary))) {
      const message = `${rigBinary} is missing — the box runs the compiled rig (run: bun run build)`;
      return fail(ExitCode.Failure, message);
    }
    if (options.binary !== false && !options.dryRun) {
      const stale = await this.notBuiltFromHead(rigBinary);
      if (stale) return fail(ExitCode.Failure, stale);
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

  /** why dist/rig is not the rig to ship, or null when it is: a box runs the rig shipped to it, and one built before a
   *  change refuses that change's flags there, on cards paid by the hour (rental 3: `engine ab --median` exited 64 on
   *  the box, its dist/rig older than #168). In a checkout it must be HEAD's tree. An install (install.sh) is a release
   *  unpacked with no .git: its dist/rig is the release's, built by release.yml from the tag's tree, so a clean stamp
   *  is enough there. The checkout is rig's own .git, not a HEAD git finds above the root (a home kept in git). */
  private async notBuiltFromHead(rigBinary: string): Promise<string | null> {
    const checkout = await this.deps.fs.exists(join(this.layout.root, ".git"));
    const remedy = checkout
      ? "rebuild it from this checkout: bun run build"
      : "install a release (install.sh), or ship from a checkout after bun run build";
    const said = await this.deps.shell.run([rigBinary, BUILT_FROM.flag], { timeoutMs: 30_000 });
    const stamp = STAMP.exec(said.code === 0 ? said.stdout.trim() : "");
    if (!stamp)
      return `${rigBinary} does not say what it was built from (compiled before rig stamped its build, or not by bun run build); ${remedy}`;
    const [built, tree, dirty] = stamp;
    if (!checkout)
      return dirty ? `${rigBinary} was built with changes no commit has; ${remedy}` : null;
    const head = await this.deps.git.revParse(this.layout.root, BUILT_FROM.ref);
    if (!head)
      return `${this.layout.root} is a git checkout with no HEAD to check ${rigBinary} against`;
    if (built === head) return null;
    if (tree === head)
      return `${rigBinary} was built with changes HEAD does not hold, so the box would run code no commit has; commit them, then rebuild: bun run build`;
    return `${rigBinary} was built from tree ${tree!.slice(0, 12)}, and HEAD's tree is ${head.slice(0, 12)}; ${remedy}`;
  }

  /** the profile serve would give the offered cards and the cache formats it would run there, refused before the
   *  rental: on the box they are refused only by serve, after the fetch, the build and the derive, on cards paid by
   *  the hour */
  private servesOn(head: Head, offer: Offer): Result<void> {
    const cards = Array.from({ length: offer.gpus }, (_, index) => ({
      index,
      cap: offer.computeCap,
      vramMiB: offer.gpuRamMiB,
    }));
    const placed = pickProfile(head, cards);
    if (!placed.ok)
      return fail(placed.code, `REFUSING to rent ${offer.gpu}: ${head.name} ${placed.message}`);
    const { profile } = placed.value;
    const draft = profileSpeculates(head, profile) ? head.speculative?.cache : undefined;
    const refusal = cacheRefusal(this.engine, profileCache(head, profile), draft);
    if (refusal)
      return fail(
        ExitCode.Unsupported,
        `REFUSING to rent ${offer.gpu}: on its profile (${describeProfile(profile)}) ${refusal}`,
      );
    return ok(undefined);
  }

  /** the market queried, the offers kept beside the box's state, the cheapest one picked: by the hour, or all in when a
   *  pack is priced. The download floor and the pack's pull read each host's measured rate where rig has one
   *  (download-rates.ts), its declaration where not. A card this engine is not measured on needs --allow-arch. */
  private async pickOffer(
    config: VastConfig,
    options: RentOptions & { vm?: boolean; priced?: { bytes: number; hours: number } },
  ): Promise<Result<Pick>> {
    const diskGb = options.diskGb ?? config.rental.disk_gb;
    const shape = {
      maxDph: options.maxDph,
      geo: options.geo,
      diskGb,
      gpus: options.gpus,
      vm: options.vm,
    };
    const floor = options.minDownMbps;
    const query = offerQuery(config, options.gpu, { ...shape, minDownMbps: floor });
    // the floor goes in the query, not only over the rows read: vast returns at most 64 offers, cheapest first, and a
    // fast one dearer than the 64th would never be seen
    const found = await this.deps.rental.searchOffers(query, diskGb);
    const pulls = await this.rates.all();
    const rateOf = (offer: Offer) => downloadRate(offer, pulls);
    const offers =
      floor === undefined
        ? found
        : found.filter((each) => {
            const rate = rateOf(each);
            if (rate.mbps >= floor) return true;
            // the query let it through on what it declares: say why it is skipped
            this.deps.log.info(
              `  offer ${each.id} at ${describeRate(rate, each)} (declares ${each.downMbps.toFixed(0)}): under the floor`,
            );
            return false;
          });
    await this.deps.fs.mkdirp(this.state.dir);
    if (offers.length === 0 && floor !== undefined) {
      // nothing at the floor: read the market without it, to say what the fastest is and what it costs
      const market = await this.deps.rental.searchOffers(
        offerQuery(config, options.gpu, shape),
        diskGb,
      );
      await this.deps.fs.writeText(this.state.path("offers.json"), JSON.stringify(market, null, 2));
      if (market.length === 0) return fail(ExitCode.Failure, `no offer matches: ${query}`);
      const fastest = market.reduce((best, each) =>
        rateOf(each).mbps > rateOf(best).mbps ? each : best,
      );
      return fail(
        ExitCode.Failure,
        `no offer downloads at ${floor} Mb/s or more: the fastest is offer ${fastest.id} at ${describeRate(rateOf(fastest), fastest)}, $${fastest.dph.toFixed(3)}/h (${fastest.geo}); nothing rented`,
      );
    }
    await this.deps.fs.writeText(this.state.path("offers.json"), JSON.stringify(offers, null, 2));
    if (offers.length === 0) return fail(ExitCode.Failure, `no offer matches: ${query}`);
    const { priced } = options;
    const ranked = priced ? rankOffers(offers, priced.bytes, priced.hours, rateOf) : undefined;
    if (ranked)
      for (const each of ranked.slice(0, 5)) this.deps.log.info(`  ${describeEstimate(each)}`);
    else for (const offer of offers.slice(0, 5)) this.deps.log.info(`  ${describeOffer(offer)}`);

    const offer = ranked ? ranked[0]!.offer : offers[0]!;
    if (!this.engine.supports(offer.computeCap) && !options.allowArch) {
      const measured = this.engine.archs.map((arch) => `sm_${arch.cap}`).join(", ");
      const message = `${offer.gpu} is sm_${offer.computeCap}, not a card this engine is measured on (${measured}); --allow-arch ${offer.computeCap} rents it for a benchmark`;
      return fail(ExitCode.Unsupported, message);
    }
    const label = offer.gpus > 1 ? `${offer.gpus}× ${offer.gpu}` : offer.gpu;
    this.deps.log.info(
      `pick: offer ${offer.id} ${label} at $${offer.dph.toFixed(3)}/h, ${offer.geo}, sm_${offer.computeCap}`,
    );
    return ok({ offer, query, label, ...(ranked ? { ranked: ranked.slice(0, 5) } : {}) });
  }

  /** a window's pull kept for the next rental's ranking. Keeping it is never the cost control's failure: a write that
   *  fails is said, and the check goes on to its verdict. */
  private async recordPull(
    box: BoxState,
    pull: { kibPerSecond: number; seconds: number },
    now: number,
  ): Promise<void> {
    try {
      const mbps = await this.rates.record(box, pull, now);
      if (mbps !== null)
        this.deps.log.info(
          `box ${box.instanceId} measured ${mbps} Mb/s pulling (${box.geo}${box.machineId ? `, machine ${box.machineId}` : ""}): the next rental is ranked on it`,
        );
    } catch (error) {
      this.deps.log.warn(`box ${box.instanceId}'s pull was not kept: ${(error as Error).message}`);
    }
  }

  /** the instance created and recorded, then waited for: running, with an ssh endpoint that
   *  answers; a box that never gets there is destroyed, not left billing */
  private async createBox(
    config: VastConfig,
    /** the head the box is for; null for a lab box, which serves none */
    head: Head | null,
    offer: Offer,
    options: {
      /** vast.toml's stock image unless a template is named, or `image` names another */
      source?: { templateHash: string };
      image?: string | undefined;
      /** run on the box once vast has written its key */
      onstart?: string | undefined;
      /** a VM, which boots slower than a container and is waited for longer */
      vm?: boolean | undefined;
      diskGb?: number | undefined;
      idleMinutes?: number | undefined;
      maxHours?: number | undefined;
    },
  ): Promise<Result<ReachableBox>> {
    const maxHours = options.maxHours ?? config.rental.max_hours;
    if (!(maxHours >= MAX_HOURS.floor && maxHours <= MAX_HOURS.ceiling))
      return fail(
        ExitCode.Usage,
        `a hard stop takes ${MAX_HOURS.floor} to ${MAX_HOURS.ceiling} hours, not ${maxHours}`,
      );
    // each serving box its own end of a tunnel here: the lowest port from vast.toml's local_port that no held box has
    // and this rental claims, so that two rentals at once, which both read no box on it, never share it
    const held = await this.state.boxes();
    let localPort: number | undefined;
    if (head) {
      const taken = new Set(held.map((each) => this.localPort(config, each)));
      localPort = config.rental.local_port;
      while (
        taken.has(localPort) ||
        !(await this.deps.fs.claim(this.state.portClaim(localPort), `${this.deps.clock.now()}\n`))
      )
        localPort++;
    }
    let instanceId: number;
    try {
      instanceId = await this.deps.rental.create(offer.id, {
        ...(options.source ?? {
          image: options.image ?? config.rental.image,
          ...(options.onstart ? { onstart: options.onstart } : {}),
        }),
        diskGb: options.diskGb ?? config.rental.disk_gb,
        label: config.rental.label,
      });
    } catch (error) {
      if (localPort !== undefined) await this.deps.fs.remove(this.state.portClaim(localPort));
      throw error;
    }
    const createdAt = this.deps.clock.now();
    const box: BoxState = {
      instanceId,
      offerId: offer.id,
      gpu: offer.gpu,
      gpus: offer.gpus,
      cap: offer.computeCap,
      dph: offer.dph,
      geo: offer.geo,
      ...(offer.machineId ? { machineId: offer.machineId } : {}),
      createdAt,
      ...(head ? { head: head.name } : {}),
      ...(options.idleMinutes !== undefined ? { idleMinutes: options.idleMinutes } : {}),
      ...(localPort !== undefined ? { localPort } : {}),
      stopAt: createdAt + maxHours * 3_600_000,
    };
    await this.state.saveBox(box);
    // the box bills from here, so the reaper and the hard stop are installed and armed here and not after
    // provisioning: a box whose provisioning dies, or whose `up` is killed, otherwise bills with nothing watching
    // it. Only tunnel.env needs the box's endpoint, so the units themselves can be written now.
    await this.installUnits(
      config,
      box,
      head?.port ?? null,
      options.idleMinutes ?? config.rental.idle_minutes,
    );
    await this.armIdleTimer(box);
    await this.armStopTimer(box);
    this.deps.log.info(
      `instance ${instanceId} created; waiting for it to run (image pull + vast's sshd install); hard stop at ${new Date(box.stopAt!).toISOString()} (${maxHours} h)`,
    );

    const running = await this.waitRunning(instanceId);
    if (!running) {
      const outcome = await this.abandon(box, "never reached running");
      return fail(ExitCode.Failure, `instance ${instanceId} never reached running; ${outcome}`);
    }
    if (!running.sshHost || !running.sshPort) {
      const outcome = await this.abandon(box, "no ssh endpoint");
      return fail(ExitCode.Failure, `instance ${instanceId} has no ssh endpoint; ${outcome}`);
    }
    // a VM is reached at the host's own address when vast gives one: its proxy port refused ssh for 13 minutes
    // after `running` on a VM whose direct port answered at the first try (box 53930876). A container keeps the
    // proxy, which is what every container box has been reached by.
    const direct = options.vm ? running.directSsh : undefined;
    const reachable: ReachableBox = {
      ...box,
      sshHost: direct?.host ?? running.sshHost,
      sshPort: direct?.port ?? running.sshPort,
    };
    await this.state.saveBox(reachable);
    await this.deps.fs.remove(this.state.files(reachable).knownHosts);

    const said = await this.waitSsh(
      this.remote(config, reachable),
      options.vm ? VM_SSH_ATTEMPTS : SSH_ATTEMPTS,
    );
    if (said !== null) {
      const outcome = await this.abandon(reachable, "ssh never answered");
      const message = `ssh never answered at ${reachable.sshHost}:${reachable.sshPort} (last ssh said: ${said}); ${outcome}`;
      return fail(ExitCode.Failure, message);
    }
    this.deps.log.info(`ssh up: root@${reachable.sshHost}:${reachable.sshPort}`);
    return ok(reachable);
  }

  /** rig, the head and the engine pin, as one tarball; the head's private [derive] assets only
   *  with --private */
  private async shipPayload(
    remote: RentedBox,
    /** null for a lab box: rig and the pin only, nothing of any head */
    head: Head | null,
    shipPrivate: boolean,
  ): Promise<Result<void>> {
    const payload = this.state.path("payload.tar.gz");
    const held = head === null || shipPrivate ? [] : privateAssets(head);
    if (head !== null && held.length > 0) {
      this.deps.log.info(
        `kept here: ${held.join(", ")} (private); the box serves ${boxServedFile(head, false)} — --private ships them`,
      );
    }
    const tar = await this.deps.shell.run(
      [
        "tar",
        "-C",
        this.layout.root,
        ...(head ? held.map((path) => `--exclude=heads/${head.name}/${path}`) : []),
        "-czf",
        payload,
        "dist/rig",
        ...(head ? [`heads/${head.name}`] : []),
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

    // the head named to prepare and build, so a head with its own engine pin is built on it
    if (!(await step("prepare", `prepare ${head.name} --gpu 0${allowArch}`))) {
      return fail(ExitCode.Failure, `prepare failed on box ${box.instanceId}`);
    }
    const space = await remote.packSpace(head.name);
    const toFetch = Math.max(0, headPackBytes(head) - space.present);
    if (space.free > 0 && space.free < toFetch + PACK_SLACK_BYTES) {
      return fail(
        ExitCode.Failure,
        `box ${box.instanceId} has ${gbOf(space.free)} GB free, and ${head.name} still needs ` +
          `${gbOf(toFetch + PACK_SLACK_BYTES)} GB (${gbOf(toFetch)} GB of pack left to fetch plus ` +
          `${gbOf(PACK_SLACK_BYTES)} GB slack): the box was created for ${headDiskGb(head)} GB and ` +
          `has less`,
      );
    }
    const fetching = remote.rig(`fetch ${head.name}`);
    const built =
      engine.kind === "prebuilt"
        ? await step("build", `build ${head.name} --gpu 0${allowArch}`)
        : engine.kind === "cached"
          ? await step(
              "build",
              `build ${head.name} --gpu 0 --from-tarball ${remote.buildTarball(engine.tarball)}${allowArch}`,
            )
          : await this.buildOnBox(remote, head, offer, step, allowArch);
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
    head: Head,
    offer: Offer,
    step: (label: string, args: string) => Promise<boolean>,
    allowArch: string,
  ): Promise<boolean> {
    const started = this.deps.clock.now();
    this.deps.log.info(
      `no cached build for sm_${offer.computeCap} — building on the box (5–20 min)`,
    );
    const built = await step("build", `build ${head.name} --gpu 0 --portable${allowArch}`);
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

  /** the box's tunnel unit to its ssh endpoint, installed and started */
  private async openTunnel(
    config: VastConfig,
    box: ReachableBox,
    head: Head,
  ): Promise<HeadEndpoint> {
    const files = this.state.files(box);
    await this.deps.fs.writeText(files.tunnelEnv, `HOST=${box.sshHost}\nPORT=${box.sshPort}\n`);
    await this.installUnits(config, box, head.port, idleBudget(config, box));
    await this.deps.systemd.restart(files.tunnelUnit);
    // a serving box always has a port here (createBox)
    return this.endpoint(config, box)!;
  }

  /** the box's idle reaper and hard stop, and its tunnel unless it serves nothing (`remotePort` null) */
  private async installUnits(
    config: VastConfig,
    box: BoxState,
    remotePort: number | null,
    idleMinutes: number,
  ): Promise<void> {
    const dir = this.deps.systemd.unitDir();
    await this.deps.fs.mkdirp(dir);
    const files = this.state.files(box);
    const localPort = this.localPort(config, box);
    const { instanceId } = box;
    const units: Record<string, string> = {
      ...(remotePort === null || localPort === null
        ? {}
        : {
            [files.tunnelUnit]: renderTunnelUnit({
              tunnelEnv: files.tunnelEnv,
              knownHosts: files.knownHosts,
              localPort,
              remotePort,
            }),
          }),
      [files.idleService]: renderIdleService({ self: this.deps.self, idleMinutes, instanceId }),
      [files.idleTimer]: renderIdleTimer(instanceId),
      ...(files.stopService && files.stopTimer && box.stopAt !== undefined
        ? {
            [files.stopService]: renderStopService({
              self: this.deps.self,
              vastai: this.deps.vastai,
              instanceId,
            }),
            [files.stopTimer]: renderStopTimer({ instanceId, at: box.stopAt }),
          }
        : {}),
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

  /** The box's units, once it is gone: its timers disabled and stopped, its tunnel stopped, and their files removed.
   *  Never a service: the box goes down from inside its own idle check or hard stop, and stopping that service would
   *  kill the process doing it. */
  private async retireUnits(box: BoxState): Promise<void> {
    const files = this.state.files(box);
    const timers = [files.idleTimer, files.stopTimer].filter((unit) => unit !== null);
    for (const timer of timers) {
      try {
        await this.deps.systemd.disable(timer);
      } catch {
        /* not installed yet */
      }
    }
    for (const unit of [...timers, files.tunnelUnit]) {
      try {
        await this.deps.systemd.stop(unit);
      } catch {
        /* not installed yet */
      }
    }
    const dir = this.deps.systemd.unitDir();
    const units = [files.idleService, files.stopService, files.tunnelUnit, ...timers];
    for (const unit of units) if (unit !== null) await this.deps.fs.remove(join(dir, unit));
    await this.deps.systemd.daemonReload();
  }

  /** the box's idle timer enabled as well as started, so a restart of the user manager or of this machine arms it
   *  again while the box bills; `down` disables it */
  private async armIdleTimer(box: BoxState): Promise<void> {
    const { idleTimer } = this.state.files(box);
    await this.deps.systemd.enable(idleTimer);
    await this.deps.systemd.restart(idleTimer);
  }

  /** the box's hard stop, enabled and started the same way; a legacy box has none */
  private async armStopTimer(box: BoxState): Promise<void> {
    const { stopTimer } = this.state.files(box);
    if (!stopTimer || box.stopAt === undefined) return;
    await this.deps.systemd.enable(stopTimer);
    await this.deps.systemd.restart(stopTimer);
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
    await this.state.clear(box);
    await this.retireUnits(box);
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

  /** this machine's end of the box's tunnel: its own port, or vast.toml's for a legacy box that serves; null for a box
   *  that serves nothing */
  private localPort(config: VastConfig, box: BoxState): number | null {
    return box.localPort ?? (box.head ? config.rental.local_port : null);
  }

  /** the box's server through its tunnel; null for a box that serves nothing */
  private endpoint(config: VastConfig, box: BoxState): HeadEndpoint | null {
    const port = this.localPort(config, box);
    if (port === null) return null;
    return new HeadEndpoint(this.deps.http, this.deps.clock, `http://127.0.0.1:${port}`);
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

  /** null once ssh answers, else what it last said after the whole wait */
  private async waitSsh(remote: RentedBox, attempts: number): Promise<string | null> {
    let said: string | null = "never tried";
    for (let attempt = 0; attempt < attempts; attempt++) {
      said = await remote.unreachable(20_000);
      if (said === null) return null;
      await this.deps.clock.sleep(5000);
    }
    return said;
  }

  /** a box that will never serve, destroyed as `down` destroys one: its state and units go only once vast no longer
   *  lists it, since vastai answers 0 to a destroy vast refused. What to say of it */
  private async abandon(box: BoxState, why: string): Promise<string> {
    this.deps.log.error(`instance ${box.instanceId} ${why}; destroying it`);
    const gone = await this.destroy(box);
    return gone.ok ? `destroyed ${box.instanceId}` : gone.message;
  }
}

/** the idle minutes that destroy this box: its own budget from `up`, else vast.toml's */
/** a received rate as the log shows it: 30.3 MB/s, 200 KB/s */
function rate(kbps: number): string {
  return kbps >= 1024 ? `${(kbps / 1024).toFixed(1)} MB/s` : `${kbps} KB/s`;
}

function idleBudget(config: VastConfig, box: BoxState): number {
  return box.idleMinutes ?? config.rental.idle_minutes;
}

/** the budget and what it lets a box nobody uses cost before it goes (box 52390478 was rented with
 *  2,880 minutes for a training run and billed idle for hours after its session ended, 2026-09-24) */
function idleExposure(minutes: number, dph: number): string {
  return `${minutes} min: up to ~$${billedCost(minutes / 60, dph).toFixed(2)} idle before it is destroyed`;
}

/** one ranked offer as the log shows it: its price all in first */
function describeEstimate(each: OfferEstimate): string {
  const { offer } = each;
  return `~$${each.dollars.toFixed(2)} all in, serving in ~${Math.round(each.minutesToServe)} min: ${describeOffer(offer)}, $${offer.downCostPerGb.toFixed(3)}/GB down, pulls at ${describeRate(each.rate, offer)}`;
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
const gbOf = (bytes: number) => (bytes / 1e9).toFixed(1);

/** the pack the box serves: this machine's with its private assets, else what a machine without
 *  them serves (head.ts): the [public] pack when head.toml declares one, else the source pack */
function boxServedFile(head: Head, shipPrivate: boolean): string {
  if (shipPrivate || privateAssets(head).length === 0) return basename(head.servedPath);
  return basename(head.declaredPublic?.path ?? head.sourcePath);
}
