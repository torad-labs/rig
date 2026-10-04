// What rig keeps here about the boxes it rents: one directory a box under local/rented-box/boxes/<instance>/ (its
// record, its ssh known_hosts, a fresh host key on a fresh host:port every time, its idle clock between checks and its
// tunnel's endpoint) and, beside them, what the boxes share: the market's last offers, the payload, the measured
// downloads, the gate runs pulled back. Each box has its own units (units.ts): reaper, hard stop, tunnel.
//
// A rig that held one box kept it at local/rented-box/instance.json, its units named for no box. Such a box is read
// where it is ("legacy") with that rig's files and units until it is destroyed, so upgrading rig never leaves a billing
// box without its reaper; no box is rented beside it (gpu-rental.service.ts preflight).
import { join } from "node:path";
import type { FileSystem, Layout, SshTarget } from "@rig/core";
import { type BoxUnits, boxUnits, LEGACY_UNITS } from "./units.ts";

export interface BoxState {
  instanceId: number;
  offerId: number;
  gpu: string;
  gpus?: number;
  cap: string;
  dph: number;
  geo: string;
  /** the host's machine, which a measured download rate is kept by; absent on a box created before rig kept it */
  machineId?: number;
  createdAt: number;
  sshHost?: string;
  sshPort?: number;
  head?: string;
  /** `up --idle-minutes`: this box's idle budget, read by every idle check */
  idleMinutes?: number;
  /** this machine's end of the box's tunnel: each serving box has its own. Absent on a box that serves nothing, and
   *  on a legacy one, which took vast.toml's local_port */
  localPort?: number;
  /** when the box's hard stop destroys it, epoch ms; absent on a legacy box, which was created without one */
  stopAt?: number;
  /** read from local/rented-box/instance.json, where a rig that held one box kept it; never written */
  legacy?: boolean;
}
export interface IdleState {
  key: string;
  ts: number;
}

/** where one box's state lives here, and its units' names */
export interface BoxFiles extends BoxUnits {
  dir: string;
  instance: string;
  tunnelEnv: string;
  knownHosts: string;
  idle: string;
}

export class RentalState {
  readonly dir: string;
  constructor(
    private readonly fs: FileSystem,
    private readonly layout: Layout,
    /** told of a box record that does not parse, which is skipped so that every other box is still read */
    private readonly warn: (message: string) => void = () => {},
  ) {
    this.dir = layout.rentedBoxDir;
  }
  /** what the boxes share */
  path(name: string) {
    return join(this.dir, name);
  }
  get boxesDir() {
    return this.path("boxes");
  }
  get legacyInstanceFile() {
    return this.path("instance.json");
  }
  /** every box's measured pull (download-rates.ts), kept past the box */
  get downloadRatesFile() {
    return this.path("download-rates.json");
  }
  get cachedBuildsDir() {
    return this.layout.cachedBuildsDir;
  }
  get pulledRunsDir() {
    return this.path("pulled-runs");
  }
  /** a serving box's tunnel port, claimed (FileSystem.claim) before the box is created so that two rentals at once
   *  never take the same one, and released when its box is cleared. A claim whose rental died before the box was
   *  saved is left: that port is skipped, never shared */
  portClaim(port: number) {
    return join(this.path("ports"), String(port));
  }

  files(box: Pick<BoxState, "instanceId" | "legacy">): BoxFiles {
    const dir = box.legacy ? this.dir : join(this.boxesDir, String(box.instanceId));
    return {
      dir,
      instance: join(dir, "instance.json"),
      tunnelEnv: join(dir, "tunnel.env"),
      knownHosts: join(dir, "known_hosts"),
      idle: join(dir, "idle.json"),
      ...(box.legacy ? LEGACY_UNITS : boxUnits(box.instanceId)),
    };
  }

  /** every box rig holds, the oldest first: the legacy one, if any, and each under boxes/ */
  async boxes(): Promise<BoxState[]> {
    const found: BoxState[] = [];
    const legacy = await this.record(this.legacyInstanceFile);
    if (legacy) found.push({ ...legacy, legacy: true });
    if (await this.fs.exists(this.boxesDir)) {
      for (const name of await this.fs.list(this.boxesDir)) {
        if (!/^\d+$/.test(name)) continue;
        const box = await this.record(join(this.boxesDir, name, "instance.json"));
        if (box) found.push(box);
      }
    }
    return found.sort((a, b) => a.createdAt - b.createdAt);
  }
  async legacyBox(): Promise<BoxState | null> {
    const legacy = await this.record(this.legacyInstanceFile);
    return legacy ? { ...legacy, legacy: true } : null;
  }
  async saveBox(box: BoxState) {
    const { legacy: _, ...record } = box;
    const files = this.files(box);
    await this.fs.mkdirp(files.dir);
    // by rename: every box's idle check reads every record, and one read mid-write must not stop them all
    await this.fs.replaceText(files.instance, JSON.stringify(record, null, 2));
  }
  /** the box's state removed: its directory, or a legacy box's files beside what the boxes share */
  async clear(box: Pick<BoxState, "instanceId" | "legacy" | "localPort">) {
    const files = this.files(box);
    if (box.localPort !== undefined) await this.fs.remove(this.portClaim(box.localPort));
    if (!box.legacy) return this.fs.remove(files.dir);
    for (const f of [files.instance, files.tunnelEnv, files.knownHosts, files.idle])
      await this.fs.remove(f);
  }
  async idle(box: BoxState): Promise<IdleState | null> {
    return this.read<IdleState>(this.files(box).idle);
  }
  async saveIdle(box: BoxState, s: IdleState) {
    const files = this.files(box);
    await this.fs.mkdirp(files.dir);
    await this.fs.replaceText(files.idle, JSON.stringify(s));
  }

  target(box: BoxState): SshTarget {
    if (!box.sshHost || !box.sshPort)
      throw new Error(`box ${box.instanceId} has no ssh endpoint yet`);
    return {
      host: box.sshHost,
      port: box.sshPort,
      user: "root",
      knownHosts: this.files(box).knownHosts,
    };
  }

  private async read<T = BoxState>(path: string): Promise<T | null> {
    return (await this.fs.exists(path)) ? (JSON.parse(await this.fs.readText(path)) as T) : null;
  }
  /** a box's record, or null (said) when it does not parse: a hand edit, a disk that filled under a write */
  private async record(path: string): Promise<BoxState | null> {
    try {
      return await this.read(path);
    } catch (error) {
      this.warn(`${path} does not parse (${(error as Error).message}); that box is skipped`);
      return null;
    }
  }
}

/** hours billed so far, to two decimals */
export function billedHours(box: Pick<BoxState, "createdAt">, now: number): number {
  return Math.round(((now - box.createdAt) / 3_600_000) * 100) / 100;
}

export function billedCost(hours: number, dph: number): number {
  return Math.round(hours * dph * 100) / 100;
}
