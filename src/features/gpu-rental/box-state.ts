// One box at a time. instance.json is written by up and removed by down; tunnel.env feeds the
// tunnel unit; known_hosts is per box (a fresh host key on a fresh host:port every time);
// idle.json is the idle check's memory between runs.
import { join } from "node:path";
import type { Layout } from "../../shared/layout.ts";
import type { FileSystem, SshTarget } from "../../shared/ports/index.ts";

export interface BoxState {
  instanceId: number;
  offerId: number;
  gpu: string;
  gpus?: number;
  cap: string;
  dph: number;
  geo: string;
  createdAt: number;
  sshHost?: string;
  sshPort?: number;
  head?: string;
  /** `up --idle-minutes`: this box's idle budget, read by every idle check */
  idleMinutes?: number;
}
export interface IdleState {
  key: string;
  ts: number;
}

export class RentalState {
  readonly dir: string;
  constructor(
    private readonly fs: FileSystem,
    layout: Layout,
  ) {
    this.dir = layout.rentedBoxDir;
  }
  path(name: string) {
    return join(this.dir, name);
  }
  get instanceFile() {
    return this.path("instance.json");
  }
  get tunnelEnv() {
    return this.path("tunnel.env");
  }
  get knownHosts() {
    return this.path("known_hosts");
  }
  get idleFile() {
    return this.path("idle.json");
  }
  get cachedBuildsDir() {
    return this.path("cached-builds");
  }
  get pulledRunsDir() {
    return this.path("pulled-runs");
  }

  async box(): Promise<BoxState | null> {
    return (await this.fs.exists(this.instanceFile))
      ? (JSON.parse(await this.fs.readText(this.instanceFile)) as BoxState)
      : null;
  }
  async saveBox(b: BoxState) {
    await this.fs.mkdirp(this.dir);
    await this.fs.writeText(this.instanceFile, JSON.stringify(b, null, 2));
  }
  async clear() {
    for (const f of [this.instanceFile, this.tunnelEnv, this.knownHosts, this.idleFile])
      await this.fs.remove(f);
  }
  async idle(): Promise<IdleState | null> {
    return (await this.fs.exists(this.idleFile))
      ? (JSON.parse(await this.fs.readText(this.idleFile)) as IdleState)
      : null;
  }
  async saveIdle(s: IdleState) {
    await this.fs.writeText(this.idleFile, JSON.stringify(s));
  }

  target(b: BoxState): SshTarget {
    if (!b.sshHost || !b.sshPort) throw new Error(`box ${b.instanceId} has no ssh endpoint yet`);
    return { host: b.sshHost, port: b.sshPort, user: "root", knownHosts: this.knownHosts };
  }
}

/** hours billed so far, to two decimals */
export function billedHours(box: Pick<BoxState, "createdAt">, now: number): number {
  return Math.round(((now - box.createdAt) / 3_600_000) * 100) / 100;
}

export function billedCost(hours: number, dph: number): number {
  return Math.round(hours * dph * 100) / 100;
}
