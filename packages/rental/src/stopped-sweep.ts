// `rig vast sweep`: the cost control for rig's boxes that no timer of this machine watches. A box rented from a head's
// template in vast's console stops itself once idle (box-guard.ts), and a stop keeps its disk, which bills until someone
// destroys it: 160 GB at 0.333 $/GB/month is 0.074 $/h, $53 a month (offer 50138870, 2026-09-30), and vast destroys
// nothing on its own ("Stopped instances: Data persists, storage charges continue", docs.vast.ai). The sweep, hourly from
// its timer, lists the account's boxes and destroys one of rig's that has not been running for `stopped_hours`; a stop
// within that keeps the pack for a restart. Rig's is a box labelled as rig labels its boxes, or an unlabelled one running
// an image from rig's registry (a box rented from the template in the console); a box with any other label was kept on
// purpose (local/box-idle's 5080 and 5090) and is never touched. The box `vast up` rented is left to its own idle timer, a
// running box to its own guard. vast lists no time a box stopped, so the hours count from the first sweep that saw it
// stopped, kept in local/rented-box/stopped.json.
import { join } from "node:path";
import type { Clock, FileSystem, Instance, Layout, Log, Rental, Systemd } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import { loadRegistryConfig } from "@rig/image";
import { RentalState } from "./box-state.ts";
import { loadVastConfig } from "./rental-config.ts";
import { renderSweepService, renderSweepTimer, SWEEP_SERVICE, SWEEP_TIMER } from "./units.ts";

export interface SweepDeps {
  rental: Rental;
  fs: FileSystem;
  clock: Clock;
  log: Log;
  systemd: Systemd;
  /** argv that runs this rig: the timer's service runs `<self> vast sweep` */
  self: readonly string[];
}

export interface SweepReport {
  destroyed: number[];
  /** rig's boxes not running, and how long the sweeps have seen them so */
  stopped: Array<{ id: number; status: string; hours: number }>;
}

const HOUR_MS = 3_600_000;

export class SweepStopped {
  private readonly state: RentalState;

  constructor(
    private readonly deps: SweepDeps,
    private readonly layout: Layout,
  ) {
    this.state = new RentalState(deps.fs, layout);
  }

  private get seenFile() {
    return this.state.path("stopped.json");
  }

  async run(): Promise<Result<SweepReport>> {
    const { rental, fs, clock, log } = this.deps;
    const config = await loadVastConfig(fs, this.layout);
    if (!config.ok) return config;
    const registry = await loadRegistryConfig(fs, this.layout.root);
    if (!registry.ok) return registry;
    const image = registry.value ? `${registry.value.host}/${registry.value.repository}` : null;
    const { label, stopped_hours: stoppedHours } = config.value.rental;
    // an image in rig's repository: host/repository:tag, or @sha256:… by digest
    const fromRegistry = (ref = "") =>
      image !== null && (ref.startsWith(`${image}:`) || ref.startsWith(`${image}@`));
    const ours = (box: Instance) =>
      box.label === label || (box.label === "" && fromRegistry(box.image));

    const tracked = (await this.state.box())?.instanceId;
    let listed: Instance[];
    try {
      listed = await rental.list();
    } catch (error) {
      return fail(
        ExitCode.Failure,
        `vast could not be read: ${(error as Error).message}; nothing swept`,
      );
    }
    const seen = await this.seen();
    const now = clock.now();
    const next: Record<string, number> = {};
    const report: SweepReport = { destroyed: [], stopped: [] };
    const asked: number[] = [];
    for (const box of listed) {
      if (!ours(box) || box.id === tracked || box.status === "running") continue;
      const since = seen[box.id] ?? now;
      const hours = Math.round(((now - since) / HOUR_MS) * 10) / 10;
      report.stopped.push({ id: box.id, status: box.status, hours });
      next[box.id] = since;
      if (now - since < stoppedHours * HOUR_MS) continue;
      log.info(`box ${box.id} has not been running for ${hours} h (${box.status}): destroying it`);
      try {
        await rental.destroy(box.id);
        asked.push(box.id);
      } catch (error) {
        log.error(`vast did not take the destroy of box ${box.id}: ${(error as Error).message}`);
      }
    }
    // a destroy is believed when the listing no longer has the box: one still listed bills, and the next sweep asks again
    let still: number[] = [];
    if (asked.length > 0) {
      await clock.sleep(5000);
      const after = new Set((await rental.list()).map((box) => box.id));
      still = asked.filter((id) => after.has(id));
      for (const id of asked) if (!after.has(id)) delete next[id];
      report.destroyed = asked.filter((id) => !after.has(id));
      report.stopped = report.stopped.filter((box) => !report.destroyed.includes(box.id));
    }
    await fs.mkdirp(this.state.dir);
    await fs.writeText(this.seenFile, `${JSON.stringify(next, null, 2)}\n`);
    if (still.length > 0)
      return fail(
        ExitCode.Failure,
        `box ${still.join(", ")} STILL listed after destroy: it bills, and the next sweep asks again`,
      );
    return ok(report);
  }

  /** the sweep's timer installed and started: hourly while this machine is up */
  async arm(): Promise<void> {
    const { fs, systemd } = this.deps;
    const dir = systemd.unitDir();
    await fs.mkdirp(dir);
    const units = {
      [SWEEP_SERVICE]: renderSweepService({ self: this.deps.self }),
      [SWEEP_TIMER]: renderSweepTimer(),
    };
    let changed = false;
    for (const [name, text] of Object.entries(units)) {
      const path = join(dir, name);
      if ((await fs.exists(path)) && (await fs.readText(path)) === text) continue;
      await fs.replaceText(path, text);
      changed = true;
    }
    if (changed) await systemd.daemonReload();
    await systemd.enable(SWEEP_TIMER);
    await systemd.restart(SWEEP_TIMER);
  }

  private async seen(): Promise<Record<string, number>> {
    return (await this.deps.fs.exists(this.seenFile))
      ? (JSON.parse(await this.deps.fs.readText(this.seenFile)) as Record<string, number>)
      : {};
  }
}
