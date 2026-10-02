// up = prepare, fetch, build, derive, unit, start: the whole bring-up, each step another
// feature's use case behind an interface this feature declares and main.ts satisfies (up
// composes; it never imports a sibling). A head already serving is left running unless
// --restart, and a restart is refused while a request is processing or queued, or while the
// port cannot be read (exit 2): the unit then carries the new build and pack and takes effect
// at the next quiet restart. After a start the head that answers must be the unit's own
// process serving the pinned pack, or the start is reported as the failure it is. Every step's
// report is in the result, for a wizard. --foreground (a container, with no systemd) takes the
// same steps but the unit, then runs the server in this process until it exits.

import { basename } from "node:path";
import type { Clock, Devices, Host, Http, Log, Systemd } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import type { Head } from "@rig/head";
import { HeadEndpoint, type Serving } from "@rig/head";

export interface BringUpSteps {
  /** the cards the installed unit serves on, when there is one */
  unitDevices(head: Head): Promise<Devices | undefined>;
  /** the cards fit one of the head's profiles, which takes `gpus` of them, and the disk holds what the steps below
   *  will write */
  room(head: Head, options: { devices: Devices }): Promise<Result<{ gpus: number[] }>>;
  prepare(options: { gpu: number; allowArch?: string | undefined }): Promise<Result<unknown>>;
  fetch(head: Head): Promise<Result<{ state: string }>>;
  build(options: {
    gpu: number;
    allowArch?: string | undefined;
  }): Promise<Result<{ dir: string; alreadyBuilt: boolean }>>;
  derive(head: Head): Promise<Result<{ state: string }>>;
  installUnit(
    head: Head,
    options: { devices: Devices; cacheRam?: number | undefined; slots?: number | undefined },
  ): Promise<Result<{ unit: string; log: string; state: string; linger: boolean | null }>>;
  /** the server in this process until it exits, its exit code the result: plan, verify, run */
  serve(
    head: Head,
    options: { devices: Devices; cacheRam?: number | undefined; slots?: number | undefined },
  ): Promise<Result<number>>;
}

export interface BringUpHeadDeps {
  steps: BringUpSteps;
  http: Http;
  systemd: Systemd;
  host: Host;
  clock: Clock;
  log: Log;
}

export interface BringUpOptions {
  /** the cards to serve on; unset, the installed unit's, or auto for a first bring-up */
  devices?: Devices | undefined;
  /** switch a serving head to the new build and pack; refused while a slot is processing */
  restart?: boolean;
  cacheRam?: number | undefined;
  /** the unit's -np, kept through re-renders (one slot where the operator runs one conversation) */
  slots?: number | undefined;
  allowArch?: string | undefined;
  healthTimeoutMs?: number;
}

export interface StepStates {
  fetch: string;
  build: string;
  derive: string;
  unit: string;
}

export interface BringUpReport {
  steps: StepStates;
  /** whether the head outlives its user's logout (logind's Linger): null when unreadable */
  linger: boolean | null;
  start: "started" | "restarted" | "left-running";
  serving?: Serving;
}

const HEALTH_TIMEOUT_MS = 240_000;

export class BringUpHead {
  constructor(private readonly deps: BringUpHeadDeps) {}

  async run(head: Head, options: BringUpOptions): Promise<Result<BringUpReport>> {
    const prepared = await this.prepared(head, options);
    if (!prepared.ok) return prepared;
    const { steps, unit, log, linger } = prepared.value;

    this.announce("start");
    const endpoint = new HeadEndpoint(
      this.deps.http,
      this.deps.clock,
      `http://127.0.0.1:${head.port}`,
    );
    const presence = await endpoint.presence();
    // "unknown" is neither: a port that did not answer /health in time and did not refuse the
    // connection may hold a busy head, or nothing that is one — never reported as a head serving
    if (presence === "unknown") {
      return fail(
        ExitCode.Busy,
        `REFUSING to start — :${head.port} did not answer /health in time and did not refuse the connection; a head may be there and busy`,
      );
    }
    if (presence === "server" && !options.restart) {
      this.deps.log.info(
        `a head is already serving on :${head.port} — left running; the unit now carries the new build and pack. Switch when it is quiet: rig up ${head.name} --restart (refuses while a request is processing or queued)`,
      );
      return ok({ steps, linger, start: "left-running" });
    }
    if (presence === "server") {
      const inFlight = await endpoint.inFlight();
      if (inFlight === null) {
        return fail(
          ExitCode.Busy,
          `REFUSING to restart — cannot read /slots and /metrics on :${head.port}, so idle cannot be proved`,
        );
      }
      if (inFlight > 0) {
        const message = `REFUSING to restart — ${inFlight} request(s) processing or queued on :${head.port}; try again when it is idle`;
        return fail(ExitCode.Busy, message);
      }
    }

    await this.deps.systemd.restart(unit);
    const came = await endpoint.waitHealthy(
      options.healthTimeoutMs ?? HEALTH_TIMEOUT_MS,
      2000,
      () => this.deps.systemd.isActive(unit),
    );
    if (came !== "healthy") {
      const why = came === "dead" ? `${unit} is no longer active` : "did not come up";
      const message = `the head ${why} on :${head.port} — its log: tail -n 40 ${log}`;
      return fail(ExitCode.Failure, message);
    }
    // the answer must be this unit's process serving the pinned pack: a leftover server on the
    // port would answer 200 while the unit fails ExecStartPre and backs off
    const pid = await this.deps.systemd.mainPid(unit);
    if (pid === null) {
      return fail(
        ExitCode.Failure,
        `:${head.port} answers but ${unit} has no main process — something else holds the port`,
      );
    }
    // a unit with a main process and an answer on the port are two facts; the socket's owner
    // joins them (a leftover server answers 200 while the unit's own start dies on the bind)
    const owner = await this.deps.host.listeningPid(head.port);
    if (owner !== pid) {
      const who = owner === null ? "no process this user can see" : `pid ${owner}`;
      return fail(
        ExitCode.Failure,
        `:${head.port} is held by ${who}, not ${unit}'s main process (pid ${pid}) — a leftover server answers while the unit's own start fails on the port`,
      );
    }
    const serving = await endpoint.serving();
    const expected = basename(head.servedPath);
    if (serving.model !== expected) {
      return fail(
        ExitCode.Failure,
        `:${head.port} serves ${serving.model}, not the pinned ${expected} — ${unit} (pid ${pid}) is not what answers`,
      );
    }
    const undrived = head.undrived ? ` UNDRIVED: ${head.undrived}` : "";
    this.deps.log.info(
      `serving ${serving.model} with ${serving.slots} slots on 127.0.0.1:${head.port} (${unit}, pid ${pid})${undrived}`,
    );
    return ok({ steps, linger, start: presence === "server" ? "restarted" : "started", serving });
  }

  /** Every step `run` takes before the unit, then the server in this process until it exits: a container has no
   *  systemd, and its entrypoint is the server's supervisor. The result is the server's exit code. */
  async foreground(head: Head, options: BringUpOptions): Promise<Result<number>> {
    const ready = await this.ready(head, options);
    if (!ready.ok) return ready;
    this.announce("serve");
    const { gpus } = ready.value;
    return this.deps.steps.serve(head, {
      devices: gpus,
      cacheRam: options.cacheRam,
      slots: options.slots,
    });
  }

  /** everything before the start: the machine, the pack, the build, the derivation, the unit */
  private async prepared(
    head: Head,
    options: BringUpOptions,
  ): Promise<Result<{ steps: StepStates; unit: string; log: string; linger: boolean | null }>> {
    const ready = await this.ready(head, options);
    if (!ready.ok) return ready;
    const { gpus, fetched, built, derived } = ready.value;

    this.announce("unit");
    const unit = await this.deps.steps.installUnit(head, {
      devices: gpus,
      cacheRam: options.cacheRam,
      slots: options.slots,
    });
    if (!unit.ok) return unit;

    const steps: StepStates = {
      fetch: fetched,
      build: built,
      derive: derived,
      unit: unit.value.state,
    };
    const { log, linger } = unit.value;
    return ok({ steps, unit: unit.value.unit, log, linger });
  }

  /** the machine, the pack, the build, the derivation: what the head needs before it can start, and the cards it
   *  takes */
  private async ready(
    head: Head,
    options: BringUpOptions,
  ): Promise<Result<{ gpus: number[]; fetched: string; built: string; derived: string }>> {
    const devices = options.devices ?? (await this.deps.steps.unitDevices(head)) ?? "auto";

    this.announce("prepare");
    // the cards first: which of them the head takes decides the card prepare and build are for (a profile's cards are
    // of one compute capability), and a machine that cannot serve the head installs nothing
    const room = await this.deps.steps.room(head, { devices });
    if (!room.ok) return room;
    const { gpus } = room.value;
    const machine = { gpu: gpus[0]!, allowArch: options.allowArch };
    const prepared = await this.deps.steps.prepare(machine);
    if (!prepared.ok) return prepared;

    this.announce("fetch");
    const fetched = await this.deps.steps.fetch(head);
    if (!fetched.ok) return fetched;

    this.announce("build");
    const built = await this.deps.steps.build(machine);
    if (!built.ok) return built;

    this.announce("derive");
    const derived = await this.deps.steps.derive(head);
    if (!derived.ok) return derived;

    return ok({
      gpus,
      fetched: fetched.value.state,
      built: built.value.alreadyBuilt ? "present" : "built",
      derived: derived.value.state,
    });
  }

  private announce(step: string): void {
    this.deps.log.info(`== ${step}`);
  }
}
