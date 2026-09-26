// install: render the unit for this head on this machine and put it in place — a changed unit is
// backed up beside itself with a date first (it lives outside version control), then written,
// daemon-reloaded and enabled. Never started here: starting is `up`'s decision, after the build
// and the pack have been verified and the live head is idle. The host's --cache-ram choice lives
// IN THE UNIT: without --cache-ram, the value the installed unit carries is kept, so a re-run
// cannot silently move a shared host to the box rule (RAM/4). An operator's `--slots N` (one slot
// where one conversation is served) is recorded in the unit and kept the same way; without one,
// the card's tier picks the slots at every render, and serve refuses a count the tier cannot hold.
import { join } from "node:path";
import type { Head } from "../../shared/head/head.ts";
import type { Layout } from "../../shared/layout.ts";
import type { Clock, FileSystem, Log, Systemd } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import { compactStamp } from "../../shared/stamp.ts";
import { cacheRamOf, renderUnit, slotsOf, unitName } from "./unit-file.ts";

/** what unit needs from serve: a plan for the head on this machine (injected; unit never imports serve) */
export interface Planner {
  plan(
    head: Head,
    options: { gpu: number; cacheRam?: number | undefined; slots?: number | undefined },
  ): Promise<
    Result<{ argv: string[]; env: Record<string, string>; cacheRam: number; slots: number }>
  >;
}
export interface ManageUnitDeps {
  fs: FileSystem;
  systemd: Systemd;
  clock: Clock;
  log: Log;
  planner: Planner;
  self: readonly string[];
}
export interface InstallOptions {
  gpu: number;
  cacheRam?: number | undefined;
  slots?: number | undefined;
}
export interface InstallReport {
  unit: string;
  path: string;
  /** where the unit writes the server's stdout and stderr (not the journal) */
  log: string;
  state: "current" | "installed" | "updated";
  backup?: string;
  cacheRam: number;
  slots: number;
  /** whether the unit outlives its user's last session (logind's Linger): null when unreadable */
  linger: boolean | null;
}
export interface UnitStatus {
  unit: string;
  path: string;
  installed: boolean;
  active: boolean;
  mainPid: number | null;
  oomScoreAdj?: number;
}

export class ManageUnit {
  constructor(
    private readonly deps: ManageUnitDeps,
    private readonly layout: Layout,
  ) {}

  unitPath(head: Head) {
    return join(this.deps.systemd.unitDir(), unitName(head.name));
  }

  logPath(head: Head) {
    return join(this.layout.logsDir, `${head.name}.log`);
  }

  async render(
    head: Head,
    options: InstallOptions,
  ): Promise<Result<{ text: string; cacheRam: number; slots: number }>> {
    const path = this.unitPath(head);
    const installed = (await this.deps.fs.exists(path))
      ? await this.deps.fs.readText(path)
      : undefined;
    const previous = installed === undefined ? undefined : cacheRamOf(installed);
    // --slots 0 gives the choice back to the card's tier
    const kept = installed === undefined ? undefined : slotsOf(installed);
    const chosen = options.slots === 0 ? undefined : (options.slots ?? kept);
    if (options.cacheRam === undefined && previous === undefined && installed !== undefined)
      this.deps.log.warn(
        `${path} exists but carries no --cache-ram to keep; the box rule (RAM/4) applies — pass --cache-ram to pin a shared host's bound`,
      );
    const plan = await this.deps.planner.plan(head, {
      gpu: options.gpu,
      cacheRam: options.cacheRam ?? previous,
      slots: chosen,
    });
    if (!plan.ok) return plan;
    const text = renderUnit({
      head,
      root: this.layout.root,
      logPath: this.logPath(head),
      argv: plan.value.argv,
      env: plan.value.env,
      gpu: options.gpu,
      self: this.deps.self,
      slots: chosen,
    });
    return ok({ text, cacheRam: plan.value.cacheRam, slots: plan.value.slots });
  }

  async install(head: Head, options: InstallOptions): Promise<Result<InstallReport>> {
    const rendered = await this.render(head, options);
    if (!rendered.ok) return rendered;
    const path = this.unitPath(head);
    const unit = unitName(head.name);
    await this.deps.fs.mkdirp(this.deps.systemd.unitDir());
    await this.deps.fs.mkdirp(this.layout.logsDir);
    const existed = await this.deps.fs.exists(path);
    if (existed && (await this.deps.fs.readText(path)) === rendered.value.text) {
      const linger = await this.ensureLinger();
      const { cacheRam, slots } = rendered.value;
      return ok({ unit, path, log: this.logPath(head), state: "current", cacheRam, slots, linger });
    }
    let backup: string | undefined;
    if (existed) {
      backup = `${path}.${compactStamp(this.deps.clock.now())}.bak`;
      await this.deps.fs.copy(path, backup);
    }
    await this.deps.fs.replaceText(path, rendered.value.text);
    await this.deps.systemd.daemonReload();
    await this.deps.systemd.enable(unit);
    this.deps.log.info(
      `${existed ? "updated" : "installed"} ${path} (GPU ${options.gpu}, -np ${rendered.value.slots}, --cache-ram ${rendered.value.cacheRam}) and enabled it; it takes effect at the next restart`,
    );
    return ok({
      unit,
      path,
      log: this.logPath(head),
      state: existed ? "updated" : "installed",
      ...(backup ? { backup } : {}),
      cacheRam: rendered.value.cacheRam,
      slots: rendered.value.slots,
      linger: await this.ensureLinger(),
    });
  }

  /** a user unit stops with its user's last session unless logind keeps the manager (linger): a
   *  head brought up over ssh or by a setup wizard would die at logout. Turned on where it is off
   *  (a user may set their own); named, with the command, where that is refused or unreadable. */
  private async ensureLinger(): Promise<boolean | null> {
    const linger = await this.deps.systemd.linger();
    if (linger === true) return true;
    if (linger === false && (await this.deps.systemd.enableLinger())) {
      if ((await this.deps.systemd.linger()) === true) {
        this.deps.log.info(
          "turned linger on for this user (loginctl enable-linger): the head outlives logout",
        );
        return true;
      }
    }
    this.deps.log.warn(
      linger === null
        ? "cannot read whether this user lingers (loginctl show-user): the head stops at logout unless it does — loginctl enable-linger"
        : "linger is off and loginctl enable-linger was refused: the head stops at logout — sudo loginctl enable-linger $USER",
    );
    return linger === null ? null : false;
  }

  async status(head: Head): Promise<UnitStatus> {
    const path = this.unitPath(head);
    const unit = unitName(head.name);
    const mainPid = await this.deps.systemd.mainPid(unit);
    // the RUNNING value, never the unit property: a --user manager silently clamps a negative
    // OOMScoreAdjust to the inherited +200, so only /proc says whether the head is shielded
    const adj =
      mainPid === null
        ? undefined
        : Number(
            (await this.deps.fs.readText(`/proc/${mainPid}/oom_score_adj`).catch(() => "")).trim(),
          );
    return {
      unit,
      path,
      installed: await this.deps.fs.exists(path),
      active: await this.deps.systemd.isActive(unit),
      mainPid,
      ...(adj !== undefined && Number.isFinite(adj) ? { oomScoreAdj: adj } : {}),
    };
  }

  async uninstall(head: Head): Promise<Result<{ unit: string; removed: boolean }>> {
    const path = this.unitPath(head);
    const unit = unitName(head.name);
    if (!(await this.deps.fs.exists(path))) return ok({ unit, removed: false });
    if (await this.deps.systemd.isActive(unit))
      return fail(
        ExitCode.Busy,
        `${unit} is active — stop it first: systemctl --user stop ${unit}`,
      );
    await this.deps.systemd.disable(unit);
    await this.deps.fs.remove(path);
    await this.deps.systemd.daemonReload();
    return ok({ unit, removed: true });
  }
}
