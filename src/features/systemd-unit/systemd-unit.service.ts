// install: render the unit for this head on this machine and put it in place — a changed unit is
// backed up beside itself with a date first (it lives outside version control), then written,
// daemon-reloaded and enabled. Never started here: starting is `up`'s decision, after the build
// and the pack have been verified and the live head is idle. The host's --cache-ram choice lives
// IN THE UNIT: without --cache-ram, the value the installed unit carries is kept, so a re-run
// cannot silently move a shared host to the box rule (RAM/4).
import { join } from "node:path";
import type { Head } from "../../shared/head/head.ts";
import type { Layout } from "../../shared/layout.ts";
import type { Clock, FileSystem, Log, Systemd } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import { compactStamp } from "../../shared/stamp.ts";
import { cacheRamOf, renderUnit, unitName } from "./unit-file.ts";

/** what unit needs from serve: a plan for the head on this machine (injected; unit never imports serve) */
export interface Planner {
  plan(
    head: Head,
    options: { gpu: number; cacheRam?: number | undefined },
  ): Promise<Result<{ argv: string[]; env: Record<string, string>; cacheRam: number }>>;
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
}
export interface InstallReport {
  unit: string;
  path: string;
  state: "current" | "installed" | "updated";
  backup?: string;
  cacheRam: number;
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

  async render(
    head: Head,
    options: InstallOptions,
  ): Promise<Result<{ text: string; cacheRam: number }>> {
    const path = this.unitPath(head);
    const previous = (await this.deps.fs.exists(path))
      ? cacheRamOf(await this.deps.fs.readText(path))
      : undefined;
    if (
      options.cacheRam === undefined &&
      previous === undefined &&
      (await this.deps.fs.exists(path))
    )
      this.deps.log.warn(
        `${path} exists but carries no --cache-ram to keep; the box rule (RAM/4) applies — pass --cache-ram to pin a shared host's bound`,
      );
    const plan = await this.deps.planner.plan(head, {
      gpu: options.gpu,
      cacheRam: options.cacheRam ?? previous,
    });
    if (!plan.ok) return plan;
    const text = renderUnit({
      head,
      root: this.layout.root,
      logPath: join(this.layout.logsDir, `${head.name}.log`),
      argv: plan.value.argv,
      env: plan.value.env,
      gpu: options.gpu,
      self: this.deps.self,
    });
    return ok({ text, cacheRam: plan.value.cacheRam });
  }

  async install(head: Head, options: InstallOptions): Promise<Result<InstallReport>> {
    const rendered = await this.render(head, options);
    if (!rendered.ok) return rendered;
    const path = this.unitPath(head);
    const unit = unitName(head.name);
    await this.deps.fs.mkdirp(this.deps.systemd.unitDir());
    await this.deps.fs.mkdirp(this.layout.logsDir);
    const existed = await this.deps.fs.exists(path);
    if (existed && (await this.deps.fs.readText(path)) === rendered.value.text)
      return ok({ unit, path, state: "current", cacheRam: rendered.value.cacheRam });
    let backup: string | undefined;
    if (existed) {
      backup = `${path}.${compactStamp(this.deps.clock.now())}.bak`;
      await this.deps.fs.copy(path, backup);
    }
    await this.deps.fs.replaceText(path, rendered.value.text);
    await this.deps.systemd.daemonReload();
    await this.deps.systemd.enable(unit);
    this.deps.log.info(
      `${existed ? "updated" : "installed"} ${path} (GPU ${options.gpu}, --cache-ram ${rendered.value.cacheRam}) and enabled it; it takes effect at the next restart`,
    );
    return ok({
      unit,
      path,
      state: existed ? "updated" : "installed",
      ...(backup ? { backup } : {}),
      cacheRam: rendered.value.cacheRam,
    });
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
        `${unit} is active — stop it first (rig up --stop, or systemctl --user stop ${unit})`,
      );
    await this.deps.systemd.disable(unit);
    await this.deps.fs.remove(path);
    await this.deps.systemd.daemonReload();
    return ok({ unit, removed: true });
  }
}
