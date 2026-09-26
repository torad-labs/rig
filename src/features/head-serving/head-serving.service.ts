// serve = verify, plan, run. `verify` is what the unit runs before every start (ExecStartPre):
// the build for this card is complete, the served pack matches its pinned sha256, every asset
// the args name exists. `plan` derives the command from the head, the card and the host; `run`
// starts llama-server as a child and forwards signals, for hand runs and rented boxes — the
// systemd unit runs the planned command directly (serve/manage-unit.ts) so no supervisor sits between
// systemd and the server.

import { artifactProblem, checkArtifact } from "../../shared/artifact.ts";
import type { Engine } from "../../shared/engine/engine.ts";
import { isBuilt } from "../../shared/engine/engine.ts";
import type { Head } from "../../shared/head/head.ts";
import { draftSidecar, type Speculative, tierSpeculates } from "../../shared/head/head-config.ts";
import { headVramMiB, pickTier } from "../../shared/head/tier.ts";
import type { FileSystem, Gpu, Hasher, Host, Log, Shell } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import { defaultCacheRam } from "./geometry.ts";
import { serverArgv, serverEnv } from "./server-argv.ts";

export interface VerifyReport {
  binDir: string;
  cap: string;
  servedPath: string;
  draftPath?: string | undefined;
}
export interface PlanOptions {
  gpu: number;
  cacheRam?: number | undefined;
  slots?: number | undefined;
  ctx?: number | undefined;
}
export interface ServePlan {
  argv: string[];
  env: Record<string, string>;
  binDir: string;
  gpu: number;
  slots: number;
  ctx: number;
  cacheRam: number;
  vramMiB: number;
  speculative: boolean;
}
export interface ServeHeadDeps {
  shell: Shell;
  fs: FileSystem;
  gpu: Gpu;
  hasher: Hasher;
  host: Host;
  log: Log;
}

export class ServeHead {
  constructor(
    private readonly deps: ServeHeadDeps,
    private readonly engine: Engine,
  ) {}

  /** `pack`, when given, is the exact `-m` path a rendered unit's ExecStart carries (unit-file.ts):
   *  verify checks THAT file against whichever pin it names (source or served — either the file may
   *  legitimately be, since installing a unit fixes `-m` while this machine's own resolution can
   *  move later, going undrived or back). A pinned file that is intact but not this machine's
   *  current resolution PASSES with a warning: refusing an intact pinned pack would turn a
   *  resolution the unit never asked to change into an outage. Without `pack` (old units),
   *  behaviour is unchanged: the served pack this machine currently resolves to. */
  async verify(head: Head, gpu: number, pack?: string): Promise<Result<VerifyReport>> {
    if (head.undrived) this.deps.log.warn(head.undrived);
    const card = await this.deps.gpu.query(gpu);
    if (!card) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${gpu}`);
    const binDir = this.engine.binDir(card.computeCap);
    if (!(await isBuilt(this.deps.fs, binDir)))
      return fail(
        ExitCode.Failure,
        `REFUSING to start: no complete build at ${binDir} for sm_${card.computeCap} (run: rig build)`,
      );
    const checkedPath = pack ?? head.servedPath;
    if (pack === undefined) {
      const served = await checkArtifact(this.deps.fs, this.deps.hasher, {
        path: head.servedPath,
        sha256: head.served.sha256,
      });
      if (served !== "ok")
        return fail(
          ExitCode.Failure,
          `REFUSING to start: the served pack is ${artifactProblem(served)} at ${head.servedPath} (run: rig derive)`,
        );
    } else {
      const pinned = [
        { path: head.sourcePath, sha256: head.source.sha256, label: "source", remedy: "rig fetch" },
        ...(head.declaredPublic
          ? [{ ...head.declaredPublic, label: "public", remedy: "rig fetch && rig derive" }]
          : []),
        {
          path: head.declaredServed.path,
          sha256: head.declaredServed.sha256,
          label: "served",
          remedy: "rig derive",
        },
      ];
      const match = pinned.find((p) => p.path === pack);
      if (!match)
        return fail(
          ExitCode.Failure,
          `REFUSING to start: --pack ${pack} is none of the pinned packs (${pinned.map((p) => `${p.label} ${p.path}`).join(", ")})`,
        );
      const state = await checkArtifact(this.deps.fs, this.deps.hasher, match);
      if (state !== "ok")
        return fail(
          ExitCode.Failure,
          `REFUSING to start: --pack ${pack} is ${artifactProblem(state)} (run: ${match.remedy})`,
        );
      if (pack !== head.servedPath) {
        this.deps.log.warn(
          `--pack ${pack} is the pinned ${match.label} pack, but ${head.name} now resolves to ${head.servedPath}` +
            (head.undrived ? ` (${head.undrived})` : "") +
            ` — the installed unit and this machine's resolution disagree; realign with: torad model pull ${head.name}-derive && rig derive ${head.name} && rig unit install ${head.name} (or, once the pack you want is already on disk: rig unit install ${head.name})`,
        );
      }
    }
    // every list that reaches the command line (server-argv.ts), not only runtime.args
    for (const arg of [
      ...head.runtime.args,
      ...head.runtime.extra,
      ...(head.runtime.lens?.enabled ? head.runtime.lens.args : []),
      ...(head.speculative?.args ?? []),
    ]) {
      if (arg.startsWith("assets/") && !(await this.deps.fs.exists(head.path(arg)))) {
        return fail(
          ExitCode.Failure,
          `REFUSING to start: asset ${arg} is missing from ${head.dir}`,
        );
      }
    }
    const tier = pickTier(head, await headVramMiB(this.deps, card, head.port));
    if (!tier.ok) return tier;
    const sidecar = head.speculative && draftSidecar(head.speculative);
    if (tierSpeculates(head, tier.value) && sidecar) {
      const draft = await checkArtifact(this.deps.fs, this.deps.hasher, {
        path: head.draftPath!,
        sha256: sidecar.sha256,
      });
      if (draft !== "ok")
        return fail(
          ExitCode.Failure,
          `REFUSING to start: the draft head is ${artifactProblem(draft)} at ${head.draftPath} (run: rig fetch)`,
        );
      return ok({
        binDir,
        cap: card.computeCap,
        servedPath: checkedPath,
        draftPath: head.draftPath,
      });
    }
    return ok({ binDir, cap: card.computeCap, servedPath: checkedPath });
  }

  async plan(head: Head, options: PlanOptions): Promise<Result<ServePlan>> {
    const card = await this.deps.gpu.query(options.gpu);
    if (!card) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${options.gpu}`);
    const vramMiB = await headVramMiB(this.deps, card, head.port);
    const tier = pickTier(head, vramMiB);
    if (!tier.ok) return tier;
    const slots = options.slots ?? tier.value.slots;
    if (slots < 1 || slots > tier.value.slots)
      return fail(
        ExitCode.Failure,
        `REFUSING: ${slots} slots — this card's tier (${vramMiB} MiB for the head) holds 1 to ${tier.value.slots}`,
      );
    // fewer slots than the tier's share its pool, never a longer window than the model's own for one conversation
    const ctx = options.ctx ?? Math.min(tier.value.ctx, slots * head.context.model);
    const cacheRam = options.cacheRam ?? defaultCacheRam(await this.deps.host.ramMiB());
    const binDir = this.engine.binDir(card.computeCap);
    const speculative = tierSpeculates(head, tier.value);
    return ok({
      argv: serverArgv(head, binDir, { slots, ctx, cacheRam, speculative: tier.value.speculative }),
      env: serverEnv(binDir, options.gpu),
      binDir,
      gpu: options.gpu,
      slots,
      ctx,
      cacheRam,
      vramMiB,
      speculative,
    });
  }

  /** Start the planned server as a child, forward SIGINT/SIGTERM, exit with its code. */
  async run(head: Head, plan: ServePlan, extra: readonly string[] = []): Promise<number> {
    this.deps.log.info(serveLogLine(head, plan));
    const child = this.deps.shell.spawn([...plan.argv, ...extra], { env: plan.env });
    const forward = (sig: "SIGINT" | "SIGTERM") => () => child.kill(sig);
    process.on("SIGINT", forward("SIGINT"));
    process.on("SIGTERM", forward("SIGTERM"));
    return child.exited;
  }
}

/** how the serve log names the draft: its file, or the in-pack head by type */
function draftLabel(s: Speculative): string {
  return "file" in s ? s.file : `${s.type} (in the pack)`;
}

/** the line `run` logs before spawning: undrived is named here too, since a hand run or a rented
 *  box never sees `rig describe`'s undrived field and `verify`'s own warning scrolls past before
 *  the server has anything to say */
export function serveLogLine(head: Head, plan: ServePlan): string {
  const draft = plan.speculative ? ` + draft ${draftLabel(head.speculative!)}` : "";
  const undrived = head.undrived ? ` UNDRIVED: ${head.undrived}` : "";
  return `serve ${head.name}: gpu ${plan.gpu} ${plan.binDir.split("/").at(-1)} vram=${plan.vramMiB}MiB -> -np ${plan.slots} -c ${plan.ctx} --cache-ram ${plan.cacheRam}${draft}; pack sha verified${undrived}`;
}
