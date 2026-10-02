// serve = verify, plan, run. `verify` is what the unit runs before every start (ExecStartPre):
// the build for the cards is complete, the served pack matches its pinned sha256, every asset
// the args name exists. `plan` derives the command from the head, the cards and the host; `run`
// starts llama-server as a child and forwards signals, for hand runs and rented boxes — the
// systemd unit runs the planned command directly (serve/manage-unit.ts) so no supervisor sits between
// systemd and the server.

import type { Devices, FileSystem, Gpu, Hasher, Host, Log, Shell } from "@rig/core";
import {
  type Artifact,
  artifactProblem,
  checkArtifact,
  ExitCode,
  fail,
  ok,
  type Result,
} from "@rig/core";
import type { Engine } from "@rig/engine";
import { isBuilt } from "@rig/engine";
import type { Head } from "@rig/head";
import {
  type CacheFormats,
  cacheRefusal,
  draftSidecar,
  type Profile,
  placeHead,
  placementLabel,
  profileCache,
  profileSpeculates,
  type Speculative,
} from "@rig/head";
import { defaultCacheRam } from "./geometry.ts";
import { serverArgv, serverEnv } from "./server-argv.ts";

export interface VerifyReport {
  binDir: string;
  cap: string;
  servedPath: string;
  draftPath?: string | undefined;
}
export interface PlanOptions {
  devices: Devices;
  cacheRam?: number | undefined;
  slots?: number | undefined;
  ctx?: number | undefined;
}
export interface ServePlan {
  argv: string[];
  env: Record<string, string>;
  binDir: string;
  /** the cards the profile takes, by nvidia-smi index */
  gpus: number[];
  slots: number;
  ctx: number;
  cacheRam: number;
  /** the VRAM the head has on each of them */
  vramMiB: number[];
  /** the profile this machine serves */
  profile: Profile;
  speculative: boolean;
  /** the profile's cache formats, rendered into argv */
  cache: CacheFormats;
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
  async verify(head: Head, devices: Devices, pack?: string): Promise<Result<VerifyReport>> {
    if (head.undrived) this.deps.log.warn(head.undrived);
    const placed = await placeHead(this.deps, head, devices);
    if (!placed.ok) return placed;
    const { profile, cards } = placed.value;
    const cap = cards[0]!.cap; // a profile's cards are of one compute capability
    const binDir = this.engine.binDir(cap);
    if (!(await isBuilt(this.deps.fs, binDir)))
      return fail(
        ExitCode.Failure,
        `REFUSING to start: no complete build at ${binDir} for sm_${cap} (run: rig build)`,
      );
    const checkedPath = pack ?? head.servedPath;
    if (pack === undefined) {
      const remedy = head.declaredServed ? "rig derive" : "rig fetch";
      const problem = await this.packProblem(head.servedFiles);
      if (problem)
        return fail(
          ExitCode.Failure,
          `REFUSING to start: the served pack is ${problem} (run: ${remedy})`,
        );
    } else {
      const pinned = [
        { files: head.sourceFiles, label: "source", remedy: "rig fetch" },
        ...(head.declaredPublic
          ? [{ files: [head.declaredPublic], label: "public", remedy: "rig fetch && rig derive" }]
          : []),
        ...(head.declaredServed
          ? [{ files: [head.declaredServed], label: "served", remedy: "rig derive" }]
          : []),
      ];
      const match = pinned.find((p) => p.files[0]!.path === pack);
      if (!match)
        return fail(
          ExitCode.Failure,
          `REFUSING to start: --pack ${pack} is none of the pinned packs (${pinned.map((p) => `${p.label} ${p.files[0]!.path}`).join(", ")})`,
        );
      const problem = await this.packProblem(match.files);
      if (problem)
        return fail(
          ExitCode.Failure,
          `REFUSING to start: --pack ${pack} is ${problem} (run: ${match.remedy})`,
        );
      if (pack !== head.servedPath) {
        this.deps.log.warn(
          `--pack ${pack} is the pinned ${match.label} pack, but ${head.name} now resolves to ${head.servedPath}` +
            (head.undrived ? ` (${head.undrived})` : "") +
            ` — the installed unit and this machine's resolution disagree; realign with: torad model pull ${head.name}-derive && rig derive ${head.name} && rig unit install ${head.name} (or, once the pack you want is already on disk: rig unit install ${head.name})`,
        );
      }
    }
    const bias = head.cache.mean_center;
    if (bias && !(await this.deps.fs.exists(head.path(bias))))
      return fail(
        ExitCode.Failure,
        `REFUSING to start: the K bias ${bias} is missing from ${head.dir}`,
      );
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
    const sidecar = head.speculative && draftSidecar(head.speculative);
    if (profileSpeculates(head, profile) && sidecar) {
      const draft = await checkArtifact(this.deps.fs, this.deps.hasher, {
        path: head.draftPath!,
        sha256: sidecar.sha256,
      });
      if (draft !== "ok")
        return fail(
          ExitCode.Failure,
          `REFUSING to start: the draft head is ${artifactProblem(draft)} at ${head.draftPath} (run: rig fetch)`,
        );
      return ok({ binDir, cap, servedPath: checkedPath, draftPath: head.draftPath });
    }
    return ok({ binDir, cap, servedPath: checkedPath });
  }

  async plan(head: Head, options: PlanOptions): Promise<Result<ServePlan>> {
    const placed = await placeHead(this.deps, head, options.devices);
    if (!placed.ok) return placed;
    const { profile, cards } = placed.value;
    const where = placementLabel(placed.value);
    const slots = options.slots ?? profile.slots;
    if (slots < 1 || slots > profile.slots)
      return fail(
        ExitCode.Failure,
        `REFUSING: ${slots} slots — ${where} holds 1 to ${profile.slots}`,
      );
    // fewer slots than the profile's share its pool, never a longer window than the model's own for one conversation;
    // a pool larger than the profile's is one its measurement never covered
    if (options.ctx !== undefined && (options.ctx < 1 || options.ctx > profile.ctx))
      return fail(
        ExitCode.Failure,
        `REFUSING: --ctx ${options.ctx} — ${where} holds a pool of 1 to ${profile.ctx} cells`,
      );
    const ctx = options.ctx ?? Math.min(profile.ctx, slots * head.context.model);
    const cacheRam = options.cacheRam ?? defaultCacheRam(await this.deps.host.ramMiB());
    const binDir = this.engine.binDir(cards[0]!.cap);
    const gpus = cards.map((card) => card.index);
    const speculative = profileSpeculates(head, profile);
    const cache = profileCache(head, profile);
    const refusal = cacheRefusal(
      this.engine,
      cache,
      speculative ? head.speculative?.cache : undefined,
    );
    if (refusal) return fail(ExitCode.Unsupported, `REFUSING on ${where}: ${refusal}`);
    const split = profile.split && { mode: profile.split, cards: cards.length };
    return ok({
      argv: serverArgv(head, binDir, {
        slots,
        ctx,
        cacheRam,
        speculative: profile.speculative,
        cache,
        split,
      }),
      env: serverEnv(binDir, gpus, profile),
      binDir,
      gpus,
      slots,
      ctx,
      cacheRam,
      vramMiB: cards.map((card) => card.vramMiB),
      profile,
      speculative,
      cache,
    });
  }

  /** why a pack's files are not its pinned bytes, the first file that is not, or null when every one is */
  private async packProblem(files: readonly Artifact[]): Promise<string | null> {
    for (const file of files) {
      const state = await checkArtifact(this.deps.fs, this.deps.hasher, file);
      if (state !== "ok") return `${artifactProblem(state)} at ${file.path}`;
    }
    return null;
  }

  /** plan, verify the cards the plan took, run: `rig serve`, and `rig up --foreground` once the head is ready */
  async serve(
    head: Head,
    options: PlanOptions,
    extra: readonly string[] = [],
  ): Promise<Result<number>> {
    const plan = await this.plan(head, options);
    if (!plan.ok) return plan;
    const verified = await this.verify(head, plan.value.gpus);
    if (!verified.ok) return verified;
    return ok(await this.run(head, plan.value, extra));
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
  const cache = `K/V ${plan.cache.k}/${plan.cache.v}${plan.cache.s ? ` state ${plan.cache.s}` : ""}`;
  const split = plan.profile.split ? ` ${plan.profile.split} split` : "";
  return `serve ${head.name}: gpu ${plan.gpus.join(",")} ${plan.binDir.split("/").at(-1)} vram=${plan.vramMiB.join("+")}MiB${split} -> -np ${plan.slots} -c ${plan.ctx} --cache-ram ${plan.cacheRam} ${cache}${draft}; pack sha verified${undrived}`;
}
