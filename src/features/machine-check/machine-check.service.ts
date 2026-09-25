// The questions a machine must answer before anything is downloaded or built, each with its own
// exit code so a caller can say the right thing: a tool is missing (1); the card is not one the
// engine's kernels are measured on (3); the driver cannot run the toolkit that would build them
// (4) — SASS is embedded for the card, so a too-old driver fails at load, after a 15-minute build.
// The card check lives here and not only in build so an unsupported card is refused before the
// 7.2 GB fetch. On a root box with apt (a rented instance) the tools are installed first; on a
// workstation they are only checked — a setup that apt-installs uninvited is the wrong kind of helpful.
// A card engine.toml pins a prebuilt build for needs no compiler: only the driver and what fetches
// and unpacks the build, and its driver is checked against the CUDA runtime the prebuilt carries;
// on a machine whose glibc is older than the build's floor the card compiles, and says why.

import { stagingPath } from "../../shared/artifact.ts";
import { type Engine, isBuilt, prebuiltSkip } from "../../shared/engine/engine.ts";
import type { Head } from "../../shared/head/head.ts";
import { deriveAsset, draftSidecar, type Tier } from "../../shared/head/head-config.ts";
import { pickTier } from "../../shared/head/tier.ts";
import type { FileSystem, Gpu, GpuInfo, Host, Log, Shell } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";

export const REQUIRED_TOOLS = ["git", "cmake", "ninja", "nvidia-smi", "curl"] as const;
/** a prebuilt engine is fetched and unpacked, never compiled (xz: NVIDIA's runtime archives) */
export const PREBUILT_TOOLS = ["nvidia-smi", "curl", "tar", "xz"] as const;
export const APT_PACKAGES = [
  "git",
  "cmake",
  "ninja-build",
  "build-essential",
  "pkg-config",
  "libcurl4-openssl-dev",
  "aria2",
  "ca-certificates",
  "curl",
] as const;
/** a root box whose card has a prebuilt installs only what fetches the packs and the engine */
export const PREBUILT_APT_PACKAGES = ["aria2", "ca-certificates", "curl", "xz-utils"] as const;
/** the most an engine install holds on disk at once: a prebuilt's 55 MB tarball, NVIDIA's 814 MB
 *  cuBLAS archive and the 664 MB they unpack to, before the downloads are removed; a compile holds
 *  less (205 MB of source, a 260 MB build tree, a 97 MB build). c008fe8, sm_120, 2026-09-24. */
export const ENGINE_PEAK_BYTES = 1_600_000_000;

export interface CheckMachineOptions {
  gpu: number;
  allowArch?: string | undefined;
  uid?: number | undefined;
}
export interface CheckMachineReport {
  card: GpuInfo;
  supported: boolean;
  toolkitCuda: string | null;
  driverCuda: string | null;
  installed: boolean;
  /** build installs the published prebuilt for this card instead of compiling */
  prebuilt: boolean;
  /** why the card's published prebuilt does not apply on this machine, when it has one */
  noPrebuilt?: string;
}
/** a file `rig up` has yet to write, at its pinned size less what a resumed download already holds */
export interface Need {
  what: string;
  bytes: number;
}
export interface RoomReport {
  tier: Tier;
  needs: Need[];
  freeBytes: number;
}
export interface CheckMachineDeps {
  shell: Shell;
  gpu: Gpu;
  fs: FileSystem;
  host: Host;
  log: Log;
}

export class CheckMachine {
  constructor(
    private readonly deps: CheckMachineDeps,
    private readonly engine: Engine,
  ) {}

  async run(options: CheckMachineOptions): Promise<Result<CheckMachineReport>> {
    // nvidia-smi comes with the driver, never from apt, so the card decides what a root box installs
    const card = (await this.deps.shell.which("nvidia-smi"))
      ? await this.deps.gpu.query(options.gpu)
      : null;
    const published = card ? this.engine.prebuiltFor(card.computeCap) : undefined;
    const noPrebuilt = published && prebuiltSkip(published, await this.deps.host.glibc());
    const prebuilt = published !== undefined && noPrebuilt === undefined;
    const installed = await this.installIfRootBox(
      options.uid ?? process.getuid?.() ?? 1000,
      prebuilt ? PREBUILT_APT_PACKAGES : APT_PACKAGES,
    );
    const missing: string[] = [];
    for (const tool of prebuilt ? PREBUILT_TOOLS : REQUIRED_TOOLS) {
      if (!(await this.deps.shell.which(tool))) missing.push(tool);
    }
    const toolkitCuda = await this.deps.gpu.toolkitCuda();
    if (!prebuilt && toolkitCuda === null) missing.push("cuda toolkit (nvcc)");
    if (missing.length) {
      const why = noPrebuilt ? ` (${noPrebuilt}, so the engine compiles here)` : "";
      return fail(ExitCode.Failure, `missing on this machine: ${missing.join(", ")}${why}`);
    }

    if (!card) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${options.gpu}`);
    const supported = this.engine.supports(card.computeCap);
    if (!supported && options.allowArch !== card.computeCap) {
      const list = this.engine.archs.map((arch) => `sm_${arch.cap} (${arch.cards})`).join(", ");
      return fail(
        ExitCode.Unsupported,
        `UNSUPPORTED card sm_${card.computeCap} (${card.name}) at index ${options.gpu} — the engine's kernels are measured on ${list}; --allow-arch ${card.computeCap} builds it for a benchmark`,
      );
    }
    const driverCuda = await this.deps.gpu.driverCuda();
    const runtimeCuda = prebuilt ? (this.engine.cuda?.version ?? null) : toolkitCuda;
    if (driverCuda && runtimeCuda && major(driverCuda) < major(runtimeCuda)) {
      const message = prebuilt
        ? `the driver supports CUDA ${driverCuda} but the prebuilt engine runs on the CUDA ${runtimeCuda} runtime: upgrade the driver to one that supports CUDA ${major(runtimeCuda)}`
        : `the driver supports CUDA ${driverCuda} but the toolkit is ${runtimeCuda}: a build would load-fail on this driver; upgrade the driver or install a ${major(driverCuda)}.x toolkit`;
      return fail(ExitCode.Driver, message);
    }
    const skipped = noPrebuilt ? { noPrebuilt } : {};
    return ok({ card, supported, toolkitCuda, driverCuda, installed, prebuilt, ...skipped });
  }

  /** The card against the head's smallest tier and the disk against every file the bring-up has
   *  yet to write, both before the first byte is fetched: otherwise a card too small for the head
   *  is refused only when its unit is written, after the 7.7 GB download, and a full disk only when
   *  a download or the derive dies on it. Everything rig writes lives under local/ (layout.ts), so
   *  the packs directory's filesystem is the one it fills. */
  async room(head: Head, options: { gpu: number }): Promise<Result<RoomReport>> {
    const card = await this.deps.gpu.query(options.gpu);
    if (!card) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${options.gpu}`);
    const tier = pickTier(head, card.memoryMiB);
    if (!tier.ok) {
      const message = `${card.name} at index ${options.gpu} cannot serve ${head.name}: ${tier.message}`;
      return fail(tier.code, message);
    }
    const needs = await this.needs(head, card.computeCap);
    const total = needs.reduce((sum, need) => sum + need.bytes, 0);
    const free = await this.deps.fs.freeBytes(head.packsDir);
    if (total > free) {
      const list = needs.map((need) => `${need.what} ${gb(need.bytes)}`).join(", ");
      return fail(
        ExitCode.Failure,
        `not enough disk under ${head.packsDir}: ${head.name} still needs ${gb(total)} (${list}) and ${gb(free)} is free`,
      );
    }
    this.deps.log.info(
      `${card.name} (${card.memoryMiB} MiB) fits ${head.name}; ${gb(total)} still to write, ${gb(free)} free`,
    );
    return ok({ tier: tier.value, needs, freeBytes: free });
  }

  /** each file `fetch`, `derive` and `build` would write, in that order, less what is there: a
   *  download resumes from its .part, and a bake removes its stale staged copy before it starts */
  private async needs(head: Head, cap: string): Promise<Need[]> {
    const files = [
      { what: head.source.file, path: head.sourcePath, bytes: head.source.bytes, staged: "part" },
    ];
    const sidecar = head.speculative && draftSidecar(head.speculative);
    if (sidecar && head.draftPath) {
      files.push({
        what: sidecar.file,
        path: head.draftPath,
        bytes: sidecar.bytes,
        staged: "part",
      });
    }
    for (const step of head.derive ?? []) {
      const asset = deriveAsset(step);
      if (asset.url && asset.bytes) {
        const path = head.assetPath(step);
        files.push({ what: asset.path, path, bytes: asset.bytes, staged: "part" });
      }
    }
    if (head.derive) {
      const { file, bytes } = head.served;
      files.push({ what: file, path: head.servedPath, bytes, staged: "deriving" });
    }
    const needs: Need[] = [];
    for (const file of files) {
      if (await this.deps.fs.exists(file.path)) continue;
      const held = (await this.deps.fs.stat(stagingPath(file.path, file.staged)))?.size ?? 0;
      needs.push({ what: file.what, bytes: Math.max(0, file.bytes - held) });
    }
    if (!(await isBuilt(this.deps.fs, this.engine.binDir(cap)))) {
      needs.push({ what: "the engine", bytes: ENGINE_PEAK_BYTES });
    }
    return needs;
  }

  private async installIfRootBox(uid: number, packages: readonly string[]): Promise<boolean> {
    if (uid !== 0 || !(await this.deps.shell.which("apt-get"))) return false;
    const env = { DEBIAN_FRONTEND: "noninteractive" };
    const update = await this.deps.shell.run(["apt-get", "update", "-qq"], {
      env,
      timeoutMs: 600_000,
    });
    if (update.code !== 0) throw new Error(`apt-get update: ${update.stderr.trim()}`);
    const install = await this.deps.shell.run(
      ["apt-get", "install", "-y", "-qq", "--no-install-recommends", ...packages],
      { env, timeoutMs: 1_200_000 },
    );
    if (install.code !== 0) throw new Error(`apt-get install: ${install.stderr.trim()}`);
    this.deps.log.info(`installed ${packages.length} apt packages (root box)`);
    return true;
  }
}

const major = (version: string) => Number(version.split(".")[0]);
const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;
