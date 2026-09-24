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

import { type Engine, prebuiltSkip } from "../../shared/engine/engine.ts";
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
