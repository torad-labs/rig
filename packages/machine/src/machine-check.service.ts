// The questions a machine must answer before anything is downloaded or built, each with its own
// exit code so a caller can say the right thing: a tool is missing (1); the card is not one the
// engine's kernels are measured on (3); the driver cannot run the toolkit that would build them
// (4) — SASS is embedded for the card, so a too-old driver fails at load, after a 15-minute build;
// the compiler is one engine.toml's [[miscompilers]] knows to build the kernels wrong for the card (1).
// The card check lives here and not only in build so an unsupported card is refused before the
// 7.2 GB fetch. On a root box with apt (a rented instance) the tools are installed first; on a
// workstation they are only checked — a setup that apt-installs uninvited is the wrong kind of helpful.
// A card engine.toml pins a prebuilt build for needs no compiler: only the driver and what fetches
// and unpacks the build, and its driver is checked against the CUDA runtime the prebuilt carries;
// on a machine whose glibc is older than the build's floor the card compiles, and says why. A card
// whose build is already installed with that runtime beside it (a tarball's, as an image bakes it)
// needs no compiler either, whether or not a prebuilt is published.

import { join } from "node:path";
import type { Devices, FileSystem, Gpu, GpuInfo, Host, Log, Shell } from "@rig/core";
import { ExitCode, fail, ok, type Result, stagingPath } from "@rig/core";
import { type Engine, isBuilt, miscompiles, prebuiltSkip } from "@rig/engine";
import type { Head } from "@rig/head";
import {
  cacheRefusal,
  deriveAsset,
  draftSidecar,
  type Profile,
  placeHead,
  placementLabel,
  profileCache,
  profileSpeculates,
} from "@rig/head";

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
  /** the card's build is installed with the pin's CUDA runtime beside it: nothing is compiled */
  built: boolean;
  /** why the card's published prebuilt does not apply on this machine, when it has one */
  noPrebuilt?: string;
}
/** a file `rig up` has yet to write, at its pinned size less what a resumed download already holds */
export interface Need {
  what: string;
  bytes: number;
}
export interface RoomReport {
  profile: Profile;
  /** the cards the profile takes, by nvidia-smi index */
  gpus: number[];
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
    const built = card ? await this.builtWithRuntime(card.computeCap) : false;
    const compiles = !prebuilt && !built;
    const installed = await this.installIfRootBox(
      options.uid ?? process.getuid?.() ?? 1000,
      compiles ? APT_PACKAGES : PREBUILT_APT_PACKAGES,
    );
    const missing: string[] = [];
    for (const tool of compiles ? REQUIRED_TOOLS : PREBUILT_TOOLS) {
      if (!(await this.deps.shell.which(tool))) missing.push(tool);
    }
    const toolkitCuda = await this.deps.gpu.toolkitCuda();
    if (compiles && toolkitCuda === null) missing.push("cuda toolkit (nvcc)");
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
    const runtimeCuda = compiles ? toolkitCuda : (this.engine.cuda?.version ?? null);
    if (driverCuda && runtimeCuda && major(driverCuda) < major(runtimeCuda)) {
      const message = !compiles
        ? `the driver supports CUDA ${driverCuda} but the ${built ? "installed" : "prebuilt"} engine runs on the CUDA ${runtimeCuda} runtime: upgrade the driver to one that supports CUDA ${major(runtimeCuda)}`
        : `the driver supports CUDA ${driverCuda} but the toolkit is ${runtimeCuda}: a build would load-fail on this driver; upgrade the driver or install a ${major(driverCuda)}.x toolkit`;
      return fail(ExitCode.Driver, message);
    }
    const miscompiled = compiles
      ? miscompiles(this.engine, toolkitCuda, card.computeCap)
      : undefined;
    if (miscompiled) return fail(ExitCode.Failure, miscompiled);
    const skipped = noPrebuilt ? { noPrebuilt } : {};
    return ok({ card, supported, toolkitCuda, driverCuda, installed, prebuilt, built, ...skipped });
  }

  /** the card's build is published here with every library of the pin's CUDA runtime beside it:
   *  a tarball's install (build-publisher.ts), which runs on the driver alone. A build compiled here
   *  links the toolkit's runtime instead, and the toolkit is still checked for it. */
  private async builtWithRuntime(cap: string): Promise<boolean> {
    const libs = (this.engine.cuda?.runtime ?? []).flatMap((archive) => archive.libs);
    if (libs.length === 0) return false;
    const dir = this.engine.binDir(cap);
    if (!(await isBuilt(this.deps.fs, dir))) return false;
    for (const lib of libs) if (!(await this.deps.fs.exists(join(dir, lib)))) return false;
    return true;
  }

  /** The cards against the head's profiles and the disk against every file the bring-up has
   *  yet to write, both before the first byte is fetched: otherwise cards too small for the head
   *  are refused only when its unit is written, after the 7.7 GB download, and a full disk only when
   *  a download or the derive dies on it. Everything rig writes lives under local/ (layout.ts), so
   *  the packs directory's filesystem is the one it fills. */
  async room(head: Head, options: { devices: Devices }): Promise<Result<RoomReport>> {
    const placed = await placeHead(this.deps, head, options.devices);
    if (!placed.ok) return placed;
    const { profile, cards, machine } = placed.value;
    const taken = machine.filter(({ info }) => cards.some((card) => card.index === info.index));
    const named = taken.map(({ info }) => `${info.name} at index ${info.index}`).join(", ");
    // the formats serve would refuse on this profile, refused before the download rather than at the first start
    const draft = profileSpeculates(head, profile) ? head.speculative?.cache : undefined;
    const refusal = cacheRefusal(this.engine, profileCache(head, profile), draft);
    if (refusal)
      return fail(
        ExitCode.Unsupported,
        `${named} cannot serve ${head.name} on ${placementLabel(placed.value)}: ${refusal}`,
      );
    const needs = await this.needs(head, cards[0]!.cap);
    const total = needs.reduce((sum, need) => sum + need.bytes, 0);
    const free = await this.deps.fs.freeBytes(head.packsDir);
    if (total > free) {
      const list = needs.map((need) => `${need.what} ${gb(need.bytes)}`).join(", ");
      return fail(
        ExitCode.Failure,
        `not enough disk under ${head.packsDir}: ${head.name} still needs ${gb(total)} (${list}) and ${gb(free)} is free`,
      );
    }
    const rooms = taken.map(
      ({ info, room }) => `${info.name} (${room.vramMiB} of ${info.memoryMiB} MiB for it)`,
    );
    this.deps.log.info(
      `${rooms.join(" + ")} fits ${head.name} at ${profile.slots} slots; ${gb(total)} still to write, ${gb(free)} free`,
    );
    return ok({ profile, gpus: cards.map((card) => card.index), needs, freeBytes: free });
  }

  /** each file `fetch`, `derive` and `build` would write, in that order, less what is there: a
   *  download resumes from its .part, and a bake removes its stale staged copy before it starts */
  private async needs(head: Head, cap: string): Promise<Need[]> {
    const files = head.sourceFiles.map((file) => ({
      what: file.file,
      path: file.path,
      bytes: file.bytes,
      staged: "part",
    }));
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
      const { file, bytes } = head.servedFiles[0]!;
      files.push({ what: file, path: head.servedPath, bytes, staged: "deriving" });
      // splices of the same type and the ablation write in place; a pack whose size is not the
      // source's was re-laid out (a retyped draft head), a whole second copy beside the staged one
      if (bytes !== head.sourceFiles[0]!.bytes) {
        const relayout = { what: `${file}, re-laid out`, path: head.servedPath, bytes };
        files.push({ ...relayout, staged: "deriving.relayout" });
      }
    }
    const needs: Need[] = [];
    for (const file of files) {
      if (await this.deps.fs.exists(file.path)) continue;
      // what the staged file holds on disk, not its size: a download's .part is sparse at the whole size from the start
      const staged = await this.deps.fs.stat(stagingPath(file.path, file.staged));
      const held = staged ? Math.min(staged.size, staged.allocated) : 0;
      needs.push({ what: file.what, bytes: Math.max(0, file.bytes - held) });
    }
    if (!(await isBuilt(this.deps.fs, this.engine.binDir(cap)))) {
      needs.push({ what: "the engine", bytes: ENGINE_PEAK_BYTES });
    }
    return needs;
  }

  private async installIfRootBox(uid: number, packages: readonly string[]): Promise<boolean> {
    if (uid !== 0 || !(await this.deps.shell.which("apt-get"))) return false;
    // an image that baked them boots without reaching a mirror: dpkg answers from its own database
    const status = await this.deps.shell.run([
      "dpkg-query",
      "-W",
      `-f=\${db:Status-Status}\n`, // dpkg's field, not a template placeholder
      ...packages,
    ]);
    const states = status.stdout.trim().split("\n");
    if (
      status.code === 0 &&
      states.length === packages.length &&
      states.every((state) => state === "installed")
    )
      return false;
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
