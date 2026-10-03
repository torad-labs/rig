// Builds the engine at its pinned commit for the compute capability of the card this machine
// serves on, and publishes it as local/engine-builds/<sha7>-sm<cap>/, the one place env
// resolution looks. The same code runs on a workstation and on a rented box; what differs is
// what the machine has: `buildgate` on PATH boxes the build when the host provides it (ninja's
// default job count, nproc+2, built the 487 objects inside the 15 GiB box on 2026-09-20; --jobs
// overrides it); --portable builds the baseline and tars the result so the next box of the
// same arch skips the 5–20 minute build; --from-tarball publishes a cached tarball instead of
// compiling, with the pin's CUDA runtime beside it as a prebuilt gets (with --off-pin NAME, a tarball
// of another commit goes to local/engine-lab/NAME/ instead, for the lab to A/B against the pin); --cap names the compute
// capability where no card is present (an image build). Where engine.toml pins a prebuilt build of the pin for the card, that is installed
// instead (the build and NVIDIA's CUDA runtime, each by sha256), so a machine with only the driver
// needs no toolkit and no compile; --compile builds from source regardless. The command lines are
// cmake-invocation.ts; the rename that makes a build real is build-publisher.ts.
import { basename, join } from "node:path";
import type { Clock, FileSystem, Git, Gpu, Hasher, Host, Layout, Log, Shell } from "@rig/core";
import {
  type Artifact,
  checkArtifact,
  ExitCode,
  fail,
  fetchPinned,
  ok,
  type Result,
} from "@rig/core";
import {
  BUILD_MARKER,
  type Engine,
  engineSource,
  engineTarballName,
  isBuilt,
  miscompiles,
  type Prebuilt,
  prebuiltSkip,
  TARGETS,
} from "../engine.ts";
import { BuildPublisher, type RuntimeArchive } from "./build-publisher.ts";
import { buildArgv, configureArgv, pruneArgv } from "./cmake-invocation.ts";

export interface BuildEngineOptions {
  gpu: number;
  /** the compute capability to build for, in place of the card's: a machine without one (an image build) */
  cap?: string | undefined;
  /** build for a card the engine is not measured on, for a benchmark */
  allowArch?: string | undefined;
  /** the AVX2 baseline instead of -march=native, and a tarball of the result */
  portable?: boolean;
  jobs?: number | undefined;
  fromTarball?: string | undefined;
  /** with fromTarball: the tarball is a build of another commit, installed for the lab as
   *  local/engine-lab/<offPin>/ and never where env resolution looks */
  offPin?: string | undefined;
  /** from source even where a prebuilt build of the pin is published for this card */
  compile?: boolean;
}

export interface BuildEngineReport {
  dir: string;
  cap: string;
  alreadyBuilt: boolean;
  marker: string;
  tarball?: string;
}

export interface BuildEngineDeps {
  shell: Shell;
  fs: FileSystem;
  git: Git;
  gpu: Gpu;
  host: Host;
  clock: Clock;
  hasher: Hasher;
  log: Log;
}

/** what a portable build carries from the compiler that built it: ggml's CPU backend links the
 *  OpenMP runtime, and a machine with only the driver has libstdc++ and libgcc_s but no libgomp
 *  (ubuntu:24.04: the first prebuilt of c008fe8 refused to publish there, 2026-09-24, on
 *  GOMP_loop_nonmonotonic_dynamic_start). GCC's runtime library exception allows shipping it. */
export const PORTABLE_RUNTIME = ["libgomp.so.1"] as const;

/** a directory name under local/engine-lab/: one path segment, so an off-pin build lands nowhere else */
const LAB_NAME = /^(?!\.\.?$)[\w.-]+$/;

const CONFIGURE_TIMEOUT_MS = 1_800_000;
const BUILD_TIMEOUT_MS = 3_600_000;

export class BuildEngine {
  private readonly publisher: BuildPublisher;

  constructor(
    private readonly deps: BuildEngineDeps,
    private readonly layout: Layout,
    private readonly engine: Engine,
  ) {
    this.publisher = new BuildPublisher(deps, engine);
  }

  async run(options: BuildEngineOptions): Promise<Result<BuildEngineReport>> {
    const offPin = options.offPin;
    if (offPin !== undefined && options.fromTarball === undefined)
      return fail(
        ExitCode.Usage,
        "--off-pin names the lab directory a build of another commit installs into: give its tarball with --from-tarball FILE",
      );
    if (offPin !== undefined && !LAB_NAME.test(offPin))
      return fail(
        ExitCode.Usage,
        `--off-pin takes one directory name under local/engine-lab/ (letters, digits, '.', '_', '-'), not ${JSON.stringify(offPin)}`,
      );
    const card = options.cap === undefined ? await this.deps.gpu.query(options.gpu) : null;
    const cap = options.cap ?? card?.computeCap;
    if (cap === undefined)
      return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${options.gpu}`);
    if (!this.engine.supports(cap) && options.allowArch !== cap) {
      const measured = this.engine.archs.map((arch) => `sm_${arch.cap}`).join(", ");
      const message = `REFUSING sm_${cap}: not a measured card (engine.toml lists ${measured}); --allow-arch ${cap} builds it for a benchmark`;
      return fail(ExitCode.Unsupported, message);
    }
    if (offPin !== undefined && options.fromTarball !== undefined)
      return this.installOffPin(options.fromTarball, offPin, cap);
    // before "already built": that answers for the pin, which a tarball of another commit is not
    const misnamed = options.fromTarball && this.publisher.misnamed(options.fromTarball, cap);
    if (misnamed) return fail(ExitCode.Failure, misnamed);

    const dir = this.engine.binDir(cap);
    // --portable answers with a tarball of the portable baseline holding every target: a build here that is native,
    // installed, or made before a target existed is compiled again (its build tree is kept, so only what it lacks
    // compiles) and published over this one
    const portable = options.portable === true && options.fromTarball === undefined;
    if (await isBuilt(this.deps.fs, dir)) {
      const marker = (await this.deps.fs.readText(join(dir, BUILD_MARKER))).trim();
      if (!portable) return ok({ dir, cap, alreadyBuilt: true, marker });
      if (await this.compiledPortable(dir, marker)) {
        const tarball = await this.publisher.pack(dir, this.layout.engineBuildsDir, cap);
        if (!tarball.ok) return tarball;
        return ok({ dir, cap, alreadyBuilt: true, marker, tarball: tarball.value });
      }
    }
    await this.deps.fs.mkdirp(this.layout.engineBuildsDir);
    await this.deps.fs.mkdirp(this.layout.logsDir);

    // --portable produces the tarball a prebuilt is published from, so it always compiles; a
    // machine below the published build's glibc compiles too (the build would fail its ldd -r)
    const published =
      options.compile || options.portable ? undefined : this.engine.prebuiltFor(cap);
    const skip = published && prebuiltSkip(published, await this.deps.host.glibc());
    if (skip) this.deps.log.info(`${skip}: compiling the engine here`);
    const prebuilt = skip ? undefined : published;
    const marker = options.fromTarball
      ? await this.installTarball(options.fromTarball, cap, dir)
      : prebuilt
        ? await this.install(prebuilt, cap, dir)
        : await this.compile(cap, dir, options);
    if (!marker.ok) return marker;

    const report: BuildEngineReport = { dir, cap, alreadyBuilt: false, marker: marker.value };
    if (portable) {
      const tarball = await this.publisher.pack(dir, this.layout.engineBuildsDir, cap);
      if (!tarball.ok) return tarball;
      report.tarball = tarball.value;
    }
    return ok(report);
  }

  /** a build compiled here as the portable baseline (not installed: an installed one holds NVIDIA's runtime beside its
   *  libraries, which a prebuilt never carries) that holds every target this rig compiles */
  private async compiledPortable(dir: string, marker: string): Promise<boolean> {
    if (!/\bnative=off\b/.test(marker) || !/\bsource=compiled\b/.test(marker)) return false;
    for (const target of TARGETS) if (!(await this.deps.fs.exists(join(dir, target)))) return false;
    return true;
  }

  /** the pin's published build and the CUDA runtime it links, each fetched by sha256 into
   *  local/downloads/ (a file already there with the pinned bytes is kept), unpacked together and
   *  published; the downloads are removed once the build is, and kept for a retry if it is not */
  private async install(prebuilt: Prebuilt, cap: string, dir: string): Promise<Result<string>> {
    const cuda = this.engine.cuda;
    if (!cuda) return fail(ExitCode.Failure, "engine.toml pins a prebuilt without [cuda]");
    this.deps.log.info(
      `installing the published build of torad-labs/llama.cpp @ ${this.engine.sha7} for sm_${cap}, with NVIDIA's CUDA ${cuda.version} runtime → ${dir}`,
    );
    const tarball = {
      path: join(this.layout.downloadsDir, engineTarballName(this.engine.sha7, cap)),
      sha256: prebuilt.sha256,
    };
    const fetched = await this.obtain(prebuilt.url, tarball, `engine build for sm_${cap}`);
    if (!fetched.ok) return fetched;
    const runtime = await this.runtime();
    if (!runtime.ok) return runtime;
    const published = await this.publisher.publishTarball(tarball.path, cap, dir, runtime.value);
    if (published.ok) {
      for (const file of [tarball.path, ...runtime.value.map((archive) => archive.path)]) {
        await this.deps.fs.remove(file);
      }
    }
    return published;
  }

  /** a portable build of the pin from a file (an earlier box's, or one built for an image), with the pin's CUDA
   *  runtime beside it where engine.toml pins one, as a published build gets: a machine with only the driver runs it */
  private async installTarball(
    tarball: string,
    cap: string,
    dir: string,
    offPin?: { sha256: string },
  ): Promise<Result<string>> {
    const runtime = await this.runtime();
    if (!runtime.ok) return runtime;
    const published = await this.publisher.publishTarball(tarball, cap, dir, runtime.value, offPin);
    if (published.ok) for (const archive of runtime.value) await this.deps.fs.remove(archive.path);
    return published;
  }

  /** a build of another commit from its tarball (a candidate's `--portable` output) for the lab, as
   *  local/engine-lab/<name>/, never where env resolution looks, with the pin's CUDA runtime beside it as an installed
   *  build carries it: an A/B of it against the pin compares the engines on one cuBLAS, which the lab's runtime guard
   *  requires. The same tarball again is a no-op; another one under that name is refused. */
  private async installOffPin(
    tarball: string,
    name: string,
    cap: string,
  ): Promise<Result<BuildEngineReport>> {
    const { fs, hasher, log } = this.deps;
    if (!(await fs.exists(tarball))) return fail(ExitCode.Failure, `${tarball} does not exist`);
    const named = /^engine-sm(\d+)-/.exec(basename(tarball))?.[1];
    if (named !== undefined && named !== cap)
      return fail(
        ExitCode.Failure,
        `${basename(tarball)} is a build for sm_${named}, not this card's sm_${cap}`,
      );
    const dir = join(this.layout.engineLabDir, name);
    const sha256 = await hasher.sha256File(tarball);
    if (await fs.exists(dir)) {
      const built = await isBuilt(fs, dir);
      const marker = built ? (await fs.readText(join(dir, BUILD_MARKER))).trim() : "";
      if (marker.includes(`sha256=${sha256}`)) return ok({ dir, cap, alreadyBuilt: true, marker });
      return fail(
        ExitCode.Failure,
        `${dir} exists and is not ${basename(tarball)} (sha256 ${sha256.slice(0, 12)}): name another directory`,
      );
    }
    await fs.mkdirp(this.layout.engineLabDir);
    log.info(
      `installing ${basename(tarball)} OFF THE PIN (not torad-labs/llama.cpp @ ${this.engine.sha7}), with the pin's CUDA runtime, for the lab → ${dir}`,
    );
    const marker = await this.installTarball(tarball, cap, dir, { sha256 });
    if (!marker.ok) return marker;
    return ok({ dir, cap, alreadyBuilt: false, marker: marker.value });
  }

  /** NVIDIA's CUDA runtime the pin links, each archive fetched by sha256 into local/downloads/; none when engine.toml
   *  pins none */
  private async runtime(): Promise<Result<RuntimeArchive[]>> {
    const runtime: RuntimeArchive[] = [];
    for (const entry of this.engine.cuda?.runtime ?? []) {
      const archive = {
        path: join(this.layout.downloadsDir, basename(new URL(entry.url).pathname)),
        sha256: entry.sha256,
      };
      const got = await this.obtain(entry.url, archive, `CUDA runtime ${basename(archive.path)}`);
      if (!got.ok) return got;
      runtime.push({ path: archive.path, libs: entry.libs });
    }
    return ok(runtime);
  }

  private async obtain(url: string, target: Artifact, what: string): Promise<Result<void>> {
    if ((await checkArtifact(this.deps.fs, this.deps.hasher, target)) === "ok")
      return ok(undefined);
    this.deps.log.info(`fetching ${basename(target.path)}`);
    return fetchPinned(this.deps, url, target, what);
  }

  /** configure and build in local/engine-build-trees/<sha7>-sm<cap>/, logs beside, then publish */
  private async compile(
    cap: string,
    dir: string,
    options: BuildEngineOptions,
  ): Promise<Result<string>> {
    const source = await engineSource(this.deps.fs, this.deps.git, this.layout, this.engine);
    if (!source.ok) return source;
    const tag = `${this.engine.sha7}-sm${cap}`;
    const buildTree = join(this.layout.engineBuildTreesDir, tag);
    this.deps.log.info(
      `building torad-labs/llama.cpp @ ${this.engine.sha7} for sm_${cap} → ${dir}`,
    );
    if (!(await this.cacheMatches(buildTree, source.value))) await this.deps.fs.remove(buildTree);

    const portable = options.portable ?? false;
    const configure = await this.deps.shell.run(
      configureArgv({ source: source.value, buildTree, cap, portable }),
      { timeoutMs: CONFIGURE_TIMEOUT_MS },
    );
    await this.keepLog(`configure-${tag}.log`, configure.stdout + configure.stderr);
    if (configure.code !== 0) {
      return fail(ExitCode.Failure, `cmake configure failed (local/logs/configure-${tag}.log)`);
    }
    // the compiler cmake chose, which need not be the nvcc on PATH prepare asked: one engine.toml
    // lists for this card builds an engine that serves wrong numbers without an error
    const nvcc = await this.deps.gpu.toolkitCuda(
      await this.cmakeCache(buildTree, "CMAKE_CUDA_COMPILER"),
    );
    const miscompiled = miscompiles(this.engine, nvcc, cap);
    if (miscompiled)
      return fail(ExitCode.Failure, `${miscompiled} (the compiler cmake configured)`);

    // A power cut mid-compile leaves an output whose data never reached the disk as a 0-byte file
    // with a fresh mtime, and ninja trusts it; deleted, it is rebuilt (2026-09-23, pq2_0's MMQ
    // instance). Objects, archives and shared libraries are ninja outputs that are never empty.
    const pruned = await this.deps.shell.run(pruneArgv(buildTree), { timeoutMs: 60_000 });
    if (pruned.code !== 0) {
      return fail(
        ExitCode.Failure,
        `pruning empty build outputs from ${buildTree} failed: ${pruned.stderr.trim()}`,
      );
    }
    const empties = pruned.stdout.split("\n").filter(Boolean);
    if (empties.length > 0) {
      this.deps.log.info(`pruned ${empties.length} empty build output(s) from ${buildTree}`);
    }

    const gate = (await this.deps.shell.which("buildgate")) ? "buildgate" : null;
    const jobs = options.jobs ?? this.deps.host.cpuCount() + 2;
    const build = await this.deps.shell.run(buildArgv({ buildTree, jobs, gate }), {
      timeoutMs: BUILD_TIMEOUT_MS,
    });
    await this.keepLog(`build-${tag}.log`, build.stdout + build.stderr);
    if (build.code !== 0)
      return fail(ExitCode.Failure, `build failed (local/logs/build-${tag}.log)`);

    const native = portable ? "off" : "on";
    const bundle = portable ? await this.portableRuntime(buildTree) : ok([]);
    if (!bundle.ok) return bundle;
    const provenance = { cap, native, source: "compiled" } as const;
    // the stub of libcuda cmake linked the engine against, for a machine with no driver to check it on
    const driverStub = await this.cmakeCache(buildTree, "CUDA_cuda_driver_LIBRARY");
    return this.publisher.publishOutput(buildTree, dir, provenance, bundle.value, driverStub);
  }

  /** where the compiler cmake configured with keeps each PORTABLE_RUNTIME library */
  private async portableRuntime(buildTree: string): Promise<Result<string[]>> {
    const cxx = (await this.cmakeCache(buildTree, "CMAKE_CXX_COMPILER")) ?? "c++";
    const files: string[] = [];
    for (const lib of PORTABLE_RUNTIME) {
      const found = await this.deps.shell.run([cxx, `-print-file-name=${lib}`], {
        timeoutMs: 60_000,
      });
      // the name comes back unchanged when the compiler has no such file
      const path = found.stdout.trim();
      if (found.code !== 0 || !path.startsWith("/") || !(await this.deps.fs.exists(path)))
        return fail(
          ExitCode.Failure,
          `a portable build ships ${lib}, and ${cxx} does not know where it is (${path || found.stderr.trim()})`,
        );
      files.push(path);
    }
    return ok(files);
  }

  /** an entry of the tree's CMakeCache.txt, undefined when the cache or the entry is missing */
  private async cmakeCache(buildTree: string, key: string): Promise<string | undefined> {
    const cache = join(buildTree, "CMakeCache.txt");
    if (!(await this.deps.fs.exists(cache))) return undefined;
    return new RegExp(`^${key}:\\w+=(.+)$`, "m").exec(await this.deps.fs.readText(cache))?.[1];
  }

  /** a build tree is reused only when CMakeCache.txt names this source and this directory (one moved from
   *  elsewhere, a copied checkout or a renamed build dir, refuses to configure) and was configured with the prebuilt
   *  UI off (one that downloaded it keeps tools/ui/dist, which an off build then embeds as stale assets) */
  private async cacheMatches(buildTree: string, source: string): Promise<boolean> {
    const cache = join(buildTree, "CMakeCache.txt");
    if (!(await this.deps.fs.exists(cache))) return true;
    const text = await this.deps.fs.readText(cache);
    const home = /^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$/m.exec(text)?.[1];
    const cacheDir = /^CMAKE_CACHEFILE_DIR:INTERNAL=(.*)$/m.exec(text)?.[1];
    // a source that is gone is not this one
    const homeReal =
      home && (await this.deps.fs.exists(home)) ? await this.deps.fs.realpath(home) : home;
    const same = homeReal === (await this.deps.fs.realpath(source)) && cacheDir === buildTree;
    if (!same) {
      this.deps.log.info(
        `discarding ${buildTree}: its cache was generated for ${home} in ${cacheDir}`,
      );
      return false;
    }
    const uiOff = /^LLAMA_USE_PREBUILT_UI:[A-Z]+=OFF$/m.test(text);
    if (!uiOff) {
      this.deps.log.info(
        `discarding ${buildTree}: it was configured with the prebuilt web UI on, and the engine embeds none`,
      );
    }
    return uiOff;
  }

  private keepLog(name: string, text: string): Promise<void> {
    return this.deps.fs.writeText(join(this.layout.logsDir, name), text);
  }
}
