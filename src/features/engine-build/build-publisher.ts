// Publishing a build is ONE RENAME: the outputs sit in <dir>.tmp-<pid>/, the BUILD marker is
// written last, and the directory is renamed into place. llama-server itself is a 6.7 KB
// launcher over a 7 MB impl and a 71 MB libggml-cuda.so, so a copy cut short by Ctrl-C or the
// OOM killer would otherwise leave a directory that "exists" and cannot run. A cached tarball is
// published the same way; its name carries its identity and anything else is refused unopened.
// A prebuilt one arrives with NVIDIA's runtime archives, whose libraries are unpacked beside the
// binaries (their $ORIGIN runpath finds them) before the symbol check runs over all of it.
import { basename, join } from "node:path";
import {
  BUILD_MARKER,
  type Engine,
  engineTarballName,
  TARGETS,
} from "../../shared/engine/engine.ts";
import type { Clock, FileSystem, Host, Shell } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";

export interface PublisherDeps {
  fs: FileSystem;
  shell: Shell;
  host: Host;
  clock: Clock;
}

/** a downloaded runtime archive and the sonames to take from it */
export interface RuntimeArchive {
  path: string;
  libs: string[];
}

export interface Provenance {
  cap: string;
  /** "on" for -march=native, "off" for the portable baseline */
  native: "on" | "off";
  /** "compiled", or "tarball:<name>" */
  source: string;
}

export class BuildPublisher {
  constructor(
    private readonly deps: PublisherDeps,
    private readonly engine: Engine,
  ) {}

  /** the name a cached build of this engine for this card must carry */
  tarballName(cap: string): string {
    return engineTarballName(this.engine.sha7, cap);
  }

  /** a staging directory beside `dir`, emptied */
  async stage(dir: string): Promise<string> {
    const staging = `${dir}.tmp-${process.pid}`;
    await this.deps.fs.remove(staging);
    return staging;
  }

  /** the build tree's outputs, with each `bundle` file (a library the build links that the machine
   *  running it may not have) copied beside them under its own name, staged and published as `dir` */
  async publishOutput(
    buildTree: string,
    dir: string,
    provenance: Provenance,
    bundle: string[] = [],
  ): Promise<Result<string>> {
    const staging = await this.stage(dir);
    await this.deps.fs.copyTree(join(buildTree, "bin"), staging);
    for (const file of bundle) await this.deps.fs.copy(file, join(staging, basename(file)));
    return this.publish(staging, dir, provenance);
  }

  /** a cached tarball, checked by name, unpacked with the libraries of each `runtime` archive
   *  beside it, and published as `dir` */
  async publishTarball(
    tarball: string,
    cap: string,
    dir: string,
    runtime: RuntimeArchive[] = [],
  ): Promise<Result<string>> {
    if (!(await this.deps.fs.exists(tarball))) {
      return fail(ExitCode.Failure, `${tarball} does not exist`);
    }
    const expected = this.tarballName(cap);
    if (basename(tarball) !== expected) {
      const message = `${basename(tarball)} is not the tarball for sm_${cap} @ ${this.engine.sha7} (expected ${expected})`;
      return fail(ExitCode.Failure, message);
    }
    const staging = await this.stage(dir);
    await this.deps.fs.mkdirp(staging);
    const untar = await this.deps.shell.run(["tar", "-C", staging, "-xzf", tarball], {
      timeoutMs: 600_000,
    });
    if (untar.code !== 0) {
      await this.deps.fs.remove(staging);
      return fail(ExitCode.Failure, `tar failed: ${untar.stderr.trim()}`);
    }
    for (const archive of runtime) {
      const unpacked = await this.unpackRuntime(archive, staging);
      if (!unpacked.ok) {
        await this.deps.fs.remove(staging);
        return unpacked;
      }
    }
    const source = `tarball:${basename(tarball)}`;
    return this.publish(staging, dir, { cap, native: "off", source });
  }

  /** an NVIDIA redistributable archive (<name>-archive/lib/<soname>*) reduced to the libraries it
   *  names, unpacked flat into `staging`: the soname link and the file it resolves to */
  private async unpackRuntime(archive: RuntimeArchive, staging: string): Promise<Result<void>> {
    const members = archive.libs.map((lib) => `*/lib/${lib}*`);
    const untar = await this.deps.shell.run(
      [
        "tar",
        "-C",
        staging,
        "-xJf",
        archive.path,
        "--strip-components=2",
        "--wildcards",
        ...members,
      ],
      { timeoutMs: 600_000 },
    );
    if (untar.code !== 0) {
      return fail(
        ExitCode.Failure,
        `unpacking ${basename(archive.path)} failed: ${untar.stderr.trim()}`,
      );
    }
    for (const lib of archive.libs) {
      if (!(await this.deps.fs.exists(join(staging, lib)))) {
        return fail(ExitCode.Failure, `${basename(archive.path)} holds no lib/${lib}`);
      }
    }
    return ok(undefined);
  }

  /** the published directory as a tarball the next box of this arch can use, or the release a
   *  prebuilt is published from: without the marker (it names the host that built it; whoever
   *  unpacks the tarball writes their own) and without this machine's user and group */
  async pack(dir: string, into: string, cap: string): Promise<Result<string>> {
    const tarball = join(into, this.tarballName(cap));
    const exclude = `--exclude=./${BUILD_MARKER}`;
    const owner = ["--owner=0", "--group=0", "--numeric-owner"];
    const tar = await this.deps.shell.run(
      ["tar", "-C", dir, "-czf", tarball, exclude, ...owner, "."],
      {
        timeoutMs: 600_000,
      },
    );
    if (tar.code !== 0) return fail(ExitCode.Failure, `tar failed: ${tar.stderr.trim()}`);
    return ok(tarball);
  }

  /** every target present, the marker last, then one rename; the marker's text is returned */
  private async publish(
    staging: string,
    dir: string,
    provenance: Provenance,
  ): Promise<Result<string>> {
    for (const target of TARGETS) {
      if (!(await this.deps.fs.exists(join(staging, target)))) {
        await this.deps.fs.remove(staging);
        return fail(ExitCode.Failure, `${target} is missing from the build output`);
      }
    }
    const unresolved = await this.unresolvedSymbol(staging);
    if (unresolved) {
      await this.deps.fs.remove(staging);
      return fail(ExitCode.Failure, unresolved);
    }
    const markerPath = join(staging, BUILD_MARKER);
    if (!(await this.deps.fs.exists(markerPath))) {
      await this.deps.fs.writeText(markerPath, `${this.marker(provenance)}\n`);
    }
    await this.deps.fs.rename(staging, dir);
    return ok((await this.deps.fs.readText(join(dir, BUILD_MARKER))).trim());
  }

  /** every symbol a target and its libraries import resolves on this machine (`ldd -r` relocates
   *  without running main, so it needs no per-tool flag). A shared library links with undefined
   *  symbols, so an object left empty by a build cut short links into a directory that loads and
   *  dies at its first call: 2026-09-23 a power cut left a 0-byte mmq-instance-pq2_0.cu.o, and the
   *  published libggml-cuda.so had no mul_mat_q_case<PQ2_0> until a gate server hit it. The
   *  libraries are resolved from the staging directory the way the unit and the gate load them
   *  (LD_LIBRARY_PATH = the build dir): an inherited LD_LIBRARY_PATH outranks the $ORIGIN runpath
   *  and would check the libraries of whatever build it names instead of these. */
  private async unresolvedSymbol(staging: string): Promise<string | null> {
    for (const target of TARGETS) {
      const ldd = await this.deps.shell.run(["ldd", "-r", join(staging, target)], {
        timeoutMs: 60_000,
        env: { LD_LIBRARY_PATH: staging },
      });
      const output = `${ldd.stdout}\n${ldd.stderr}`;
      const missing = output.split("\n").find((line) => line.includes("undefined symbol"));
      if (ldd.code !== 0 || missing) {
        return `${target} does not resolve: ${(missing ?? ldd.stderr).trim()} (a stale or empty object in the build tree; the next build prunes empty ones)`;
      }
    }
    return null;
  }

  private marker(provenance: Provenance): string {
    const built = new Date(this.deps.clock.now()).toISOString();
    return `fork=${this.engine.fork.sha} cap=sm_${provenance.cap} native=${provenance.native} built=${built} host=${this.deps.host.hostname()} source=${provenance.source}`;
  }
}
