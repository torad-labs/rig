// `rig build --prebuilt`: the pin's prebuilt engine, built in tools/prebuilt/Dockerfile's image so its floor is that
// image's (Ubuntu 22.04, glibc 2.35) and not whatever the building machine runs. `rig build --portable` runs inside
// the container as the calling user, with its own local/ (local/prebuilt/, so no build tree mixes with the host's),
// capped at 14 GiB and 6 CPUs: the engine's largest CUDA objects need ~13 GiB to compile. The container gets no card:
// the host's nvidia-smi gives the card's compute capability, handed in as --cap, because a compile never uses the card
// and so must not hold one (a lease over a build kept a card idle for every build). Its symbol check finds no
// libcuda.so.1 there and uses the toolkit's stub (build-publisher.ts). --sha builds a fork commit other than the pin (a rental image
// ahead of it) through a copy of engine.toml naming that commit and no published prebuilt; the pin is unchanged.
// The tarball it reports is what `rig e2e --prebuilt` gates before it is published.
import { join } from "node:path";
import type { Containers, FileSystem, Gpu, Hasher, Layout, Log, Shell } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import { type Engine, engineTarballName } from "../engine.ts";

export interface BuildPrebuiltOptions {
  gpu: number;
  jobs: number;
  /** a full 40-hex fork commit to build instead of the pin */
  sha?: string | undefined;
}

export interface BuildPrebuiltReport {
  image: string;
  /** the fork commit built */
  sha: string;
  tarball: string;
}

export interface BuildPrebuiltDeps {
  shell: Shell;
  fs: FileSystem;
  hasher: Hasher;
  gpu: Gpu;
  containers: Containers;
  log: Log;
}

/** the release floor's container: 14 GiB with no swap past it, 6 CPUs */
const LIMITS = { memory: "14g", cpus: 6 } as const;
const BUILD_TIMEOUT_MS = 4 * 3_600_000;
const CLI_TIMEOUT_MS = 600_000;

export class BuildPrebuilt {
  constructor(
    private readonly deps: BuildPrebuiltDeps,
    private readonly layout: Layout,
    private readonly engine: Engine,
  ) {}

  async run(options: BuildPrebuiltOptions): Promise<Result<BuildPrebuiltReport>> {
    const { shell, fs, hasher, gpu, containers, log } = this.deps;
    if (options.sha !== undefined && !/^[0-9a-f]{40}$/.test(options.sha))
      return fail(ExitCode.Usage, "--sha takes a full 40-hex fork commit");
    const card = await gpu.query(options.gpu);
    if (!card) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${options.gpu}`);

    const context = join(this.layout.root, "tools", "prebuilt");
    const image = `rig-prebuilt:${(await hasher.sha256File(join(context, "Dockerfile"))).slice(0, 12)}`;
    await fs.mkdirp(this.layout.logsDir);
    const imageLog = join(this.layout.logsDir, "prebuilt-image.log");
    const built = await containers.build(context, image, {}, imageLog);
    if (built.code !== 0)
      return fail(ExitCode.Failure, `docker build ${image} failed (${imageLog})`);
    // the binary the container runs, compiled from this checkout
    const cli = await shell.run(["bun", "run", "build"], {
      cwd: this.layout.root,
      timeoutMs: CLI_TIMEOUT_MS,
    });
    if (cli.code !== 0) return fail(ExitCode.Failure, `bun run build failed: ${tail(cli.stderr)}`);

    const out = join(this.layout.localDir, "prebuilt");
    await fs.mkdirp(out);
    const mounts: Record<string, string> = { [this.layout.root]: "/rig" };
    const sha = options.sha ?? this.engine.fork.sha;
    if (options.sha !== undefined) {
      const toml = pinFork(await fs.readText(join(this.layout.engineDir, "engine.toml")), sha);
      if (!toml) return fail(ExitCode.Failure, "engine.toml has no [fork] sha line to replace");
      // one copy per commit: a single shared file is rewritten in place under the container that has it mounted, so
      // two builds of different commits at once would each read the other's pin
      const pinned = join(out, `engine-${sha.slice(0, 7)}.toml`);
      await fs.writeText(pinned, toml);
      mounts[pinned] = "/rig/engine/engine.toml";
    }
    log.info(`building torad-labs/llama.cpp @ ${sha.slice(0, 7)} with --portable in ${image}`);
    const run = await containers.run(
      image,
      [
        "/rig/dist/rig",
        "build",
        "--cap",
        card.computeCap,
        "--portable",
        "--jobs",
        `${options.jobs}`,
      ],
      {
        asCaller: true,
        limits: LIMITS,
        env: { HOME: "/tmp", RIG_ROOT: "/rig" },
        mounts,
        writable: { [out]: "/rig/local" },
        timeoutMs: BUILD_TIMEOUT_MS,
      },
    );
    const runLog = join(this.layout.logsDir, `prebuilt-${sha.slice(0, 7)}.log`);
    await fs.writeText(runLog, `${run.stdout}\n${run.stderr}`);
    if (run.code !== 0)
      return fail(
        ExitCode.Failure,
        `the build in ${image} failed (exit ${run.code}, ${runLog}): ${tail(run.stderr)}`,
      );
    const tarball = join(
      this.layout.releaseBuildsDir,
      engineTarballName(sha.slice(0, 7), card.computeCap),
    );
    if (!(await fs.exists(tarball)))
      return fail(ExitCode.Failure, `the build in ${image} left no ${tarball} (${runLog})`);
    return ok({ image, sha, tarball });
  }
}

/** engine.toml with its [fork] commit replaced by `sha` and the pin's [[prebuilt]] entries dropped (they name the
 *  pin's tarballs, which loadEngine refuses for another commit); null when it has no sha line to replace */
function pinFork(toml: string, sha: string): string | null {
  let skip = false;
  const lines: string[] = [];
  for (const line of toml.replace(/\n$/, "").split("\n")) {
    if (line.startsWith("[")) skip = line === "[[prebuilt]]";
    if (!skip) lines.push(line.replace(/^sha = "[0-9a-f]{40}"/, `sha = "${sha}"`));
  }
  // every kept line ends in a newline, the last one too, whichever section the file ends in
  if (!lines.some((line) => line.startsWith(`sha = "${sha}"`))) return null;
  return lines.map((line) => `${line}\n`).join("");
}

const tail = (text: string) => text.trim().split("\n").slice(-5).join("\n");
