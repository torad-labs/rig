// `rig e2e`: the gate a prebuilt engine passes before it is published, on a machine with only the NVIDIA driver. A
// fresh ubuntu:22.04 container (the oldest base a prebuilt supports: glibc 2.35, libstdc++ 12; --base another image)
// gets one card through CDI (the driver's libcuda and nvidia-smi, nothing else from NVIDIA) and only curl. This
// checkout is packed as release.yml packs it, installed by install.sh, and then `rig prepare` must report a prebuilt
// with no toolkit, `rig build` must install the prebuilt and NVIDIA's pinned runtime and pass its ldd -r check, and
// the installed llama-bench must decode the pack on the card.
//
// --head NAME, with the pack that head's source pack: then `rig fetch NAME --from` adopts it and `rig derive NAME`
// must derive what a machine without Torad's private assets serves, fetching the head's public [derive] assets from
// their pinned URLs (a copy of the pack in the container, ~2x its size). --prebuilt installs a local tarball instead
// of the one engine.toml pins (the staged engine.toml points its [[prebuilt]] entry at the file, with the file's
// sha256): the check a build passes BEFORE it is uploaded and pinned. Without it, the pinned URL is fetched as a user
// would. A gate lock (RIG_GATE_LOCK) is held with flock around the decode only: a shared gate card.
import { basename, join } from "node:path";
import type { Containers, FileSystem, Hasher, Layout, Log, Shell } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";

export interface DriverOnlyGateOptions {
  /** a GGUF the installed llama-bench decodes; with `head`, that head's source pack */
  pack: string;
  /** a local engine tarball installed instead of the one engine.toml pins */
  prebuilt?: string | undefined;
  gpu: number;
  /** the container image the machine starts from */
  base: string;
  head?: string | undefined;
  /** a file flock holds around the decode (a shared gate card) */
  lock?: string | undefined;
}

export interface DriverOnlyGateReport {
  base: string;
  prebuilt: string | null;
  log: string;
}

export interface DriverOnlyGateDeps {
  shell: Shell;
  fs: FileSystem;
  hasher: Hasher;
  containers: Containers;
  log: Log;
}

/** what release.yml's checkout holds besides dist/: the files git tracks under these */
const RELEASED = ["heads", "LICENSE", "README.md", "engine/engine.toml"] as const;
const RELEASE = "rig-linux-x64.tar.gz";
const GATE_TIMEOUT_MS = 2 * 3_600_000;
const STEP_TIMEOUT_MS = 600_000;

/** the machine's checks, run by bash in the container: its argv, so no file of its own */
const CHECKS = String.raw`
set -euo pipefail
apt-get update -qq > /dev/null
apt-get install -y -qq curl ca-certificates > /dev/null   # what install.sh needs; rig adds the rest
echo "== the machine: no toolkit, no compiler"
grep PRETTY_NAME /etc/os-release; ldd --version | sed -n 1p # sed reads it all: head would SIGPIPE ldd under pipefail
for tool in nvcc cmake ninja git gcc; do
  if command -v $tool > /dev/null; then echo "UNEXPECTED: $tool is here"; exit 1; fi
done
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo "== install.sh"
RIG_RELEASE_URL=file:///rel sh /rel/install.sh
export PATH=$HOME/.local/bin:$PATH
echo "== rig prepare"
rig prepare --json | tee /tmp/prepare.json
grep -q "\"prebuilt\": true" /tmp/prepare.json || { echo "prepare: no prebuilt for this card"; exit 1; }
grep -q "\"toolkitCuda\": null" /tmp/prepare.json || { echo "prepare: a toolkit is on this machine"; exit 1; }
echo "== rig build"
rig build
dir=$(ls -d $HOME/.local/share/rig/local/engine-builds/*-sm*)
cat "$dir/BUILD"
echo "== the runtime resolves from the build directory"
ldd "$dir/libggml-cuda.so" | grep -E "cudart|cublas|gomp|libcuda\.so"
echo "== decode"
# CUDA that fails to initialize (a driver older than the runtime) leaves zero devices, and llama-bench
# then decodes on the CPU and exits 0: the card must be listed before a decode can pass
"$dir/llama-bench" --list-devices | tee /tmp/devices.txt
grep -q "^  CUDA0: " /tmp/devices.txt || { echo "decode: no CUDA device, so llama-bench would run on the CPU"; exit 1; }
flock /gate.lock "$dir/llama-bench" -m /pack.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 2
if [ -n "$HEAD_NAME" ]; then
  echo "== $HEAD_NAME: the source pack adopted, then derived as a machine without private assets derives it"
  rig fetch "$HEAD_NAME" --from /pack.gguf --json
  rig derive "$HEAD_NAME" --json | tee /tmp/derive.json
  grep -q "\"state\": \"derived\"" /tmp/derive.json || { echo "derive: nothing derived"; exit 1; }
  rig describe "$HEAD_NAME" | grep -E "\"(served_file|undrived)\""
fi
echo "== PASS"
`;

export class DriverOnlyGate {
  constructor(
    private readonly deps: DriverOnlyGateDeps,
    private readonly layout: Layout,
  ) {}

  async run(options: DriverOnlyGateOptions): Promise<Result<DriverOnlyGateReport>> {
    const { fs } = this.deps;
    for (const file of [options.pack, options.prebuilt])
      if (file !== undefined && !(await fs.exists(file)))
        return fail(ExitCode.Failure, `${file} does not exist`);
    const stage = join(this.layout.localDir, `e2e-driver-only-${process.pid}`);
    await fs.remove(stage);
    try {
      return await this.gate(options, stage);
    } finally {
      await fs.remove(stage);
    }
  }

  private async gate(
    options: DriverOnlyGateOptions,
    stage: string,
  ): Promise<Result<DriverOnlyGateReport>> {
    const { fs, hasher, containers, log } = this.deps;
    const { root } = this.layout;
    const pack = await fs.realpath(options.pack);
    const prebuilt = options.prebuilt === undefined ? null : await fs.realpath(options.prebuilt);

    // the release, as release.yml packs it: its checkout holds only the files git tracks and a dist/ with the one
    // binary it built. A copy of this checkout's directories would also carry what git ignores here: the private
    // refusal-ablation LoRA under heads/bonsai-2-27b/assets/lora, which `rig derive` then applies, and old
    // dist/rig.bak-* binaries.
    const checkout = join(stage, "pack", "rig");
    const release = join(stage, "release");
    await fs.mkdirp(checkout);
    await fs.mkdirp(release);
    const cli = await this.step(["bun", "run", "build"], root);
    if (!cli.ok) return cli;
    const tracked = await this.step(["git", "-C", root, "ls-files", "-z", "--", ...RELEASED]);
    if (!tracked.ok) return tracked;
    const list = join(stage, "files");
    await fs.writeText(list, `${tracked.value}dist/rig\0`);
    const files = join(stage, "files.tar");
    for (const argv of [
      ["tar", "-C", root, "--null", "-T", list, "-cf", files],
      ["tar", "-C", checkout, "-xf", files],
    ]) {
      const copied = await this.step(argv);
      if (!copied.ok) return copied;
    }

    const mounts: Record<string, string> = { [release]: "/rel", [pack]: "/pack.gguf" };
    if (prebuilt) {
      const name = basename(prebuilt);
      const toml = join(checkout, "engine", "engine.toml");
      const staged = stagePrebuilt(
        await fs.readText(toml),
        name,
        await hasher.sha256File(prebuilt),
      );
      if (!staged)
        return fail(
          ExitCode.Failure,
          `engine.toml pins no prebuilt named ${name} (the pin's own build is engine-sm<cap>-<sha7>.tar.gz)`,
        );
      await fs.writeText(toml, staged);
      mounts[prebuilt] = `/prebuilt/${name}`;
    }
    const tarball = join(release, RELEASE);
    const packed = await this.step([
      "tar",
      "-C",
      join(stage, "pack"),
      "--owner=0",
      "--group=0",
      "--numeric-owner",
      "-czf",
      tarball,
      "rig",
    ]);
    if (!packed.ok) return packed;
    await fs.writeText(`${tarball}.sha256`, `${await hasher.sha256File(tarball)}  ${RELEASE}\n`);
    await fs.copy(join(root, "install.sh"), join(release, "install.sh"));

    log.info(
      `the driver-only gate in ${options.base} on card ${options.gpu}${prebuilt ? ` with ${basename(prebuilt)}` : ""}`,
    );
    const run = await containers.run(options.base, ["bash", "-c", CHECKS], {
      gpu: options.gpu,
      env: { HEAD_NAME: options.head ?? "" },
      mounts,
      writable: { [options.lock ?? "/dev/null"]: "/gate.lock" },
      timeoutMs: GATE_TIMEOUT_MS,
    });
    await fs.mkdirp(this.layout.logsDir);
    const logPath = join(this.layout.logsDir, "e2e-driver-only.log");
    await fs.writeText(logPath, `${run.stdout}\n${run.stderr}`);
    if (run.code !== 0 || !/^== PASS$/m.test(run.stdout))
      return fail(
        ExitCode.Failure,
        `the driver-only gate failed (exit ${run.code}, ${logPath}): ${tail(`${run.stdout}\n${run.stderr}`)}`,
      );
    return ok({ base: options.base, prebuilt: prebuilt && basename(prebuilt), log: logPath });
  }

  /** a command that must succeed: its stdout, or a failure naming it */
  private async step(argv: string[], cwd?: string): Promise<Result<string>> {
    const run = await this.deps.shell.run(argv, {
      ...(cwd ? { cwd } : {}),
      timeoutMs: STEP_TIMEOUT_MS,
    });
    if (run.code !== 0)
      return fail(ExitCode.Failure, `${argv.slice(0, 3).join(" ")} failed: ${tail(run.stderr)}`);
    return ok(run.stdout);
  }
}

/** engine.toml with the [[prebuilt]] entry whose url names `name` pointed at the mounted copy, and the sha256 after
 *  it set to the file's; null when no entry's url names it */
function stagePrebuilt(toml: string, name: string, sha256: string): string | null {
  const url = new RegExp(`^url = .*/${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"$`);
  let found = false;
  let hit = false;
  const lines = toml.split("\n").map((line) => {
    if (url.test(line)) {
      found = hit = true;
      return `url = "file:///prebuilt/${name}"`;
    }
    if (hit && line.startsWith("sha256 = ")) {
      hit = false;
      return `sha256 = "${sha256}"`;
    }
    return line;
  });
  return found ? lines.join("\n") : null;
}

const tail = (text: string) => text.trim().split("\n").slice(-5).join("\n");
