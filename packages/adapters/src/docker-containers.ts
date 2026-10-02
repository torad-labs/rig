import type { ContainerRun, Containers, RunResult, Shell } from "@rig/core";

const BUILD_TIMEOUT_MS = 3_600_000;
const RUN_TIMEOUT_MS = 1_800_000;

export class DockerContainers implements Containers {
  constructor(private readonly shell: Shell) {}
  async build(context: string, tag: string, labels: Record<string, string>, logPath: string) {
    const label = Object.entries(labels).flatMap(([key, value]) => ["--label", `${key}=${value}`]);
    const built = await this.shell.run(
      ["docker", "build", "--progress=plain", "-t", tag, ...label, context],
      { timeoutMs: BUILD_TIMEOUT_MS },
    );
    await Bun.write(logPath, `${built.stdout}\n${built.stderr}`);
    return built;
  }
  run(image: string, cmd: readonly string[], options: ContainerRun = {}): Promise<RunResult> {
    const device = options.gpu === undefined ? [] : ["--device", `nvidia.com/gpu=${options.gpu}`];
    const mounts = Object.entries(options.mounts ?? {}).flatMap(([host, at]) => [
      "-v",
      `${host}:${at}:ro`,
    ]);
    return this.shell.run(["docker", "run", "--rm", ...device, ...mounts, image, ...cmd], {
      timeoutMs: RUN_TIMEOUT_MS,
    });
  }
  save(image: string, tarball: string) {
    return this.shell.run(["docker", "save", "-o", tarball, image], {
      timeoutMs: BUILD_TIMEOUT_MS,
    });
  }
}
