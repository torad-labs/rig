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
    const user = options.asCaller ? ["--user", `${process.getuid?.()}:${process.getgid?.()}`] : [];
    const { memory, cpus } = options.limits ?? {};
    const limits = options.limits
      ? ["--memory", `${memory}`, "--memory-swap", `${memory}`, "--cpus", `${cpus}`]
      : [];
    const env = Object.entries(options.env ?? {}).flatMap(([key, value]) => [
      "-e",
      `${key}=${value}`,
    ]);
    const mounts = [
      ...Object.entries(options.mounts ?? {}).map(([host, at]) => `${host}:${at}:ro`),
      ...Object.entries(options.writable ?? {}).map(([host, at]) => `${host}:${at}`),
    ].flatMap((mount) => ["-v", mount]);
    return this.shell.run(
      ["docker", "run", "--rm", ...device, ...user, ...limits, ...env, ...mounts, image, ...cmd],
      { timeoutMs: options.timeoutMs ?? RUN_TIMEOUT_MS },
    );
  }
  save(image: string, tarball: string) {
    return this.shell.run(["docker", "save", "-o", tarball, image], {
      timeoutMs: BUILD_TIMEOUT_MS,
    });
  }
}
