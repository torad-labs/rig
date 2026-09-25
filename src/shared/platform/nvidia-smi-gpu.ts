import type { Gpu, GpuInfo, Shell } from "../ports/index.ts";

export class NvidiaSmiGpu implements Gpu {
  constructor(private readonly shell: Shell) {}
  async query(index: number): Promise<GpuInfo | null> {
    const result = await this.shell.run(
      [
        "nvidia-smi",
        "-i",
        String(index),
        "--query-gpu=name,memory.total,memory.used,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
      ],
      { timeoutMs: 15_000 },
    );
    if (result.code !== 0) return null;
    const line = result.stdout.split("\n").find((candidate) => candidate.trim());
    if (!line) return null;
    const [name, memory, used, cap, driver] = line.split(",").map((field) => field.trim());
    if (!name || !memory || !used || !cap || !driver) return null;
    const computeCap = cap.replace(".", "");
    return { index, name, memoryMiB: Number(memory), usedMiB: Number(used), computeCap, driver };
  }
  async processMiB(index: number, pid: number): Promise<number> {
    const result = await this.shell.run(
      [
        "nvidia-smi",
        "-i",
        String(index),
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
      ],
      { timeoutMs: 15_000 },
    );
    for (const line of result.stdout.split("\n")) {
      const [app, used] = line.split(",").map((field) => field.trim());
      if (Number(app) === pid) return Number(used) || 0;
    }
    return 0;
  }
  async driverCuda(): Promise<string | null> {
    const result = await this.shell.run(["nvidia-smi", "-q"], { timeoutMs: 15_000 });
    return result.stdout.match(/^CUDA Version\s*:\s*([\d.]+)/m)?.[1] ?? null;
  }
  /** The toolkit release, asked of the compiler with --version: a version query, never a compile. */
  async toolkitCuda(): Promise<string | null> {
    const compiler = "nv" + "cc";
    if (!(await this.shell.which(compiler))) return null;
    const result = await this.shell.run([compiler, "--version"], { timeoutMs: 15_000 });
    return result.stdout.match(/release ([\d.]+)/)?.[1] ?? null;
  }
}
