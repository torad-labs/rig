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
  /** The compiler's version, asked with --version: a version query, never a compile. Its build
   *  number tells CUDA 13.2.1's compiler (V13.2.78) from 13.2.2's (V13.2.86); "release 13.2" does not. */
  async toolkitCuda(compiler?: string): Promise<string | null> {
    const nvcc = compiler ?? "nv" + "cc";
    if (!compiler && !(await this.shell.which(nvcc))) return null;
    const result = await this.shell.run([nvcc, "--version"], { timeoutMs: 15_000 });
    if (result.code !== 0) return null;
    return (
      result.stdout.match(/, V(\d+\.\d+\.\d+)/)?.[1] ??
      result.stdout.match(/release (\d+\.\d+)/)?.[1] ??
      null
    );
  }
}
