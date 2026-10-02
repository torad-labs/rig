import type { Gpu, GpuInfo, Shell } from "@rig/core";

export class NvidiaSmiGpu implements Gpu {
  constructor(private readonly shell: Shell) {}
  async query(index: number): Promise<GpuInfo | null> {
    const [card] = await this.cards(["-i", String(index)]);
    return card ?? null;
  }
  async list(): Promise<GpuInfo[]> {
    return this.cards([]);
  }
  private async cards(select: string[]): Promise<GpuInfo[]> {
    const result = await this.shell.run(
      [
        "nvidia-smi",
        ...select,
        "--query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
      ],
      { timeoutMs: 15_000 },
    );
    if (result.code !== 0) return [];
    const cards: GpuInfo[] = [];
    for (const line of result.stdout.split("\n")) {
      if (!line.trim()) continue;
      const [index, name, memory, used, cap, driver] = line.split(",").map((field) => field.trim());
      if (!index || !name || !memory || !used || !cap || !driver) continue;
      const computeCap = cap.replace(".", "");
      cards.push({
        index: Number(index),
        name,
        memoryMiB: Number(memory),
        usedMiB: Number(used),
        computeCap,
        driver,
      });
    }
    return cards;
  }
  async utilization(): Promise<number[]> {
    const result = await this.shell.run(
      ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
      { timeoutMs: 15_000 },
    );
    if (result.code !== 0) return [];
    return result.stdout
      .split("\n")
      .filter((line) => line.trim())
      .map((line) => Number(line.trim()))
      .filter(Number.isFinite);
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
