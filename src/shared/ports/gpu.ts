// Gpu: one seam between rig and the machine. A port names a capability, never a tool.
export interface GpuInfo {
  index: number;
  name: string;
  memoryMiB: number;
  /** MiB in use on the card now, by every process: a desktop's compositor and browser included */
  usedMiB: number;
  computeCap: string;
  driver: string;
}
export interface Gpu {
  query(index: number): Promise<GpuInfo | null>;
  /** MiB the process `pid` holds on the card at `index`, 0 when it holds none */
  processMiB(index: number, pid: number): Promise<number>;
  /** the CUDA major.minor the driver can run, from the driver, not the toolkit */
  driverCuda(): Promise<string | null>;
  /** the version of the CUDA compiler at `compiler` (nvcc on PATH when omitted) as --version
   *  prints it, major.minor.build like 13.2.78, or null when there is none */
  toolkitCuda(compiler?: string): Promise<string | null>;
}
