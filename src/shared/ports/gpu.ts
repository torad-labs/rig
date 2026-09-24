// Gpu: one seam between rig and the machine. A port names a capability, never a tool.
export interface GpuInfo {
  index: number;
  name: string;
  memoryMiB: number;
  computeCap: string;
  driver: string;
}
export interface Gpu {
  query(index: number): Promise<GpuInfo | null>;
  /** the CUDA major.minor the driver can run, from the driver, not the toolkit */
  driverCuda(): Promise<string | null>;
  /** the CUDA toolkit release, from nvcc, or null when there is no toolkit */
  toolkitCuda(): Promise<string | null>;
}
