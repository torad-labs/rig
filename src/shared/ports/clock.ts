// Clock: one seam between rig and the machine. A port names a capability, never a tool.
export interface Clock {
  now(): number;
  sleep(ms: number): Promise<void>;
}
