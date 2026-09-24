// Log: one seam between rig and the machine. A port names a capability, never a tool.
export interface Log {
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
}
