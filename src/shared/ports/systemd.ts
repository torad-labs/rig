// Systemd: one seam between rig and the machine. A port names a capability, never a tool.
export interface Systemd {
  unitDir(): string;
  daemonReload(): Promise<void>;
  enable(unit: string): Promise<void>;
  disable(unit: string): Promise<void>;
  restart(unit: string): Promise<void>;
  stop(unit: string): Promise<void>;
  isActive(unit: string): Promise<boolean>;
  mainPid(unit: string): Promise<number | null>;
}
