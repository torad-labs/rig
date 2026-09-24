// Host: one seam between rig and the machine. A port names a capability, never a tool.
export interface Host {
  hostname(): string;
  cpuCount(): number;
  /** the RAM this process may use: the cgroup limit when there is one, else the machine's */
  ramMiB(): Promise<number>;
  /** the pid holding the TCP listener on port; null when none is found or it cannot be read */
  listeningPid(port: number): Promise<number | null>;
}
