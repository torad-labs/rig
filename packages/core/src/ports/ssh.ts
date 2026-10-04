// Ssh: one seam between rig and the machine. A port names a capability, never a tool.
import type { RunResult } from "./shell.ts";
export interface SshTarget {
  host: string;
  port: number;
  user: string;
  knownHosts: string;
}
export interface SshResult extends RunResult {
  /** ssh refused the host: it answered with a key other than the one `knownHosts` holds for it */
  hostKeyChanged?: true;
}
export interface Ssh {
  run(
    target: SshTarget,
    cmd: string,
    opts?: { timeoutMs?: number; stdin?: string },
  ): Promise<SshResult>;
  /** stream a local file to a remote path */
  push(target: SshTarget, local: string, remote: string): Promise<void>;
  pull(target: SshTarget, remote: string, local: string): Promise<void>;
}
