// Shell: one seam between rig and the machine. A port names a capability, never a tool.
export interface RunResult {
  code: number;
  stdout: string;
  stderr: string;
}
export interface RunOptions {
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs?: number;
  stdin?: string;
}
export interface SpawnOptions {
  cwd?: string;
  env?: Record<string, string>;
  stdoutPath?: string;
  stderrPath?: string;
  detached?: boolean;
}
export interface Process {
  pid: number;
  kill(signal?: "SIGINT" | "SIGTERM" | "SIGKILL"): void;
  exited: Promise<number>;
}
export interface Shell {
  run(cmd: readonly string[], opts?: RunOptions): Promise<RunResult>;
  spawn(cmd: readonly string[], opts?: SpawnOptions): Process;
  which(name: string): Promise<string | null>;
}
