// ssh/scp with a per-box known_hosts file: every rented box has a fresh host key on a fresh
// host:port, so accept-new against a file that starts empty per box is the right strictness.
import type { RunResult, Shell, Ssh, SshTarget } from "../ports/index.ts";

const sshOptions = (target: SshTarget) => [
  "-o",
  "BatchMode=yes",
  "-o",
  "ConnectTimeout=10",
  "-o",
  "StrictHostKeyChecking=accept-new",
  "-o",
  `UserKnownHostsFile=${target.knownHosts}`,
  "-o",
  "ServerAliveInterval=15",
  "-o",
  "ServerAliveCountMax=3",
  "-o",
  "LogLevel=ERROR",
];

export class OpenSsh implements Ssh {
  constructor(private readonly shell: Shell) {}
  run(
    target: SshTarget,
    cmd: string,
    options: { timeoutMs?: number; stdin?: string } = {},
  ): Promise<RunResult> {
    return this.shell.run(
      [
        "ssh",
        ...sshOptions(target),
        "-p",
        String(target.port),
        `${target.user}@${target.host}`,
        cmd,
      ],
      {
        timeoutMs: options.timeoutMs ?? 3_600_000,
        ...(options.stdin !== undefined ? { stdin: options.stdin } : {}),
      },
    );
  }
  async push(target: SshTarget, local: string, remote: string) {
    const copied = await this.shell.run(
      [
        "scp",
        ...sshOptions(target),
        "-P",
        String(target.port),
        local,
        `${target.user}@${target.host}:${remote}`,
      ],
      { timeoutMs: 3_600_000 },
    );
    if (copied.code !== 0) throw new Error(`scp ${local} -> ${remote}: ${copied.stderr.trim()}`);
  }
  async pull(target: SshTarget, remote: string, local: string) {
    const copied = await this.shell.run(
      [
        "scp",
        ...sshOptions(target),
        "-P",
        String(target.port),
        `${target.user}@${target.host}:${remote}`,
        local,
      ],
      { timeoutMs: 3_600_000 },
    );
    if (copied.code !== 0) throw new Error(`scp ${remote} -> ${local}: ${copied.stderr.trim()}`);
  }
}
