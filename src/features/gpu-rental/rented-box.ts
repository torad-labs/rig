// The rented box as rig sees it over ssh: the same layout as here under remote_dir, the
// compiled rig shipped in a payload, and the head brought up with rig's own commands. Every
// remote command line lives in this file; the service asks for what it wants done.
import { basename, dirname } from "node:path";
import { type Layout, layoutAt } from "../../shared/layout.ts";
import type { RunResult, Ssh, SshTarget } from "../../shared/ports/index.ts";

export class RentedBox {
  readonly layout: Layout;
  readonly rigBinary: string;

  constructor(
    private readonly ssh: Ssh,
    private readonly target: SshTarget,
    readonly remoteDir: string,
  ) {
    this.layout = layoutAt(remoteDir);
    this.rigBinary = this.layout.binary;
  }

  /** ssh answers at all: vast installs sshd after the container starts */
  async reachable(timeoutMs: number): Promise<boolean> {
    const probe = await this.ssh.run(this.target, "true", { timeoutMs });
    return probe.code === 0;
  }

  /** `rig <args>` on the box, the compiled binary from the payload */
  rig(args: string): Promise<RunResult> {
    return this.ssh.run(this.target, `${this.rigBinary} ${args}`);
  }

  /** the payload tarball (dist/rig, the head, the engine pin) unpacked under remote_dir */
  async receivePayload(localTarball: string): Promise<void> {
    const remoteTarball = `${this.remoteDir}/payload.tar.gz`;
    await this.ssh.run(
      this.target,
      `mkdir -p ${this.layout.engineBuildsDir} ${this.layout.logsDir}`,
    );
    await this.ssh.push(this.target, localTarball, remoteTarball);
    await this.ssh.run(
      this.target,
      `tar -C ${this.remoteDir} -xzf ${remoteTarball} && rm ${remoteTarball} && chmod +x ${this.rigBinary}`,
    );
  }

  /** where a build tarball of this name lives on the box */
  buildTarball(name: string): string {
    return `${this.layout.engineBuildsDir}/${name}`;
  }

  receiveBuild(localTarball: string, name: string): Promise<void> {
    return this.ssh.push(this.target, localTarball, this.buildTarball(name));
  }

  sendBuild(name: string, localTarball: string): Promise<void> {
    return this.ssh.pull(this.target, this.buildTarball(name), localTarball);
  }

  /** `rig serve` detached: the braces and the group's own redirection are the whole point.
   *  `cmd >log & echo $!` redirects cmd but leaves the SUBSHELL that `&` creates holding ssh's
   *  stdout and stderr, and that subshell waits on a server that never exits, so ssh never sees
   *  EOF and the bring-up hangs before the tunnel is installed. Grouping and redirecting the
   *  group also makes $! the nohup'd server rather than the subshell, which is what stopServer
   *  kills. Measured on box 51727683 (2026-09-20). */
  async startServer(head: string): Promise<void> {
    const serve = `nohup ${this.rigBinary} serve ${head} --gpu 0`;
    const log = `${this.layout.logsDir}/server.log`;
    await this.ssh.run(
      this.target,
      `cd ${this.remoteDir} && { ${serve} >> ${log} 2>&1 < /dev/null & echo $! > ${this.layout.pidFile} ; } > /dev/null 2>&1`,
    );
  }

  async stopServer(): Promise<void> {
    await this.ssh.run(
      this.target,
      `kill -INT $(cat ${this.layout.pidFile} 2>/dev/null) 2>/dev/null || true; sleep 3`,
    );
  }

  async serverLogTail(lines: number): Promise<string> {
    const tail = await this.ssh.run(
      this.target,
      `tail -${lines} ${this.layout.logsDir}/server.log`,
    );
    return tail.stdout;
  }

  /** a gate run's directory on the box, as a tarball here */
  async sendGateRun(remoteRunDir: string, localTarball: string): Promise<void> {
    const remoteTarball = `${this.remoteDir}/gate-run.tar.gz`;
    await this.ssh.run(
      this.target,
      `cd ${this.remoteDir} && tar -czf ${remoteTarball} -C ${dirname(remoteRunDir)} ${basename(remoteRunDir)}`,
    );
    await this.ssh.pull(this.target, remoteTarball, localTarball);
  }
}
