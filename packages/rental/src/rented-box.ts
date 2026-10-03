// The rented box as rig sees it over ssh: the same layout as here under remote_dir, the
// compiled rig shipped in a payload, and the head brought up with rig's own commands. Every
// remote command line lives in this file (the idle sampler's shell text, in box-sampler.ts); the service asks for what it
// wants done.
import { basename, dirname } from "node:path";
import type { RunResult, Ssh, SshTarget } from "@rig/core";
import { type Layout, layoutAt } from "@rig/core";
import { SAMPLER_PATHS, samplerScript, windowRead } from "./box-sampler.ts";

/** where a template box's on-start logs (box-template.ts) */
const TEMPLATE_LOGS = "/var/log/rig";

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

  /** null when ssh answers (vast installs sshd after the container starts), else the last thing ssh said, so a
   *  box that refuses the key reads differently from one that has not booted */
  async unreachable(timeoutMs: number): Promise<string | null> {
    const probe = await this.ssh.run(this.target, "true", { timeoutMs });
    if (probe.code === 0) return null;
    return probe.stderr.trim().split("\n").at(-1)?.slice(0, 160) || `exit ${probe.code}`;
  }

  /** The card on the box: its utilization now, its peak over the last `windowSeconds` as a sampler on the box saw it, and
   *  what the box received over the same window.
   *  A utilization is a point sample, and work that runs between two reads is invisible to a read: 2026-10-02 a box in use
   *  by runs of a few minutes was destroyed after six reads of under 10 %. So the box keeps a sampler, one line every 5
   *  seconds into a file, and each read asks for the peak over the window since the last read. The read starts the
   *  sampler when it is not running (a rebooted box has none), by a pid file, writing the script beside it by rename so a
   *  running copy is never rewritten in place; a sampler of the first generation, which wrote no download column, is
   *  retired by its own pid file. `now` is the busiest card's percent; null when nvidia-smi gave no number
   *  (and "unreachable" when ssh itself did not answer: the command always exits 0 once a shell ran it, so any other exit
   *  is the transport). `window` is null when the sampler has no sample in the window yet, `downloadKBps` the average
   *  received over the window, null when no sample in it carries the column. */
  async cardUtilization(
    timeoutMs: number,
    windowSeconds: number,
  ): Promise<
    { now: number; window: number | null; downloadKBps: number | null } | "unreachable" | null
  > {
    const probe = await this.ssh.run(
      this.target,
      [
        'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null; echo "rc=$?"',
        `S=${SAMPLER_PATHS.script}; P=${SAMPLER_PATHS.pid}; F=${SAMPLER_PATHS.firstPid}`,
        `[ -f $F ] && { kill "$(cat $F)" 2>/dev/null; rm -f $F; }`,
        `cat > $S.new <<'RIG_SAMPLER'`,
        ...samplerScript(),
        "RIG_SAMPLER",
        "mv $S.new $S",
        'kill -0 "$(cat $P 2>/dev/null)" 2>/dev/null || { nohup setsid sh $S >/dev/null 2>&1 < /dev/null & } >/dev/null 2>&1',
        windowRead(SAMPLER_PATHS.samples, windowSeconds),
        "exit 0",
      ].join("\n"),
      { timeoutMs },
    );
    if (probe.code !== 0) return "unreachable";
    if (!/^rc=0$/m.test(probe.stdout)) return null;
    const percents = probe.stdout
      .split("\n")
      .filter((line) => /^\d+$/.test(line.trim()))
      .map(Number);
    if (percents.length === 0) return null;
    const window = /^window=(\d+)$/m.exec(probe.stdout)?.[1];
    const download = /^download=(\d+)$/m.exec(probe.stdout)?.[1];
    return {
      now: Math.max(...percents),
      window: window === undefined ? null : Number(window),
      downloadKBps: download === undefined ? null : Number(download),
    };
  }

  /** `rig <args>` on the box, the compiled binary from the payload */
  rig(args: string): Promise<RunResult> {
    return this.ssh.run(this.target, `${this.rigBinary} ${args}`);
  }

  /** the box's free bytes where packs land, and the bytes of this head's pack already there, so a
   *  fetch resumed on a partly filled box is measured against what is left to fetch. The disk a box
   *  is sold is not a promise: 53786017 was created for 160 GB and ran out of space 91 GB into a
   *  134 GB pack, so the box is asked rather than trusted. A free of 0 means it did not answer. */
  async packSpace(head: string): Promise<{ free: number; present: number }> {
    const probe = await this.ssh.run(
      this.target,
      `df -B1 --output=avail ${this.remoteDir} | tail -1; du -sb ${this.layout.packs(head)} 2>/dev/null | cut -f1`,
    );
    const [free, present] = probe.stdout.trim().split(/\s+/);
    return { free: Number(free) || 0, present: Number(present) || 0 };
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
    const serve = `nohup ${this.rigBinary} serve ${head} --gpu auto`; // every card: the head's profile takes what it needs
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

  /** a template box bringing its head up on its own (box-template.ts's on-start, which logs under /var/log/rig): whether
   *  its supervisor gave up, and the last line rig wrote */
  async templateBoot(): Promise<{ failed: boolean; last: string }> {
    const read = await this.ssh.run(
      this.target,
      `test -e ${TEMPLATE_LOGS}/FAILED && echo FAILED; tail -n 1 ${TEMPLATE_LOGS}/up.log 2>/dev/null`,
      { timeoutMs: 30_000 },
    );
    const lines = read.stdout.trimEnd().split("\n");
    const failed = lines[0] === "FAILED";
    return { failed, last: (failed ? lines.slice(1) : lines).join("\n").trim() };
  }

  /** a template box's rig log (up.log, guard.log) copied here */
  pullTemplateLog(name: string, localPath: string): Promise<void> {
    return this.ssh.pull(this.target, `${TEMPLATE_LOGS}/${name}`, localPath);
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
