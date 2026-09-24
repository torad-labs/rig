// A llama-server for one gate leg: the head's runtime args on the gate card and port, one pack,
// a context and slot count the probe chooses, stdout/stderr to the run directory. Stopped with
// SIGINT and waited for, so the next leg starts on a free card.
import { draftArgv } from "../../shared/head/draft-head.ts";
import type { Head } from "../../shared/head/head.ts";
import type { Clock, FileSystem, Http, Log, Process, Shell } from "../../shared/ports/index.ts";
import type { GateLegs, LegOptions } from "./gate-legs.ts";
import { LlamaClient } from "./llama-client.ts";

export type { LegOptions } from "./gate-legs.ts";

export interface GateServerDeps {
  shell: Shell;
  fs: FileSystem;
  http: Http;
  clock: Clock;
  log: Log;
}

const STARTUP_TIMEOUT_MS = 300_000;

export class GateServer implements GateLegs {
  private proc: Process | null = null;

  constructor(
    private readonly deps: GateServerDeps,
    private readonly head: Head,
    private readonly binDir: string,
    private readonly gpu: number,
    private readonly port: number,
    private readonly runDir: string,
  ) {}

  get client() {
    return new LlamaClient(this.deps.http, `http://127.0.0.1:${this.port}`);
  }

  argv(options: LegOptions): string[] {
    const asset = (arg: string) => (arg.startsWith("assets/") ? this.head.path(arg) : arg);
    const draft = options.draft ? draftArgv(this.head, { speculative: true }) : [];
    return [
      `${this.binDir}/llama-server`,
      "-m",
      options.pack,
      "-ngl",
      "99",
      "--jinja",
      "-fa",
      "on",
      ...this.head.runtime.args.map(asset),
      ...draft,
      "-c",
      String(options.ctx),
      "-np",
      String(options.slots),
      "--host",
      "127.0.0.1",
      "--port",
      String(this.port),
      ...(options.extra ?? []),
    ];
  }

  async start(options: LegOptions): Promise<void> {
    if (this.proc) throw new Error(`a gate server is already running (${this.proc.pid})`);
    if (options.draft) await this.requireDraft(options.label);

    const log = `${this.runDir}/server-${options.label}.log`;
    const proc = this.deps.shell.spawn(this.argv(options), {
      env: {
        CUDA_DEVICE_ORDER: "PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES: String(this.gpu),
        LD_LIBRARY_PATH: this.binDir,
      },
      stdoutPath: log,
      stderrPath: log,
    });
    this.proc = proc;
    await this.volunteerForOom(proc, options.label);

    const deadline = this.deps.clock.now() + STARTUP_TIMEOUT_MS;
    while (!(await this.client.healthy())) {
      if (this.deps.clock.now() >= deadline) {
        await this.stop();
        throw new Error(`gate server ${options.label} did not come up in 300 s — ${log}`);
      }
      await this.deps.clock.sleep(1000);
    }
    const pack = options.pack.split("/").at(-1);
    this.deps.log.info(
      `  server ${options.label}: ${pack} -c ${options.ctx} -np ${options.slots} on :${this.port} (GPU ${this.gpu})`,
    );
  }

  async stop(): Promise<void> {
    if (!this.proc) return;
    this.proc.kill("SIGINT");
    await this.proc.exited;
    this.proc = null;
    await this.deps.clock.sleep(1000);
  }

  /** run `fn` against a fresh server on `pack`, stopping it whatever happens */
  async leg<T>(options: LegOptions, fn: (client: LlamaClient) => Promise<T>): Promise<T> {
    await this.start(options);
    try {
      return await fn(this.client);
    } finally {
      await this.stop();
    }
  }

  /** the draft the leg loads: declared by the head, and on disk when it is a sidecar file
   *  (an in-pack head is inside the served pack the leg already verified) */
  private async requireDraft(label: string): Promise<void> {
    if (!this.head.speculative) {
      throw new Error(
        `the ${label} leg asks for the draft head, which is not declared by this head`,
      );
    }
    const path = this.head.draftPath;
    if (!path || (await this.deps.fs.exists(path))) return;
    throw new Error(
      `the ${label} leg asks for the draft head, which is missing at ${path} (run: rig fetch)`,
    );
  }

  /** A gate leg is the most disposable multi-GiB process on the box, and it is the one that
   *  creates the pressure: it volunteers as the memory killer's first victim so a serving head
   *  or a paid run is not taken instead (scar 2026-09-20 03:28:38: the live head, 13,379 MiB,
   *  SIGTERMed while a leg loaded a second pack). Raising oom_score_adj needs no privilege;
   *  losing a leg costs a re-run, and the run says which probe died. */
  private async volunteerForOom(proc: Process, label: string): Promise<void> {
    try {
      await this.deps.fs.writeText(`/proc/${proc.pid}/oom_score_adj`, "800\n");
    } catch {
      this.deps.log.warn(`could not raise oom_score_adj for the ${label} server (pid ${proc.pid})`);
    }
  }
}
