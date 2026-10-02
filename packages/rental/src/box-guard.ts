// The idle rule on a box that was rented from a template, where no machine of ours holds a timer:
// `rig vast guard <head>`, started beside `rig up <head> --foreground` by the template's on-start,
// watches the box from inside and stops it through vast's API with the box's own key
// (CONTAINER_API_KEY, which can start, stop or destroy that one instance and nothing else) once
// nothing has happened for the idle budget. Something happening is any of: the server's token
// counters moving or a request in flight, the container's CPU busy (a bench, a gate someone runs
// over ssh, the pack being hashed), or the pack still being written under its directory (the first
// start's download is work, not idleness; a download that hangs stops being written and is idle).
// Not the card's utilization: on the template's first PRO 6000 box (2026-09-28) an empty card at
// 0 MiB with no process read 100 %, which would have kept that box billing for good. Written is read from sizes
// and modification times both: aria2c writes its segments into a sparse file whose size is the
// whole shard's from its first minutes, so a size alone goes still while the bytes still arrive
// (measured on the first rental from the template, 2026-09-28). Two more ends stop the box at once:
// the on-start's give-up file (`--stop-when`), which its supervisor writes when `rig up` failed every
// try, and a lifetime cap (`--max-hours`), whatever the readings say, so a reading that is wrong never
// bills for good. A stop keeps the disk, so the pack is there when the box is started again.
import { join } from "node:path";
import type { Clock, FileSystem, Host, Http, Log } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import type { Head } from "@rig/head";
import { HeadEndpoint } from "@rig/head";

/** the container's CPU over a tick, in cores, at which the box is in use whatever its server says: a bench launching
 *  kernels holds a core, an idle llama-server waits on its queue */
export const CPU_BUSY_CORES = 0.5;
const TICK_MS = 60_000;
export const VAST_API = "https://console.vast.ai/api/v0";

export interface GuardDeps {
  http: Http;
  fs: FileSystem;
  host: Host;
  clock: Clock;
  log: Log;
  /** the box's own identity, from the environment vast gives every instance */
  box: { id?: string | undefined; apiKey?: string | undefined };
}

export interface GuardOptions {
  idleMinutes: number;
  /** stop once the guard has run this long, whatever the readings say */
  maxHours?: number | undefined;
  /** stop at once when this file exists: the on-start's supervisor gave up on the head */
  stopWhen?: string | undefined;
}

export interface GuardReport {
  action: "stopped";
  why: "idle" | "gave up" | "lifetime";
  idleMinutes: number;
}

export class GuardBox {
  constructor(private readonly deps: GuardDeps) {}

  /** runs until the box has been idle for `idleMinutes`, its supervisor gave up or its lifetime ran out, then stops
   *  it; returns once vast took the stop */
  async run(head: Head, options: GuardOptions): Promise<Result<GuardReport>> {
    const { http, fs, host, clock, log, box } = this.deps;
    if (!box.id || !box.apiKey)
      return fail(
        ExitCode.Failure,
        "not on a vast box: CONTAINER_ID and CONTAINER_API_KEY are what vast gives an instance to stop itself with",
      );
    const endpoint = new HeadEndpoint(http, clock, `http://127.0.0.1:${head.port}`);
    const budget = options.idleMinutes * 60_000;
    const lifetime =
      options.maxHours === undefined ? "" : `, and after ${options.maxHours} h in any case`;
    log.info(
      `guarding box ${box.id}: stopped after ${options.idleMinutes} min with nothing served, computed or fetched${lifetime}`,
    );
    const start = clock.now();
    let last = "";
    let since = start;
    let cpu = { at: start, micros: await host.cpuMicros() };
    for (;;) {
      const activity = await endpoint.activity();
      const fetched = await treeState(fs, head.packsDir).catch(
        (error: Error) => `unreadable (${error.message})`,
      );
      const now = clock.now();
      const micros = await host.cpuMicros();
      const cores =
        micros !== null && cpu.micros !== null && now > cpu.at
          ? (micros - cpu.micros) / ((now - cpu.at) * 1000)
          : 0;
      cpu = { at: now, micros };
      const key = `${activity.key} packs=${fetched}`;
      const idle = Math.floor((now - since) / 60_000);
      const reading = `${activity.key}, CPU ${cores.toFixed(2)} cores`;
      let why: GuardReport["why"] | null = null;
      if (options.stopWhen && (await fs.exists(options.stopWhen))) {
        why = "gave up";
        log.info(
          `${options.stopWhen} is there, rig up gave up (${reading}): stopping box ${box.id}`,
        );
      } else if (options.maxHours !== undefined && now - start >= options.maxHours * 3_600_000) {
        why = "lifetime";
        log.info(`guarded for ${options.maxHours} h (${reading}): stopping box ${box.id}`);
      } else if (activity.busy > 0 || cores >= CPU_BUSY_CORES || key !== last) {
        last = key;
        since = now;
      } else if (now - since >= budget) {
        why = "idle";
        log.info(`idle for ${idle} min (${reading}): stopping box ${box.id}`);
      }
      if (why && (await this.stop(box.id, box.apiKey)))
        return ok({ action: "stopped", why, idleMinutes: Math.floor((now - since) / 60_000) });
      await clock.sleep(TICK_MS);
    }
  }

  /** whether vast took the stop; the box keeps billing until it does, so a refusal is said and asked again next tick */
  private async stop(id: string, apiKey: string): Promise<boolean> {
    const stopped = await this.deps.http
      .request("PUT", `${VAST_API}/instances/${id}/`, {
        body: { state: "stopped" },
        headers: { authorization: `Bearer ${apiKey}` },
        timeoutMs: 30_000,
      })
      .catch((error: Error) => ({ status: 0, text: error.message }));
    if (stopped.status === 200) return true;
    this.deps.log.error(
      `vast did not take the stop (HTTP ${stopped.status}): ${stopped.text.slice(0, 200)}`,
    );
    return false;
  }
}

/** every byte under `dir` and the newest write to any of it: a download's .part changes one or both */
async function treeState(fs: FileSystem, dir: string): Promise<string> {
  let bytes = 0;
  let newest = 0;
  const walk = async (at: string) => {
    for (const name of await fs.list(at)) {
      const path = join(at, name);
      const stat = await fs.stat(path);
      if (!stat) continue;
      newest = Math.max(newest, stat.mtimeMs);
      if (stat.isDirectory && !stat.isSymlink) await walk(path);
      else bytes += stat.size;
    }
  };
  if (await fs.exists(dir)) await walk(dir);
  return `${bytes}@${newest}`;
}
