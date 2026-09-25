// A serving head as rig sees it over plain http: llama-server on a port, here or through a
// tunnel. Answers at all (a head is there, even one still loading), healthy, what it serves,
// slots in flight, and the token counters. Probing a head's answers is head-gating's own client.
import type { Clock, Http } from "../ports/index.ts";

export interface Serving {
  model: string;
  slots: number;
}

/** the counters an idle check compares between runs */
export interface Activity {
  /** the token totals as one string, "unreachable" when the server does not answer */
  key: string;
  /** requests in flight right now */
  busy: number;
}

const QUICK_MS = 3000;

/** what a probe of the port found: a server (any http answer), nothing listening, or no way to tell
 *  (a timeout, a reset): "unknown" is never treated as idle by anything that restarts */
export type Presence = "server" | "none" | "unknown";

/** the one error that proves nothing listens: a refused connection, named by the Http port */
function refused(error: unknown): boolean {
  return (error as { name?: unknown } | null)?.name === "ConnectionRefused";
}

export class HeadEndpoint {
  constructor(
    private readonly http: Http,
    private readonly clock: Clock,
    readonly url: string,
  ) {}

  /** any http answer, a 503 "loading" included, is a server; a refused connection is none */
  async presence(): Promise<Presence> {
    try {
      await this.http.request("GET", `${this.url}/health`, { timeoutMs: QUICK_MS });
      return "server";
    } catch (error) {
      return refused(error) ? "none" : "unknown";
    }
  }

  async healthy(): Promise<boolean> {
    try {
      const reply = await this.http.request("GET", `${this.url}/health`, { timeoutMs: QUICK_MS });
      return reply.status === 200;
    } catch {
      return false;
    }
  }

  /** poll /health until 200; `alive` (the unit is still active) ends the wait early with "dead"
   *  instead of burning the timeout on a process systemd has already given up on */
  async waitHealthy(
    timeoutMs: number,
    everyMs = 5000,
    alive?: () => Promise<boolean>,
  ): Promise<"healthy" | "timeout" | "dead"> {
    const deadline = this.clock.now() + timeoutMs;
    while (!(await this.healthy())) {
      if (alive && !(await alive())) return "dead";
      if (this.clock.now() >= deadline) return "timeout";
      await this.clock.sleep(everyMs);
    }
    return "healthy";
  }

  async serving(): Promise<Serving> {
    try {
      const reply = await this.http.request("GET", `${this.url}/props`, { timeoutMs: QUICK_MS });
      const props = JSON.parse(reply.text) as { model_path?: string; total_slots?: number };
      const model = (props.model_path ?? "?").split("/").at(-1) ?? "?";
      return { model, slots: props.total_slots ?? 0 };
    } catch {
      return { model: "?", slots: 0 };
    }
  }

  /** requests the head is working on or holding: slots processing plus requests deferred behind
   *  them (/metrics); null when either cannot be read — a guard that would restart on null is blind,
   *  so null is never idle */
  async inFlight(): Promise<number | null> {
    try {
      const slots = await this.http.request("GET", `${this.url}/slots`, { timeoutMs: QUICK_MS });
      const parsed: unknown = JSON.parse(slots.text);
      if (!Array.isArray(parsed)) return null; // --no-slots answers with an error object
      const processing = (parsed as Array<{ is_processing?: boolean }>).filter(
        (slot) => slot.is_processing,
      ).length;
      const metrics = await this.http.request("GET", `${this.url}/metrics`, {
        timeoutMs: QUICK_MS,
      });
      const deferred = /^llamacpp:requests_deferred (\d+)/m.exec(metrics.text);
      if (!deferred) return null;
      return processing + Number(deferred[1]);
    } catch {
      return null;
    }
  }

  async activity(): Promise<Activity> {
    try {
      const reply = await this.http.request("GET", `${this.url}/metrics`, { timeoutMs: 5000 });
      return parseActivity(reply.text);
    } catch {
      return { key: "unreachable", busy: 0 };
    }
  }
}

/** llama-server's prometheus text: the token totals and the requests in flight */
export function parseActivity(metrics: string): Activity {
  const totals = /^llamacpp:(prompt_tokens_total|tokens_predicted_total) (\S+)/gm;
  const key =
    [...metrics.matchAll(totals)].map((match) => `${match[1]}=${match[2]}`).join(";") ||
    "no-counters";
  const busy = Number(/^llamacpp:requests_processing (\d+)/m.exec(metrics)?.[1] ?? 0);
  return { key, busy };
}
