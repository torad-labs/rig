// A probe is a measurement with a verdict: `pass` is true or false by a criterion gates.toml
// states, `lines` are what a reader sees, `data` is what the run directory keeps. Probes on the
// gate card take the GateLegs and start their own legs; probes on the live head take a client.
import type { CacheFormats } from "../../../shared/head/cache-formats.ts";
import type { Head } from "../../../shared/head/head.ts";
import type { FileSystem, Log, Shell } from "../../../shared/ports/index.ts";
import type { GateLegs } from "../gate-legs.ts";
import type { Gates } from "../gates-config.ts";
import type { HeadClient } from "../head-client.ts";

/** `pass` is "measured" for a probe that records a number no criterion judges (a depth matrix) */
export interface ProbeResult {
  name: string;
  pass: boolean | "measured";
  summary: string;
  lines: string[];
  data: unknown;
}

export interface ProbeContext {
  head: Head;
  gates: Gates;
  server: GateLegs;
  live: HeadClient | null;
  binDir: string;
  gpu: number;
  /** the gate card's compute capability (120 for sm_120), "" when no probe runs on a card */
  cap: string;
  /** the cache formats the gate card's tier serves (the head's own without a card) */
  cache: CacheFormats;
  fs: FileSystem;
  shell: Shell;
  log: Log;
  runDir: string;
  gateRunsDir: string;
  /** the long text the haystack probes are built from, read once when first asked for */
  corpus: () => Promise<string>;
}

/** what a probe needs: fresh servers on the gate card, the card itself (llama-bench), or the
 *  live head; `applies` says why it cannot measure this head (no draft head to compare), null
 *  when it can */
export interface Probe {
  name: string;
  needs: "server" | "card" | "live";
  applies?(head: Head): string | null;
  run(ctx: ProbeContext): Promise<ProbeResult>;
}

/** the live head a live probe was selected for; the service never runs one without it */
export function liveHead(ctx: ProbeContext): HeadClient {
  if (!ctx.live) throw new Error("a live probe ran without a live head");
  return ctx.live;
}

/** content, or the reasoning when the model put everything there (thinking off, but the
 *  template may still) */
export const answerText = (reply: { text: string; reasoning: string }) =>
  reply.text || (reply.reasoning ? `[reasoning] ${reply.reasoning}` : "");

export const short = (text: string, chars = 90) => JSON.stringify(text.slice(0, chars));

export const fixed1 = (value: number) => value.toFixed(1);

export const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
