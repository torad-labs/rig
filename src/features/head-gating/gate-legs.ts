// A fresh server per leg on the gate card: what a probe asks for and what it gets back. The
// domain's contract; infrastructure/GateServer.ts spawns the real llama-server.
import type { HeadClient } from "./head-client.ts";

/** `draft: true` loads the head's draft beside the pack (a probe that measures speculation asks for it; the quality probes measure the pack alone) */
export interface LegOptions {
  label: string;
  pack: string;
  ctx: number;
  slots: number;
  draft?: boolean;
  extra?: string[];
}

export interface GateLegs {
  /** run `fn` against a fresh server on `pack`, stopping it whatever happens */
  leg<T>(options: LegOptions, fn: (client: HeadClient) => Promise<T>): Promise<T>;
}
