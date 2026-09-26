// The model's two caches in the formats the engine names: the attention K/V cache (--cache-type-k/-v, one element
// per K and per V value of every attention layer, per pooled token) and the recurrent state cache (-cts, the Gated
// DeltaNet state of each sequence). A head declares the formats every tier serves and the element counts their bytes
// follow from; a tier may name its own. serve and the gates render the flags of the tier a card gets and the tier
// check charges exactly those bytes, so a format is one field, never a hand-recomputed constant.
import type { Engine } from "../engine/engine.ts";
import type { Head } from "./head.ts";
import type { HeadConfig, Tier } from "./head-config.ts";

/** the K/V cache types the pinned engine parses (common/arg.cpp, kv_cache_types) */
export const KV_TYPES = [
  "f32",
  "f16",
  "bf16",
  "q8_0",
  "q4_0",
  "q4_1",
  "iq4_nl",
  "q5_0",
  "q5_1",
] as const;
/** the recurrent state types it takes (common/arg.cpp, --cache-type-s) */
export const STATE_TYPES = ["f32", "f16", "bf16", "q8_0"] as const;
export type KvType = (typeof KV_TYPES)[number];
export type StateType = (typeof STATE_TYPES)[number];

/** bytes per element: a ggml block's bytes over the elements it holds (ggml.c, type_traits) */
const BYTES: Record<KvType, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 34 / 32,
  q4_0: 18 / 32,
  q4_1: 20 / 32,
  iq4_nl: 18 / 32,
  q5_0: 22 / 32,
  q5_1: 24 / 32,
};

export interface CacheFormats {
  k: KvType;
  v: KvType;
  s: StateType;
}

/** the formats a tier serves: the head's, with each one the tier names in its place */
export function tierCache(
  head: Pick<HeadConfig, "cache">,
  tier: Pick<Tier, "cache">,
): CacheFormats {
  return {
    k: tier.cache?.k ?? head.cache.k,
    v: tier.cache?.v ?? head.cache.v,
    s: tier.cache?.s ?? head.cache.s,
  };
}

/** the formats of a draft head's own attention cache (-ctkd/-ctvd) and the K (and V) elements one pooled token adds
 *  to it: a draft's formats are its own, whatever a tier names for the target's */
export interface DraftCache {
  k: KvType;
  v: KvType;
  kv_elements_per_token: number;
}

/** the K and V bytes one pooled token adds */
export function kvBytesPerToken(head: Pick<HeadConfig, "cache">, c: CacheFormats): number {
  return head.cache.kv_elements_per_token * (BYTES[c.k] + BYTES[c.v]);
}

/** the K and V bytes one pooled token adds to a draft head's cache */
export function draftKvBytesPerToken(c: DraftCache): number {
  return c.kv_elements_per_token * (BYTES[c.k] + BYTES[c.v]);
}

/** MiB of one sequence's recurrent state, rounded up: the part -cts narrows and the part it leaves f32 */
export function stateMiBPerCopy(head: Pick<HeadConfig, "cache">, c: CacheFormats): number {
  return Math.ceil(
    (head.cache.state_elements_per_copy * BYTES[c.s]) / 1048576 + head.cache.state_fixed_mib,
  );
}

/** the engine's flags for the formats, in serve's order; the K bias rides a q4_0 K only (the engine refuses it on
 *  any other type) */
export function cacheArgv(head: Pick<Head, "cache" | "path">, c: CacheFormats): string[] {
  const bias =
    c.k === "q4_0" && head.cache.mean_center
      ? ["--kv-mean-center", head.path(head.cache.mean_center)]
      : [];
  return ["--cache-type-k", c.k, ...bias, "--cache-type-v", c.v, "-cts", c.s];
}

/** what a llama-bench leg runs of the formats: the K and V it is given (-ctk/-ctv), and the context's defaults for the
 *  rest, since llama-bench at the pin parses no -cts and no --kv-mean-center (tools/llama-bench/llama-bench.cpp: -ctk and
 *  -ctv are its only cache flags; llama_context_default_params' type_s is f32). A bench leg records `ran`, never the
 *  tier's formats: its state is f32 and its K unbiased whatever the tier serves */
export function benchCache(c: CacheFormats): {
  args: string[];
  ran: CacheFormats & { k_bias: false };
} {
  return { args: ["-ctk", c.k, "-ctv", c.v], ran: { k: c.k, v: c.v, s: "f32", k_bias: false } };
}

/** why the pinned engine cannot serve these formats on a CUDA card, or null when it can: a K/V pair its flash
 *  attention has no CUDA kernel for runs attention on the CPU, which no tier's numbers survive, and a state type its
 *  graph does not run aborts the server on its first decode. `draft` is the draft head's cache when the server loads
 *  one: its attention runs the same flash attention */
export function cacheRefusal(
  engine: Pick<Engine, "caches" | "sha7">,
  c: CacheFormats,
  draft?: Pick<DraftCache, "k" | "v">,
): string | null {
  const { fa_kv, state } = engine.caches;
  for (const [what, k, v] of [
    ["K/V", c.k, c.v],
    ...(draft ? [["the draft's K/V", draft.k, draft.v]] : []),
  ]) {
    const pair = `${k}/${v}`;
    if (!fa_kv.includes(pair))
      return `the pinned engine (${engine.sha7}) has no CUDA flash attention for ${what} ${pair} (it runs ${fa_kv.join(", ")}); attention would run on the CPU`;
  }
  if (!state.includes(c.s))
    return `the pinned engine (${engine.sha7}) does not run a ${c.s} recurrent state (it runs ${state.join(", ")}); the server would abort on its first decode`;
  return null;
}
