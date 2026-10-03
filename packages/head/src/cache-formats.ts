// The model's caches in the formats the engine names: the attention K/V cache (--cache-type-k/-v, one element
// per K and per V value of every attention layer, per pooled token), the recurrent state cache (-cts, the Gated
// DeltaNet state of each sequence, in a model that has one) and the sparse-attention indexer's cache (-ctki, GLM-5.3's
// DSA keys and compressor gates). A head declares the formats every profile serves, a
// profile may name its own, and [sizing] declares the element counts their bytes follow from. serve and the gates
// render the flags of the profile a machine gets and the profile check charges exactly those bytes, so a format is
// one field, never a hand-recomputed constant.
import type { Engine } from "@rig/engine";
import type { Head } from "./head.ts";
import type { HeadConfig, Profile, Sizing } from "./head-config.ts";

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
/** the indexer cache types it takes (common/arg.cpp, --cache-type-idx) */
export const INDEXER_TYPES = ["f16", "q8_0"] as const;
export type KvType = (typeof KV_TYPES)[number];
export type StateType = (typeof STATE_TYPES)[number];
export type IndexerType = (typeof INDEXER_TYPES)[number];

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
  /** absent: the model's recurrent state (if it has one) in the engine's own format, -cts not passed */
  s?: StateType | undefined;
  /** absent: the indexer's cache in the engine's own f16, -ctki not passed */
  idx?: IndexerType | undefined;
}

/** the formats a profile serves: the head's, with each one the profile names in its place */
export function profileCache(
  head: Pick<HeadConfig, "cache">,
  profile: Pick<Profile, "cache">,
): CacheFormats {
  return {
    k: profile.cache?.k ?? head.cache.k,
    v: profile.cache?.v ?? head.cache.v,
    s: profile.cache?.s ?? head.cache.s,
    idx: profile.cache?.idx ?? head.cache.idx,
  };
}

/** the K and V formats of a draft's own attention cache (-ctkd/-ctvd): its own, whatever a profile names for the
 *  target's */
export interface DraftCache {
  k: KvType;
  v: KvType;
}

/** the K and V bytes one pooled token adds */
export function kvBytesPerToken(
  sizing: Pick<Sizing, "kv_elements_per_token">,
  c: CacheFormats,
): number {
  return sizing.kv_elements_per_token * (BYTES[c.k] + BYTES[c.v]);
}

/** the K and V bytes one pooled token adds to a draft's cache, at its formats (the engine's f16 when it names none) */
export function draftKvBytesPerToken(
  draft: { kv_elements_per_token: number },
  c: DraftCache | undefined,
): number {
  return draft.kv_elements_per_token * (BYTES[c?.k ?? "f16"] + BYTES[c?.v ?? "f16"]);
}

/** MiB of one sequence's recurrent state, rounded up: the part -cts narrows and the part it leaves f32 (a model
 *  without a state has neither) */
export function stateMiBPerCopy(
  sizing: Pick<Sizing, "state_elements_per_copy" | "state_fixed_mib">,
  c: CacheFormats,
): number {
  return Math.ceil(
    (sizing.state_elements_per_copy * BYTES[c.s ?? "f32"]) / 1048576 + sizing.state_fixed_mib,
  );
}

/** the engine's flags for the formats, in serve's order; the K bias rides a q4_0 K only (the engine refuses it on
 *  any other type), the state's and the indexer's format only when one is named */
export function cacheArgv(head: Pick<Head, "cache" | "path">, c: CacheFormats): string[] {
  const bias =
    c.k === "q4_0" && head.cache.mean_center
      ? ["--kv-mean-center", head.path(head.cache.mean_center)]
      : [];
  const state = c.s ? ["-cts", c.s] : [];
  const indexer = c.idx ? ["-ctki", c.idx] : [];
  return ["--cache-type-k", c.k, ...bias, "--cache-type-v", c.v, ...state, ...indexer];
}

/** what a llama-bench leg runs of the formats: the K, V, state and indexer it is given (-ctk/-ctv/-cts/-ctki; the pin's
 *  llama-bench parses -cts since fork b2eb4336a and an f32 type since 4104c47d5), and no K bias, since llama-bench
 *  parses no --kv-mean-center (tools/llama-bench/llama-bench.cpp: -ctk, -ctv, -cts and -ctki are its only cache flags).
 *  A bench leg records `ran`, never the profile's formats: its K is unbiased whatever the profile serves */
export function benchCache(c: CacheFormats): {
  args: string[];
  ran: CacheFormats & { k_bias: false };
} {
  return {
    args: [
      "-ctk",
      c.k,
      "-ctv",
      c.v,
      ...(c.s ? ["-cts", c.s] : []),
      ...(c.idx ? ["-ctki", c.idx] : []),
    ],
    ran: { k: c.k, v: c.v, s: c.s, idx: c.idx, k_bias: false },
  };
}

/** why the pinned engine cannot serve these formats on a CUDA card, or null when it can: a K/V pair its flash
 *  attention has no CUDA kernel for runs attention on the CPU, which no profile's numbers survive, and a state type its
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
  if (c.s && !state.includes(c.s))
    return `the pinned engine (${engine.sha7}) does not run a ${c.s} recurrent state (it runs ${state.join(", ")}); the server would abort on its first decode`;
  return null;
}
