import { describe, expect, test } from "bun:test";
import {
  benchCache,
  type CacheFormats,
  cacheArgv,
  cacheRefusal,
  draftKvBytesPerToken,
  kvBytesPerToken,
  profileCache,
  stateMiBPerCopy,
} from "./cache-formats.ts";

// bonsai's counts: 16 attention layers × 4 KV heads × 256, and 48 Gated DeltaNet layers × 48 heads × 128 × 128
const cache = { k: "q4_0", v: "q4_0", s: "q8_0", mean_center: "assets/bias.gguf" } as const;
const sizing = {
  kv_elements_per_token: 16384,
  state_elements_per_copy: 37748736,
  state_fixed_mib: 5.625,
};
const head = { cache, path: (rel: string) => `/h/${rel}` };
const served: CacheFormats = { k: "q4_0", v: "q4_0", s: "q8_0" };

describe("cache formats", () => {
  test("bytes follow from the element counts and ggml's block sizes", () => {
    const kv = (k: CacheFormats["k"], v: CacheFormats["v"]) =>
      kvBytesPerToken(sizing, { ...served, k, v });
    expect(kv("q4_0", "q4_0")).toBe(18432); // 18 B per 32 values
    expect(kv("q5_1", "q5_1")).toBe(24576); // 24 B per 32
    expect(kv("q5_1", "q8_0")).toBe(29696);
    expect(kv("q8_0", "q8_0")).toBe(34816); // 34 B per 32
    expect(kv("f16", "f16")).toBe(65536);
    const state = (s: CacheFormats["s"]) => stateMiBPerCopy(sizing, { ...served, s });
    expect(state("q8_0")).toBe(44); // 38.25 + 5.625 of f32 conv state: 43.875, as measured
    expect(state("f16")).toBe(78); // 72 + 5.625
    expect(state("bf16")).toBe(78);
    expect(state("f32")).toBe(150); // 144 + 5.625: 149.625, as measured
  });
  test("a profile names any of the three in place of the head's, each on its own", () => {
    expect(profileCache(head, {})).toEqual(served);
    expect(profileCache(head, { cache: { k: "q8_0" } })).toEqual({
      k: "q8_0",
      v: "q4_0",
      s: "q8_0",
    });
    expect(profileCache(head, { cache: { s: "f16" } })).toEqual({ k: "q4_0", v: "q4_0", s: "f16" });
  });
  test("the flags come in serve's order, the K bias with a q4_0 K only", () => {
    expect(cacheArgv(head, served)).toEqual([
      "--cache-type-k",
      "q4_0",
      "--kv-mean-center",
      "/h/assets/bias.gguf",
      "--cache-type-v",
      "q4_0",
      "-cts",
      "q8_0",
    ]);
    expect(cacheArgv(head, { k: "q8_0", v: "q8_0", s: "f16" })).toEqual([
      "--cache-type-k",
      "q8_0",
      "--cache-type-v",
      "q8_0",
      "-cts",
      "f16",
    ]);
    const unbiased = { ...head, cache: { ...cache, mean_center: undefined } };
    expect(cacheArgv(unbiased, served)).not.toContain("--kv-mean-center");
    // a model with no recurrent state, or one served in the engine's own format, gets no -cts at all
    expect(cacheArgv(unbiased, { k: "f16", v: "f16" })).toEqual([
      "--cache-type-k",
      "f16",
      "--cache-type-v",
      "f16",
    ]);
    // llama-bench takes the K, V and state and no K bias, and that is what a bench leg records, not the profile's formats
    expect(benchCache({ k: "q5_1", v: "q8_0", s: "f16" })).toEqual({
      args: ["-ctk", "q5_1", "-ctv", "q8_0", "-cts", "f16"],
      ran: { k: "q5_1", v: "q8_0", s: "f16", k_bias: false },
    });
  });
  test("a K/V pair the engine's CUDA flash attention does not run, or a state type it does not run, is refused by name", () => {
    const engine = {
      sha7: "abc1234",
      caches: { fa_kv: ["f16/f16", "q4_0/q4_0", "q8_0/q8_0", "bf16/bf16"], state: ["f32", "q8_0"] },
    };
    expect(cacheRefusal(engine, served)).toBeNull();
    expect(cacheRefusal(engine, { k: "f16", v: "f16" })).toBeNull(); // no state named, none to refuse
    expect(cacheRefusal(engine, { k: "q8_0", v: "q8_0", s: "f32" })).toBeNull();
    for (const s of ["f16", "bf16"] as const)
      expect(cacheRefusal(engine, { ...served, s })).toContain(
        `the pinned engine (abc1234) does not run a ${s} recurrent state (it runs f32, q8_0)`,
      );
    for (const [k, v] of [
      ["q5_1", "q5_1"],
      ["q8_0", "q4_0"],
    ] as const) {
      expect(cacheRefusal(engine, { k, v, s: "q8_0" })).toContain(
        `the pinned engine (abc1234) has no CUDA flash attention for K/V ${k}/${v}`,
      );
    }
    expect(cacheRefusal(engine, served, { k: "q4_0", v: "q4_0" })).toBeNull();
    expect(cacheRefusal(engine, served, { k: "q5_1", v: "q4_0" })).toContain(
      "has no CUDA flash attention for the draft's K/V q5_1/q4_0",
    );
  });
  test("a draft's cache bytes follow from its own formats and element count, the engine's f16 when it names none", () => {
    const draft = { kv_elements_per_token: 1024 };
    expect(draftKvBytesPerToken(draft, { k: "q4_0", v: "q4_0" })).toBe(1152);
    expect(draftKvBytesPerToken(draft, { k: "q8_0", v: "f16" })).toBe(3136);
    expect(draftKvBytesPerToken(draft, undefined)).toBe(4096);
  });
});
