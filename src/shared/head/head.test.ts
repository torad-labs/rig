import { describe, expect, test } from "bun:test";
import { putHead, withSidecarDraft } from "../../../test/fakes/head-fixtures.ts";
import { fakePorts } from "../../../test/fakes/index.ts";
import { layoutAt } from "../layout.ts";
import {
  draftKvBytesPerToken,
  type KvType,
  kvBytesPerToken,
  stateMiBPerCopy,
  tierCache,
} from "./cache-formats.ts";
import { loadHead, parseHeadToml } from "./head.ts";
import {
  CUDA_GRAPHS_MAX,
  cudaGraphsMiB,
  GRAPH_SLOT_ROWS,
  GRAPH_SLOTS,
  graphSlotsMiB,
  type Tier,
  tierNeedMiB,
  tierSpeculates,
} from "./head-config.ts";

const real = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();
const evidence = await Bun.file(
  `${import.meta.dir}/../../../heads/bonsai-2-27b/evidence.md`,
).text();

describe("head.toml", () => {
  test("the real bonsai head parses and every tier fits its constants", () => {
    const r = parseHeadToml(real);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    for (const t of r.value.geometry.tiers)
      expect(tierNeedMiB(r.value, t)).toBeLessThanOrEqual(t.min_vram_mib);
    // the [cache] element counts reproduce what was measured: 7,488 MiB of q4_0 K/V at 425,984 tokens on the 5080
    // and 43.875 MiB a q8_0 state copy (526.5 MiB at 4 slots x (1 + n_max 2)), charged 44; the head's f16 state is
    // 72 MiB a copy and the f32 conv states, charged 78
    const cache = tierCache(r.value, {});
    expect(kvBytesPerToken(r.value, cache)).toBe(18432);
    expect(Math.ceil((425984 * kvBytesPerToken(r.value, cache)) / 1048576)).toBe(7488);
    expect(stateMiBPerCopy(r.value, { ...cache, s: "q8_0" })).toBe(44);
    expect(stateMiBPerCopy(r.value, cache)).toBe(78);
    expect(stateMiBPerCopy(r.value, { ...cache, s: "f32" })).toBe(150); // 149.625: 1,795.5 MiB at 12 copies
  });
  test("lens is optional and accepts only explicit booleans and string arguments", () => {
    const absent = parseHeadToml(real.replace(/\[runtime\.lens\][\s\S]*?(?=\[client\])/, ""));
    expect(absent.ok).toBe(true);
    if (absent.ok) expect(absent.value.runtime.lens).toBeUndefined();
    for (const enabled of [false, true]) {
      const r = parseHeadToml(real.replace("enabled = false", `enabled = ${enabled}`));
      expect(r.ok).toBe(true);
      if (r.ok) expect(r.value.runtime.lens?.enabled).toBe(enabled);
    }
    for (const [replacement, key] of [
      ['enabled = "false"', "enabled"],
      ["", "enabled"],
      ["enabled = false\nunknown = true", "unknown"],
    ]) {
      const r = parseHeadToml(real.replace("enabled = false", replacement!));
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.message).toContain(key!);
    }
    for (const args of ["", "args = [42]"]) {
      const r = parseHeadToml(
        real.replace(
          /\[runtime\.lens\][\s\S]*?(?=\[client\])/,
          `[runtime.lens]\nenabled = false\n${args}\n\n`,
        ),
      );
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.message).toContain("args");
    }
  });
  test("a cutoff must be below 1: the engine's top-1 does reach 1.0f at saturation, so 1 buys a fully charged head that drafts almost never", () => {
    for (const key of ["p_min", "chain_p_min"]) {
      const at = (p: string) =>
        parseHeadToml(real.replace('type = "draft-mtp"\n', `type = "draft-mtp"\n${key} = ${p}\n`));
      expect(at("0.999").ok).toBe(true);
      expect(at("0").ok).toBe(true);
      const r = at("1.0");
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.message).toContain(`${key} must be below 1`);
    }
  });
  test("a rendered flag inside a free-form list is refused by name: the copy would win over the typed value silently", () => {
    for (const [section, key, value] of [
      ["speculative", "args", '["-ctkd", "q4_0", "--spec-draft-chain-p-min", "0.2"]'],
      ["runtime", "extra", '["-lv", "4", "--parallel", "1"]'], // the alias: 1 slot served, 4 charged
      ["runtime", "extra", '["-lv", "4", "--ctx_size", "8192"]'], // the engine reads it as --ctx-size
      ["runtime", "sampling", '["--temp", "1.0", "-ctkd", "f16"]'], // after the draft's own: it would win
      ["runtime.lens", "args", '["--lens-layers", "63", "--cache-ram", "0"]'],
    ]) {
      const r = parseHeadToml(
        real.replace(
          new RegExp(
            `(\\[${section!.replace(".", "\\.")}\\][\\s\\S]*?^)${key} = \\[[\\s\\S]*?\\]`,
            "m",
          ),
          `$1${key} = ${value}`,
        ),
      );
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.message).toContain(`${section}.${key} carries`);
    }
  });
  test("lens enabled without --lens-layers is refused: the unit would be identical to lens off", () => {
    const r = parseHeadToml(
      real
        .replace("enabled = false", "enabled = true")
        .replace('"--lens-layers", "48,52,56,60,62,63",', ""),
    );
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("no --lens-layers");
  });
  test("a missing sha is rejected by name", () => {
    const r = parseHeadToml(real.replace(/^sha256 = "389b.*$/m, ""));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("served.sha256");
  });
  test("a draft cutoff outside the unit interval or not a number is rejected by name; the head sets none", () => {
    const withLine = (line: string) =>
      real.replace('type = "draft-mtp"\n', `type = "draft-mtp"\n${line}\n`);
    for (const key of ["chain_p_min", "p_min"] as const) {
      for (const bad of ["1.5", "-0.1", '"0.3"']) {
        const r = parseHeadToml(withLine(`${key} = ${bad}`));
        expect(r.ok).toBe(false);
        if (!r.ok) expect(r.message).toContain(key);
      }
      const good = parseHeadToml(withLine(`${key} = 0.3`));
      expect(good.ok && good.value.speculative?.[key]).toBe(0.3);
    }
    const r = parseHeadToml(real);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.value.speculative?.chain_p_min).toBeUndefined();
      expect(r.value.speculative?.p_min).toBeUndefined();
    }
  });
  test("chain_p_min on a draft type the engine ignores it for is rejected by name; p_min is not", () => {
    const sidecar = withSidecarDraft(real);
    expect(sidecar).toContain('type = "draft-dflash"\n');
    const withLine = (line: string) =>
      sidecar.replace('type = "draft-dflash"\n', `type = "draft-dflash"\n${line}\n`);
    const r = parseHeadToml(withLine("chain_p_min = 0.3"));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("chain_p_min");
    expect(parseHeadToml(withLine("p_min = 0.3")).ok).toBe(true);
    const simple = parseHeadToml(
      withLine("chain_p_min = 0.3").replace('type = "draft-dflash"', 'type = "draft-simple"'),
    );
    expect(simple.ok && simple.value.speculative?.chain_p_min).toBe(0.3);
  });
  test("an unknown key is rejected, so a typo cannot silently do nothing", () => {
    const r = parseHeadToml(real.replace("[context]", "[context]\ncontext_window = 1"));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("context_window");
  });
  test("[public] pins the public steps' output: they come first, and it exists exactly when private steps follow them", () => {
    const refused = (toml: string, message: string) => {
      const r = parseHeadToml(toml);
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.message).toContain(message);
    };
    const url = /^url = .*\n/m;
    const urlLine = real.match(url)?.[0] ?? "";
    // the public step moved after the private one
    refused(
      real.replace(url, "").replace("row_cap = 0.10\n", `row_cap = 0.10\n${urlLine}`),
      "follows a private one",
    );
    // mixed steps without [public]: a stranger would get the source pack
    refused(real.replace(/\[public\][^\n]*\n(?:[^\n[]+\n)+/, ""), "declares no [public]");
    // [public] with no private step after the public ones (every step public, or none)
    refused(real.replace(url, ""), "[public] needs public [derive] steps");
    // [public] that is the source's own bytes
    refused(
      real.replace(
        /sha256 = "0e5524be[0-9a-f]+"/,
        'sha256 = "3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4"',
      ),
      "must be its own file and bytes",
    );
  });
  test("every file rig writes pins its size, so `rig up` checks the disk before it fetches: a public asset without one is rejected by name", () => {
    const source = parseHeadToml(real.replace(/^bytes = .*$/m, ""));
    expect(!source.ok && source.message).toContain("source.bytes");
    const asset = parseHeadToml(real.replace(/^head_bytes = .*$/m, ""));
    expect(!asset.ok && asset.message).toContain(
      "the [derive] asset bonsai-2-27b-mtp-r2.gguf has a url but no size",
    );
    // the private adapter is never fetched, so it needs none
    expect(real).not.toContain("lora_bytes");
  });
  test("served must equal source without a derive step", () => {
    const noDerive = real.replace(/\[\[derive\]\][\s\S]*?\n\n/g, "\n");
    const r = parseHeadToml(noDerive);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("served must equal source");
  });
  test("a tier that cannot hold its own geometry is rejected", () => {
    const r = parseHeadToml(real.replace("min_vram_mib = 16000", "min_vram_mib = 12000"));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("cannot hold slots=4 ctx=294912");
  });
  test("advertise above the trained window is rejected", () => {
    const r = parseHeadToml(real.replace("advertise = 245760", "advertise = 300000"));
    expect(r.ok).toBe(false);
  });
  test("the in-pack MTP head is charged to every tier, the 5080 included: weights, overhead, its KV per token and n_max state snapshots per slot", () => {
    const r = parseHeadToml(real);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const h = r.value,
      s = h.speculative!;
    expect(s.type).toBe("draft-mtp");
    expect("file" in s).toBe(false); // the head is inside the served pack, pinned by its sha
    const [pro6000, h100, r5090, r5080] = h.geometry.tiers as [Tier, Tier, Tier, Tier];
    expect([pro6000, h100, r5090, r5080].map((t) => tierSpeculates(h, t))).toEqual([
      true,
      true,
      true,
      true,
    ]);
    expect(tierNeedMiB(h, r5090) - tierNeedMiB(h, { ...r5090, speculative: false })).toBe(
      s.weights_mib +
        s.overhead_mib +
        Math.ceil((r5090.ctx * (draftKvBytesPerToken(s.cache) + s.bytes_per_token)) / 1048576) +
        r5090.slots * h.geometry.compute_per_output_row_mib * s.n_max +
        r5090.slots * stateMiBPerCopy(h, tierCache(h, r5090)) * s.n_max +
        // the graph slots: each of the target's holds 32 output rows where it held 8, and the draft context has its own
        GRAPH_SLOTS *
          ((GRAPH_SLOT_ROWS - r5090.slots) * h.geometry.compute_per_output_row_mib +
            s.graph_slot.mib +
            s.graph_slot.per_seq_mib * r5090.slots +
            (r5090.ctx * s.graph_slot.bytes_per_token) / 1048576) +
        CUDA_GRAPHS_MAX * s.cuda_graph_mib, // and its CUDA graphs
    );
    expect(tierNeedMiB(h, r5090)).toBe(26238); // 8 slots × (78 MiB of f16 rollback state + 3 MiB of verify row) × n_max 3, each slot's own row, 3 × 149 of graph slots, 8 × 12 of CUDA graphs, the 5090's 335 of runtime
    // the 5080: full-pool MTP KV and bit-packed attention masks, 4 × (78 + 3) × 3 of rollback state and verify rows,
    // the draft vocabulary's 127.5 MiB of LM-head rows in weights_mib, the 312 MiB of token embeddings on the card, 241 of
    // graph slots, 8 × 12 of CUDA graphs and the 5080's 55 of runtime
    expect(tierNeedMiB(h, r5080)).toBe(15216);
    // the compute buffer as the 5080 measured it (167.22 MiB at 294,912 and 12 = 4 × (1 + 2) rows) is
    // charged in full: the fixed part, the per-token mask and every output row, the first per slot too
    const row = h.geometry.compute_per_output_row_mib;
    const buffer = (slots: number, nMax: number) =>
      h.geometry.compute_mib + Math.ceil((294912 * 64) / 1048576) + slots * row * (1 + nMax);
    expect(buffer(4, 2)).toBeGreaterThanOrEqual(167.22);
    expect(buffer(4, 8)).toBeGreaterThanOrEqual(239.22);
    // each slot costs its recurrent state and its own output row, drafting or not, and in each graph slot its sequence
    // and that row again
    const undrafted = { ...r5080, speculative: false };
    expect(tierNeedMiB(h, undrafted) - tierNeedMiB(h, { ...undrafted, slots: 0 })).toBe(
      4 * (stateMiBPerCopy(h, tierCache(h, r5080)) + row) +
        GRAPH_SLOTS * 4 * (h.geometry.graph_slot.per_seq_mib + row),
    );
    // the 1.625 undrafted windows no longer fit beside it
    expect(tierNeedMiB(h, { ...r5080, ctx: 425984 })).toBeGreaterThan(r5080.min_vram_mib);
  });
  test("the graph slots are charged what the engine logs at load for each, target and draft, and within 1 MiB a slot of it", () => {
    const r = parseHeadToml(real);
    if (!r.ok) throw new Error(r.message);
    const at = (n: number) => ({ ...r.value, speculative: { ...r.value.speculative!, n_max: n } });
    const r5080 = r.value.geometry.tiers.find((t) => t.min_vram_mib === 16000)!;
    // "graph slot buffer size <= X MiB, 3 slots" on the 5080, engine 87a3596, the target's and the draft's (2026-09-27)
    const logged = [
      { slots: 1, ctx: 262144, s: "q8_0", n_max: 3, target: 21.99, draft: 8.95 },
      { slots: 2, ctx: 262144, s: "q8_0", n_max: 3, target: 37.22, draft: 9.9 },
      { slots: 4, ctx: 262144, s: "q8_0", n_max: 3, target: 67.69, draft: 11.79 },
      { slots: 4, ctx: 294912, s: "f16", n_max: 3, target: 67.81, draft: 11.92 },
      { slots: 4, ctx: 294912, s: "f16", n_max: 4, target: 79.81, draft: 11.92 },
    ] as const;
    const off = logged.flatMap((l) => {
      const bound = GRAPH_SLOTS * (l.target + l.draft);
      const charged = graphSlotsMiB(at(l.n_max), {
        ...r5080,
        slots: l.slots,
        ctx: l.ctx,
        cache: { s: l.s },
      });
      return charged < bound || charged > bound + GRAPH_SLOTS
        ? [
            `${l.slots} × ${l.ctx} at n_max ${l.n_max}: ${bound.toFixed(2)} logged, ${charged} charged`,
          ]
        : [];
    });
    expect(off).toEqual([]);
  });
  test("the CUDA graphs are charged CUDA_GRAPHS_MAX of each context's largest, as the engine logs it, on every tier", () => {
    const r = parseHeadToml(real);
    if (!r.ok) throw new Error(r.message);
    const h = r.value;
    // "N CUDA graphs held at most (cap C), the largest took X MiB to instantiate" (engine d0f8bae), the target's and the
    // draft's largest under concurrent requests: the 5080 at its tier, uncapped and at caps 16 and 8, and a rented 5090
    // at the 5080's geometry and its own (ngram-mtp-2026-09-26/graph-cap, 2026-09-27)
    const logged = [
      { card: "5080", slots: 4, ctx: 294912, target: 10, draft: 2 },
      { card: "5090", slots: 4, ctx: 294912, target: 8, draft: 0 },
      { card: "5090", slots: 8, ctx: 786432, target: 8, draft: 0 },
    ] as const;
    const under = h.geometry.tiers.flatMap((tier) =>
      logged
        .filter((l) => cudaGraphsMiB(h, tier) < CUDA_GRAPHS_MAX * (l.target + l.draft))
        .map(
          (l) =>
            `tier ${tier.min_vram_mib}: ${cudaGraphsMiB(h, tier)} charged, ${l.card} ${l.slots} × ${l.ctx} logged more`,
        ),
    );
    expect(under).toEqual([]);
    // and each tier's need carries it: with graphs that take nothing, every tier needs that much less
    const none = parseHeadToml(real.replace(/^cuda_graph_mib = \d+/gm, "cuda_graph_mib = 0"));
    if (!none.ok) throw new Error(none.message);
    expect(
      h.geometry.tiers.map(
        (t, i) => tierNeedMiB(h, t) - tierNeedMiB(none.value, none.value.geometry.tiers[i]!),
      ),
    ).toEqual(
      h.geometry.tiers.map(
        () => CUDA_GRAPHS_MAX * (h.geometry.cuda_graph_mib + h.speculative!.cuda_graph_mib),
      ),
    );
  });
  test("a sidecar draft is charged by the same constants and names its file", () => {
    const r = parseHeadToml(withSidecarDraft(real));
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const h = r.value,
      s = h.speculative!;
    expect(s.type).toBe("draft-dflash");
    expect("file" in s && s.file).toBe("Bonsai-2-27B-DFlash2-Q8_0.gguf");
    const r5090 = h.geometry.tiers[2]!;
    expect(tierNeedMiB(h, r5090) - tierNeedMiB(h, { ...r5090, speculative: false })).toBe(
      100 +
        50 +
        Math.ceil((r5090.ctx * (2048 + 64)) / 1048576) + // f16 K + V over 512 elements, and its compute
        8 * h.geometry.compute_per_output_row_mib * 3 +
        8 * stateMiBPerCopy(h, tierCache(h, r5090)) * 3 +
        GRAPH_SLOTS * (24 * h.geometry.compute_per_output_row_mib + 10 + 8 * 1 + 3) + // its slots, the target's 24 more rows
        CUDA_GRAPHS_MAX * 2, // its CUDA graphs
    );
  });
  test("a tier that asks for a draft head the head does not declare is rejected by name", () => {
    const r = parseHeadToml(
      real
        .replace(/\[speculative\][\s\S]*?\n\n/, "")
        .replace("ctx = 294912\n", "ctx = 294912\nspeculative = true\n"),
    );
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("asks for a draft head");
  });
  test("a tier that cannot hold the draft head beside its window is rejected by name", () => {
    const r = parseHeadToml(real.replace("ctx = 294912\n", "ctx = 425984\n"));
    expect(r.ok).toBe(false);
    if (!r.ok)
      expect(r.message).toContain(
        "cannot hold slots=4 ctx=425984: needs 17683 MiB (KV + weights + compute + graph slots + CUDA graphs + slot state + draft weights, overhead, compute and rollback snapshots + the driver's runtime)",
      );
  });
  test("a tier's own cache formats are charged at their bytes, the state on every copy it keeps", () => {
    const at = (text: string) => {
      const r = parseHeadToml(text);
      if (!r.ok) throw new Error(r.message);
      return r.value.geometry.tiers;
    };
    // the 5090 in q4_1 K and V: 786,432 cells × 2,048 more bytes = 1,536 MiB over its q4_0 need
    const q41 = at(
      real.replace("ctx = 786432\n", 'ctx = 786432\ncache = { k = "q4_1", v = "q4_1" }\n'),
    );
    const h = parseHeadToml(real);
    if (!h.ok) throw new Error(h.message);
    expect(tierNeedMiB(h.value, q41[2]!)).toBe(26238 + 1536);
    // the 5080 with a q8_0 state: 34 MiB less a copy (44 against the head's f16 78), 4 slots × (1 + n_max 3) copies
    const q8 = at(real.replace("ctx = 294912\n", 'ctx = 294912\ncache = { s = "q8_0" }\n'));
    expect(tierNeedMiB(h.value, q8[3]!)).toBe(15216 - 16 * 34);
    expect(tierNeedMiB(h.value, q8[2]!)).toBe(26238); // the tiers that name nothing keep the head's formats
  });
  // What a card held at its highest through a run (nvidia-smi memory.used: the buffers the constants charge and what the
  // driver holds beside them), each on the card of the tier named, at the run's own geometry and formats: the charge,
  // that tier's runtime_mib included, is never under one. `host`: a run before the token embeddings moved to the card
  // (d84bf3c), which holds them now at 312 MiB more (in geometry.weights_mib).
  const peaks = [
    // a 5070 Ti beside its desktop, engine 3c7e643, n_max 3 with the draft vocabulary, a q8_0 state (evidence.md, 2026-09-25)
    { tier: 14100, slots: 4, ctx: 262144, s: "q8_0", draft: true, host: true, peak: 13410 },
    { tier: 13700, slots: 2, ctx: 262144, s: "q8_0", draft: true, host: true, peak: 13006 },
    { tier: 13500, slots: 1, ctx: 262144, s: "q8_0", draft: true, host: true, peak: 12828 },
    // the 5080 at its tier, a q8_0 state (evidence.md, 2026-09-25)
    { tier: 16000, slots: 4, ctx: 294912, s: "q8_0", draft: true, host: true, peak: 14012 },
    // the 5080 at one window, engines 48ebd21 and 4283c36, the head's argv (engine10-gate-2026-09-26/exact-local.sh)
    { tier: 16000, slots: 1, ctx: 262144, s: "f16", draft: true, host: false, peak: 13360 },
    { tier: 16000, slots: 1, ctx: 262144, s: "f16", draft: false, host: false, peak: 12126 },
    // a rented 5090, engine 48ebd21, a q8_0 state (the engine-10 gate, sampled every 250 ms, 2026-09-26)
    { tier: 30000, slots: 1, ctx: 262144, s: "q8_0", draft: true, host: false, peak: 13525 },
    { tier: 30000, slots: 4, ctx: 294912, s: "q8_0", draft: true, host: false, peak: 14695 },
    // the operator's 5090 head at its tier on rig 0.1.9 after its first requests (2026-09-26)
    { tier: 30000, slots: 8, ctx: 786432, s: "f16", draft: true, host: false, peak: 25574 },
    // a rented 5090 at the 5080's and the desktop's geometry, engine 9ddd463 (graph slots), four concurrent 21.8K-token
    // requests and a 5.1K one, sampled every 50 ms (engine11-gate-2026-09-26, 2026-09-27)
    { tier: 30000, slots: 4, ctx: 294912, s: "f16", draft: true, host: false, peak: 15430 },
    { tier: 30000, slots: 4, ctx: 262144, s: "q8_0", draft: true, host: false, peak: 14290 },
    // the 5080 at its tier, engine 9760e96 + 87a3596, four concurrent streams of 13K-31K-token agentic requests, served
    // sampling and greedy (ngram-mtp-2026-09-26/slots-charge, 2026-09-27)
    { tier: 16000, slots: 4, ctx: 294912, s: "f16", draft: true, host: false, peak: 15216 },
    { tier: 16000, slots: 4, ctx: 294912, s: "f16", draft: true, host: false, peak: 15204 },
    // the same 5090 at the 5080's geometry and its own, engine d0f8bae (8 CUDA graphs a context), four and eight concurrent
    // 21.8K-token requests and a 5.1K one, and the 5080 at its tier under its four streams, served sampling
    // (ngram-mtp-2026-09-26/graph-cap, 2026-09-27)
    { tier: 30000, slots: 4, ctx: 294912, s: "f16", draft: true, host: false, peak: 15430 },
    { tier: 30000, slots: 8, ctx: 786432, s: "f16", draft: true, host: false, peak: 25938 },
    { tier: 16000, slots: 4, ctx: 294912, s: "f16", draft: true, host: false, peak: 15180 },
  ] as const;
  test("every peak recorded on a card is within the charge at its geometry, the tier's runtime_mib included", () => {
    const r = parseHeadToml(real);
    if (!r.ok) throw new Error(r.message);
    const h = r.value;
    const under = peaks.flatMap((p) => {
      const tier = h.geometry.tiers.find((t) => t.min_vram_mib === p.tier)!;
      const charged = tierNeedMiB(h, {
        ...tier,
        slots: p.slots,
        ctx: p.ctx,
        cache: { ...tier.cache, s: p.s },
        speculative: p.draft,
      });
      const held = p.peak + (p.host ? 312 : 0);
      return charged < held
        ? [`${p.slots} × ${p.ctx} on tier ${p.tier}: ${held} held, ${charged} charged`]
        : [];
    });
    expect(under).toEqual([]);
  });
  test("the cache flags are refused in runtime.args under every spelling: serve renders them from [cache]", () => {
    for (const flag of [
      "--cache-type-k",
      "-ctk",
      "--cache-type-v",
      "-ctv",
      "-cts",
      "--cache-type-s",
      "--kv-mean-center",
    ]) {
      const r = parseHeadToml(
        real.replace(
          'args = [\n  "--attn-mask-bits"',
          `args = [\n  "${flag}", "x",\n  "--attn-mask-bits"`,
        ),
      );
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.message).toContain(`runtime.args carries ${flag}`);
    }
  });
  test("a K bias no tier passes is refused as a dead element", () => {
    // every tier in q8_0 K: the bias rides a q4_0 K only, so it would never reach the engine
    const r = parseHeadToml(real.replace('\nk = "q4_0"', '\nk = "q8_0"')); // [cache]'s, not the draft's
    expect(r.ok).toBe(false);
    if (!r.ok)
      expect(r.message).toContain("cache.mean_center is declared but no tier serves a q4_0 K");
  });
});

describe("evidence.md", () => {
  // the geometry table restates what tierNeedMiB computes, and a restatement drifts: every cell of
  // it went stale the moment the verify-row charge landed, while the prose above it described the
  // charge (2026-09-21). The table is held against the constants here rather than by hand.
  const num = (cell: string) => Number(cell.replace(/,/g, "").match(/\d+/)![0]);

  test("every cell of the geometry table is what the head's own constants charge at its n_max and at 8", () => {
    const r = parseHeadToml(real);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const at = (n: number) => ({ ...r.value, speculative: { ...r.value.speculative!, n_max: n } });
    const served = r.value.speculative!.n_max;
    expect(evidence).toContain(`needs by the constants (n_max ${served} / 8)`); // the column names the n_max the head serves
    const body = evidence.slice(evidence.indexOf("| card | VRAM |")).split("\n").slice(2); // past the header and its rule
    const end = body.findIndex((line) => !line.startsWith("|")); // this table only, not the next one
    const rows = body.slice(0, end).map((line) => line.split("|").slice(1, -1));
    expect(rows.length).toBe(r.value.geometry.tiers.length); // every tier has a row, and no row has no tier
    // one comparison over the whole table, so a failure names every wrong cell at once
    const written = [];
    const charged = [];
    for (const cells of rows) {
      const [card, , slots, ctx, , needs] = cells as [
        string,
        string,
        string,
        string,
        string,
        string,
      ];
      const tier = r.value.geometry.tiers.find((t) => t.slots === num(slots) && t.ctx === num(ctx));
      expect(
        tier,
        `${card.trim()}: no tier with slots=${num(slots)} ctx=${num(ctx)}`,
      ).toBeDefined();
      const [atServed, eight] = needs.split("/").map(num) as [number, number];
      written.push(`${card.trim()}: ${atServed} / ${eight}`);
      charged.push(
        `${card.trim()}: ${tierNeedMiB(at(served), tier!)} / ${tierNeedMiB(at(8), tier!)}`,
      );
    }
    expect(written).toEqual(charged);
  });
  test("every cell of the windows-per-format table is what the charge leaves at the tier's floor", () => {
    const r = parseHeadToml(real);
    if (!r.ok) throw new Error(r.message);
    const h = r.value;
    const lines = evidence.slice(evidence.indexOf("| tier | served now |")).split("\n");
    const cells = (line: string) =>
      line
        .split("|")
        .slice(1, -1)
        .map((cell) => cell.trim());
    const formats = cells(lines[0]!)
      .slice(2)
      .map((pair) => pair.split(" / ") as [KvType, KvType]);
    const body = lines.slice(2);
    const rows = body
      .slice(
        0,
        body.findIndex((line) => !line.startsWith("|")),
      )
      .map(cells);
    // the largest pool a tier holds within its floor in these formats (a q8_0 state), in trained windows
    const windows = (tier: Tier, k: KvType, v: KvType) => {
      const fits = (ctx: number) =>
        tierNeedMiB(h, { ...tier, ctx, cache: { k, v, s: "q8_0" } }) <= tier.min_vram_mib;
      let [lo, hi] = [0, 1 << 26];
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (fits(mid)) lo = mid;
        else hi = mid - 1;
      }
      return lo / h.context.model;
    };
    // the windows a pool holds, cut to two places (never rounded up past what it holds)
    const span = (xs: number[], cut?: boolean) => {
      const [lo, hi] = [Math.min(...xs), Math.max(...xs)].map((x) =>
        cut ? (Math.floor(x * 100) / 100).toFixed(2) : String(x),
      );
      return lo === hi ? lo! : `${lo}–${hi}`;
    };
    // a row names its tier by the floor it starts with; the one naming none is every tier the others leave
    const named = rows.filter((row) => /^≥/.test(row[0]!)).map((row) => num(row[0]!));
    const written = [];
    const charged = [];
    for (const row of rows) {
      const tiers = h.geometry.tiers.filter((t) =>
        /^≥/.test(row[0]!) ? t.min_vram_mib === num(row[0]!) : !named.includes(t.min_vram_mib),
      );
      expect(tiers.length, `${row[0]}: no tier`).toBeGreaterThan(0);
      written.push(`${row[0]}: ${row.slice(1).join(" | ")}`);
      charged.push(
        `${row[0]}: ${[
          span(tiers.map((t) => t.ctx / h.context.model)),
          ...formats.map(([k, v]) =>
            span(
              tiers.map((t) => windows(t, k, v)),
              true,
            ),
          ),
        ].join(" | ")}`,
      );
    }
    expect(written).toEqual(charged);
  });
});

describe("loadHead", () => {
  test("resolves the head's paths under heads/ and local/", async () => {
    const p = fakePorts();
    const layout = layoutAt("/r");
    putHead(p.fs, "/r", real);
    const r = await loadHead(p.fs, layout, "bonsai-2-27b");
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value.path("assets/chat-template.jinja")).toBe(
      "/r/heads/bonsai-2-27b/assets/chat-template.jinja",
    );
    expect(r.value.servedPath).toBe(
      "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
    );
    expect(r.value.draftPath).toBeUndefined(); // the head is in the pack
    p.fs.put("/r/heads/bonsai-2-27b/head.toml", withSidecarDraft(real));
    const sidecar = await loadHead(p.fs, layout, "bonsai-2-27b");
    expect(sidecar.ok && sidecar.value.draftPath).toBe(
      "/r/local/packs/bonsai-2-27b/Bonsai-2-27B-DFlash2-Q8_0.gguf",
    );
  });
  test("a clone without the private adapter serves the public pack, the source with our draft head; the derived pack keeps the steps", async () => {
    const p = fakePorts();
    const layout = layoutAt("/r");
    p.fs.put("/r/heads/bonsai-2-27b/head.toml", real); // not even the public head yet: `rig fetch` gets it
    const clone = await loadHead(p.fs, layout, "bonsai-2-27b");
    if (!clone.ok) throw new Error(clone.message);
    expect(clone.value.undrived).toContain("assets/lora/bonsai-abliterate-lora.gguf");
    expect(clone.value.undrived).toContain("serving the public pack");
    expect(clone.value.derive?.map((step) => step.kind)).toEqual(["draft-head-splice"]);
    expect(clone.value.served).toEqual({
      file: "Ternary-Bonsai-2-27B-PQ2_0-MTP-r2.gguf",
      sha256: "0e5524befc7cf0c446a2d003bc5c71c39bd4816fdbb49dc38c248eab0e08d4d1",
      bytes: 7657489728,
    });
    expect(clone.value.servedPath).toBe(
      "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-r2.gguf",
    );
    expect(clone.value.declaredPublic).toEqual({
      path: "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-r2.gguf",
      sha256: "0e5524befc7cf0c446a2d003bc5c71c39bd4816fdbb49dc38c248eab0e08d4d1",
    });
    // the public head is fetched into local/ (an upgrade replaces heads/ wholesale), the adapter
    // stays where `torad model pull` puts it
    const [splice] = clone.value.derive ?? [];
    expect(splice && clone.value.assetPath(splice)).toBe(
      "/r/local/packs/bonsai-2-27b/bonsai-2-27b-mtp-r2.gguf",
    );
    putHead(p.fs, "/r", real);
    const full = await loadHead(p.fs, layout, "bonsai-2-27b");
    if (!full.ok) throw new Error(full.message);
    expect(full.value.undrived).toBeUndefined();
    expect(full.value.derive?.map((step) => full.value.assetPath(step))).toEqual([
      "/r/local/packs/bonsai-2-27b/bonsai-2-27b-mtp-r2.gguf",
      "/r/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf",
    ]);
    await p.fs.remove("/r/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf");
    // the declared served pin survives going undrived: a verify asked to check the derived
    // path by name still has the pin it was published against, even though `served` itself
    // is now the public pack
    expect(clone.value.declaredServed).toEqual({
      path: "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      sha256: "389b6d3caefc1fa15eb94d82008562919142ec10051fc0109b464104ee52cae2",
    });
    p.fs.put(
      "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      "",
    );
    const derived = await loadHead(p.fs, layout, "bonsai-2-27b");
    expect(derived.ok && derived.value.undrived).toBeUndefined();
    expect(derived.ok && derived.value.servedPath).toBe(
      "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
    );
    expect(derived.ok && derived.value.declaredServed).toEqual({
      path: "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      sha256: "389b6d3caefc1fa15eb94d82008562919142ec10051fc0109b464104ee52cae2",
    });
  });
  test("a directory name that does not match the head's name is refused", async () => {
    const p = fakePorts();
    const layout = layoutAt("/r");
    p.fs.put("/r/heads/other/head.toml", real);
    const r = await loadHead(p.fs, layout, "other");
    expect(r.ok).toBe(false);
  });
  test("an EACCES (or any error beyond ENOENT) on the adapter or the derived pack refuses loudly, never silently falls back to undrived", async () => {
    const p = fakePorts();
    const layout = layoutAt("/r");
    p.fs.put("/r/heads/bonsai-2-27b/head.toml", real); // no adapter on disk, like a stranger's clone
    p.fs.deny("/r/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf", "EACCES");
    const denied = await loadHead(p.fs, layout, "bonsai-2-27b");
    expect(denied.ok).toBe(false);
    if (denied.ok) return;
    expect(denied.message).toContain("assets/lora/bonsai-abliterate-lora.gguf");
    expect(denied.message).toContain("EACCES");
    expect(denied.message).not.toContain("undrived"); // never reaches the fallback wording at all
    // the same for the derived pack itself, once the adapter question is answerable
    p.fs.denied.clear();
    p.fs.deny(
      "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      "EIO",
    );
    const deniedServed = await loadHead(p.fs, layout, "bonsai-2-27b");
    expect(deniedServed.ok).toBe(false);
    if (deniedServed.ok) return;
    expect(deniedServed.message).toContain("EIO");
    // ENOTDIR is the same case as ENOENT (a parent that would have to be a directory is a file
    // instead, so the path cannot exist either way): still "absent", not a refusal
    p.fs.denied.clear();
    p.fs.deny(
      "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      "ENOTDIR",
    );
    const notdir = await loadHead(p.fs, layout, "bonsai-2-27b");
    expect(notdir.ok).toBe(true);
    if (!notdir.ok) return;
    expect(notdir.value.undrived).toContain("assets/lora/bonsai-abliterate-lora.gguf");
  });
});
