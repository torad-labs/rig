// head.toml — everything rig knows about one model, as data. A head is a directory under heads/
// holding this file, an assets/ folder the runtime args may reference, a gates.toml with the
// head's probes, and evidence.md with the measurements behind its numbers. Adding a model is
// adding a directory; if it needs a code path, that is a bug in rig, not a feature of the head.
//
// What every head says: the pack it serves ([source]: one GGUF, or a split GGUF's shards), the
// window a conversation gets ([context]), how it is served on each kind of machine ([[profiles]]:
// how many cards, how the model spans them, slots and pool) and its server args ([runtime]).
// Everything else is a capability a head declares when it has it: a derivation of the served pack
// from the source ([[derive]], [served], [public]), a draft for speculative decoding
// ([speculative]), cache formats other than the engine's f16 ([cache]), and the memory model its
// single-card profiles are checked against ([sizing]).

import { forkSha, PrebuiltSchema } from "@rig/engine";
import * as v from "valibot";
import {
  draftKvBytesPerToken,
  INDEXER_TYPES,
  KV_TYPES,
  kvBytesPerToken,
  profileCache,
  STATE_TYPES,
  stateMiBPerCopy,
} from "./cache-formats.ts";
import renderedFlagAliases from "./rendered-flag-aliases.json";

const hex = (digits: number) =>
  v.pipe(v.string(), v.regex(new RegExp(`^[0-9a-f]{${digits}}$`), `a ${digits}-hex-digit hash`));
const int = v.pipe(v.number(), v.integer());
const posInt = v.pipe(int, v.minValue(1));
const relPath = v.pipe(
  v.string(),
  v.regex(/^(?!\/)(?!.*\.\.)[^\0]+$/, "a path relative to the head directory, no .."),
);
const hfRepo = v.pipe(
  v.string(),
  v.regex(/^[\w.-]+\/[\w.-]+$/, "a Hugging Face repo id owner/name"),
);
const httpsUrl = v.pipe(v.string(), v.regex(/^https:\/\/\S+$/, "an https:// URL"));
// A derive step's asset with a url is public: `rig fetch` fetches it, by sha256, into the head's
// packs directory (local/, which an upgrade never touches), and the path is relative to that
// directory. Without one it is private: fetched out of band into the head's own directory.
const assetUrl = { url: v.optional(httpsUrl) };

/** a file rig writes, pinned by its bytes: its path (in the repo it is fetched from, and under the head's packs
 *  directory), its sha256, and its size, which `rig up` checks the disk has room for before it fetches */
const PinnedFileSchema = v.strictObject({ file: relPath, sha256: hex(64), bytes: posInt });
export type PinnedFile = v.InferOutput<typeof PinnedFileSchema>;

/** the graph slots the engine keeps in every context beside its compute buffer (llama_context::graph_slots_max,
 *  engine 9ddd463), each holding graphs of up to GRAPH_SLOT_ROWS rows (graph_slot_max_tokens): a slot's buffers grow
 *  to the graphs it holds, so each is charged the context's largest graph of that many rows */
export const GRAPH_SLOTS = 3;
export const GRAPH_SLOT_ROWS = 32;

// what one graph slot of a context holds at most: its largest graph of GRAPH_SLOT_ROWS rows over every sequence and
// on the full pool, which the engine measures and logs at load ("graph slot buffer size <= X MiB", engine 87a3596)
const GraphSlotSchema = v.strictObject({
  mib: v.pipe(v.number(), v.minValue(0)), // its fixed part
  per_seq_mib: v.pipe(v.number(), v.minValue(0)), // per sequence the graph spans: the profile's slots, at most GRAPH_SLOT_ROWS
  bytes_per_token: v.pipe(int, v.minValue(0)), // per pooled token: the attention mask at GRAPH_SLOT_ROWS rows
});
type GraphSlot = v.InferOutput<typeof GraphSlotSchema>;

/** the CUDA graph executables the engine keeps per context at most (GGML_CUDA_GRAPH_MAX, engine d0f8bae) unless a
 *  profile names its own cap: one a graph shape and graph slot, each holding device memory beside the buffers above, a
 *  new one evicting the least recently used. Serve and the gate pass the profile's cap to the engine, so what a profile
 *  is charged is what the process keeps */
export const CUDA_GRAPHS_MAX = 8;
// the most device memory one CUDA graph of a context took to instantiate, as the engine logs it ("the largest took X MiB
// to instantiate", engine d0f8bae): charged CUDA_GRAPHS_MAX times
const cudaGraphMiB = v.pipe(v.number(), v.minValue(0));

// A draft the served pack verifies (speculative decoding): exact output, more accepted tokens per weight read. What
// the server is told: the draft type, how deep a round drafts and when it stops early, the draft's own cache formats
// (-ctkd/-ctvd; absent, the engine's) and its other flags. What it costs on a card is [sizing.draft]'s.
const speculativeConfig = {
  n_max: v.pipe(posInt, v.maxValue(64)), // --spec-draft-n-max: at most the head's block size
  // --spec-draft-p-min: a round stops drafting once the head's top-1 probability (over its top ten, the
  // engine's draft sampler) falls under this; absent, every round emits n_max whatever the confidence
  // below 1: the engine's top-1 does reach 1.0f — outright for a single candidate, and by rounding
  // once the other nine fall ~19 nats behind (llama-sampler.cpp, the size == 1 branch and the
  // p /= sum_cum pass) — so p_min = 1 is not "never drafts" but "drafts only at float saturation",
  // a head charged for its weights, KV and rollback snapshots on every profile to draft almost never
  p_min: v.optional(
    v.pipe(
      v.number(),
      v.minValue(0),
      v.check((p) => p < 1, "p_min must be below 1"),
    ),
  ),
  // --spec-draft-chain-p-min (engine e785bcc): a round stops drafting once the product of the chain's top-1
  // probabilities is under this, a position's expected yield; absent, only p_min gates. The engine applies it
  // to draft-simple, draft-eagle3 and draft-mtp and ignores it elsewhere, so the schema refuses it for the
  // other types (the check on SpeculativeSchema below). Below 1 for the same reason as p_min: a product of
  // top-1 probabilities reaches 1.0f when every position saturates, so 1 buys a head that drafts almost never
  chain_p_min: v.optional(
    v.pipe(
      v.number(),
      v.minValue(0),
      v.check((p) => p < 1, "chain_p_min must be below 1"),
    ),
  ),
  // the draft's own attention cache, rendered by serve (-ctkd/-ctvd)
  cache: v.optional(v.strictObject({ k: v.picklist(KV_TYPES), v: v.picklist(KV_TYPES) })),
  args: v.optional(v.array(v.string()), []), // the draft's other flags (a draft vocabulary …)
};
export const SpeculativeSchema = v.pipe(
  v.variant("type", [
    // a sidecar draft: its own file beside the pack, pinned like the pack's bytes
    v.strictObject({
      type: v.picklist(["draft-dflash", "draft-simple", "draft-eagle3", "draft-dspark"]), // --spec-type
      repo: hfRepo,
      rev: hex(40),
      file: relPath,
      sha256: hex(64),
      bytes: posInt, // its size, which `rig up` checks the disk has room for before it fetches
      ...speculativeConfig,
    }),
    // a multi-token-prediction head carried inside the served pack: pinned by the pack's own sha
    v.strictObject({
      type: v.literal("draft-mtp"),
      ...speculativeConfig,
    }),
  ]),
  // the engine applies the chain cutoff to draft-simple, draft-eagle3 and draft-mtp and ignores it elsewhere
  v.check(
    (s) =>
      s.chain_p_min === undefined ||
      s.type === "draft-simple" ||
      s.type === "draft-eagle3" ||
      s.type === "draft-mtp",
    "chain_p_min: the engine applies the chain cutoff to draft-simple, draft-eagle3 and draft-mtp only; this draft type keeps p_min alone",
  ),
);

/** the draft types whose drafts the engine verifies by rolling the recurrent state back n_max
 *  tokens (common_params_speculative::need_n_rs_seq in the engine): each slot keeps n_max more
 *  copies of its state */
const rollsBackRecurrentState = new Set([
  "draft-mtp",
  "draft-eagle3",
  "draft-dflash",
  "draft-dspark",
]);

export const DeriveSchema = v.variant("kind", [
  // Flip ternary digits on the PQ2_0 lattice to remove a direction (the refusal direction of a
  // rank-1 adapter); the served file is byte-for-byte reproducible from source + adapter + args.
  v.strictObject({
    kind: v.literal("pq2-lattice-ablation"),
    lora: relPath,
    lora_sha256: hex(64),
    lora_bytes: v.optional(posInt), // its size; required with a url (`rig up` checks the disk has room for what it fetches)
    blocks: v.pipe(v.string(), v.regex(/^\d+-\d+$/, "an inclusive block range like 15-63")),
    rows: posInt,
    lambda: v.pipe(v.number(), v.minValue(0), v.maxValue(1)),
    row_cap: v.pipe(v.number(), v.minValue(0), v.maxValue(1)),
    ...assetUrl,
  }),
  // Write a draft head over the pack's own (a retrained MTP block): every tensor in the asset
  // replaces the pack's tensor of the same name and shape, byte for byte. Of the same type nothing
  // else moves; of another type (a requantized head) the tensors after it move to their new
  // offsets and every other byte stays. The served file is reproducible from source + asset like
  // any other step.
  v.strictObject({
    kind: v.literal("draft-head-splice"),
    head: relPath,
    head_sha256: hex(64),
    head_bytes: v.optional(posInt), // its size; required with a url (`rig up` checks the disk has room for what it fetches)
    ...assetUrl,
  }),
]);

// The formats a profile may serve in place of the head's own, each on its own (cache-formats.ts)
const CacheOverrideSchema = v.strictObject({
  k: v.optional(v.picklist(KV_TYPES)),
  v: v.optional(v.picklist(KV_TYPES)),
  s: v.optional(v.picklist(STATE_TYPES)),
  idx: v.optional(v.picklist(INDEXER_TYPES)),
});

/** how llama-server spans a model over several cards (-sm): whole layers per card, rows of each matrix, or each
 *  layer's tensors split so every card streams its share of every token (the engine's meta backend) */
export const SPLIT_MODES = ["layer", "row", "tensor"] as const;
export type SplitMode = (typeof SPLIT_MODES)[number];

// One way to serve the head, on a kind of machine: how many cards (all of one compute capability) and how much VRAM
// each must have free for it, how the model spans them, and the slots and pool it serves there. A machine gets the
// first profile its cards satisfy, so a head lists them most capable first. These are measurements, not a formula:
// [sizing], when the head declares it, is what each single-card profile is checked against at load.
export const ProfileSchema = v.strictObject({
  devices: v.optional(v.pipe(posInt, v.maxValue(16)), 1),
  min_vram_mib: posInt, // on each of its cards
  split: v.optional(v.picklist(SPLIT_MODES)), // required over several cards, meaningless on one
  slots: v.pipe(posInt, v.maxValue(64)),
  ctx: posInt,
  // what the driver holds on the card beside the buffers and CUDA graphs [sizing] charges: a card's peak with CUDA
  // graphs off less its charge, which follows the card (its size), not the geometry; charged in profileNeedMiB
  runtime_mib: v.optional(v.pipe(int, v.minValue(0))),
  speculative: v.optional(v.boolean()), // load the head's draft on this profile; unset = yes when the head declares one
  cache: v.optional(CacheOverrideSchema), // the cache formats this profile serves in place of the head's [cache] ones
  // the most CUDA graphs a context keeps (GGML_CUDA_GRAPH_MAX), 0 for no cap; unset, CUDA_GRAPHS_MAX. A tensor split
  // cycles through one graph per all-reduce step on each card, far more than 8, and a cap below its working set
  // evicts every graph before its reuse
  cuda_graphs: v.optional(v.pipe(int, v.minValue(0))),
});

// The memory model of the head on one card: every byte a profile's slots and pool cost, from constants the engine logs
// at load. Declared, it holds each single-card profile to what the card can hold; a head without it serves its
// profiles as measured.
const SizingSchema = v.strictObject({
  weights_mib: posInt,
  compute_mib: v.pipe(int, v.minValue(0)), // the compute buffer's fixed part, at no output rows
  compute_bytes_per_token: v.pipe(int, v.minValue(0)), // the compute buffer's share per pooled token (the attention mask)
  // per output row: the engine sizes the target compute buffer for n_outputs_max = slots × (1 +
  // n_max) rows, n_max 0 on a profile that does not draft (common_speculative_get_output_limits);
  // 167.22 MiB at 12 rows, 239.22 at 36 on the 5080, 2026-09-21
  compute_per_output_row_mib: v.pipe(int, v.minValue(0)),
  graph_slot: GraphSlotSchema, // one of the target context's graph slots, its output rows at compute_per_output_row_mib
  cuda_graph_mib: cudaGraphMiB, // the target context's largest CUDA graph
  kv_elements_per_token: posInt, // K (and V) elements one pooled token adds over the attention layers
  state_elements_per_copy: v.optional(v.pipe(int, v.minValue(0)), 0), // one sequence's recurrent state, the part -cts narrows
  state_fixed_mib: v.optional(v.pipe(v.number(), v.minValue(0)), 0), // the part of it that stays f32 whatever -cts says
  // the draft's footprint, charged to every profile that loads it: its weights, a fixed overhead (its own cache, the
  // target's larger verification graph), its compute buffer per pooled token (this engine has no --ctx-size-draft, so
  // the draft's context follows the target's -c) and, for a draft the engine rolls the recurrent state back for, n_max
  // state snapshots per slot
  draft: v.optional(
    v.strictObject({
      weights_mib: posInt, // its model buffer (an in-pack head: its share of the pack's)
      overhead_mib: v.pipe(int, v.minValue(0)), // fixed: its cache, the target's larger graph, the pool — measured
      bytes_per_token: v.pipe(int, v.minValue(0)), // per pooled token beyond its cache: its compute buffer
      graph_slot: GraphSlotSchema, // one of the draft context's graph slots, its output row a sequence in per_seq_mib
      cuda_graph_mib: cudaGraphMiB, // the draft context's largest CUDA graph
      kv_elements_per_token: posInt, // K (and V) elements one pooled token adds to its cache
    }),
  ),
});

export const HeadSchema = v.strictObject({
  name: v.pipe(v.string(), v.regex(/^[a-z0-9][a-z0-9.-]*$/, "lowercase, digits, dots, dashes")),
  title: v.string(),
  port: v.pipe(int, v.minValue(1024), v.maxValue(65535)),
  // the pack as published, pinned to a revision and its bytes: one GGUF, or a split GGUF's shards in order (the first is
  // the one the server loads; the engine finds the rest beside it)
  source: v.strictObject({
    repo: hfRepo,
    rev: hex(40),
    files: v.pipe(v.array(PinnedFileSchema), v.minLength(1)),
  }),
  derive: v.optional(v.pipe(v.array(DeriveSchema), v.minLength(1))), // [[derive]] steps, applied in order
  served: v.optional(PinnedFileSchema), // the pack the [derive] steps produce from the source
  // the pack the public steps alone produce (the leading [[derive]] steps with a url): what a
  // machine without the private assets serves, instead of the source pack
  public: v.optional(PinnedFileSchema),
  speculative: v.optional(SpeculativeSchema),
  // the fork commit this head serves on, in place of engine/engine.toml's (a model the pin does not run yet), and the
  // builds published for it; the cards, caches, compilers and CUDA runtime stay engine.toml's (headEngine)
  engine: v.optional(
    v.strictObject({
      sha: forkSha,
      prebuilt: v.optional(v.array(PrebuiltSchema)),
    }),
  ),
  context: v.strictObject({ model: posInt, advertise: posInt }), // the window one conversation gets, and what it is told
  // the formats of the model's caches every profile serves unless it names its own; serve renders --cache-type-k/-v,
  // -cts and -ctki from these, never runtime.args. The attention cache is the engine's f16 unless named; the recurrent state is
  // passed only when named (a model without one has nothing to narrow)
  cache: v.optional(
    v.strictObject({
      k: v.optional(v.picklist(KV_TYPES), "f16"),
      v: v.optional(v.picklist(KV_TYPES), "f16"),
      s: v.optional(v.picklist(STATE_TYPES)),
      idx: v.optional(v.picklist(INDEXER_TYPES)), // the sparse-attention indexer's cache (-ctki), passed only when named
      mean_center: v.optional(relPath), // the K bias (--kv-mean-center), passed on a q4_0 K only
    }),
    { k: "f16", v: "f16" },
  ),
  profiles: v.pipe(v.array(ProfileSchema), v.minLength(1)),
  sizing: v.optional(SizingSchema),
  runtime: v.strictObject({
    args: v.optional(v.array(v.string()), []), // how the pack is loaded: template, checkpoint grid, the engine's other switches ([cache] holds the formats)
    sampling: v.optional(v.array(v.string()), []), // the model's own sampling; llama-server does not read it from the GGUF
    extra: v.optional(v.array(v.string()), []), // anything else the unit passes (log level)
    lens: v.optional(
      v.strictObject({
        enabled: v.boolean(),
        args: v.array(v.string()),
      }),
    ), // live diagnostics only; never added to gate legs
  }),
  client: v.strictObject({
    rejects_reasoning_effort: v.boolean(),
    slot_pinning: v.boolean(),
    any_model_id: v.boolean(),
  }),
});

export type HeadConfig = v.InferOutput<typeof HeadSchema>;
export type Derive = v.InferOutput<typeof DeriveSchema>;
export type Profile = v.InferOutput<typeof ProfileSchema>;
export type Speculative = v.InferOutput<typeof SpeculativeSchema>;
export type SidecarDraft = Extract<Speculative, { file: string }>;
export type Sizing = v.InferOutput<typeof SizingSchema>;

const sized = (bytes: number | undefined) => (bytes ? { bytes } : {});

/** the file a derive step reads, pinned by its sha256: public (fetched from url into the packs
 *  directory) or private (fetched out of band into the head's directory, never in git) */
export function deriveAsset(step: Derive): {
  path: string;
  sha256: string;
  url?: string;
  bytes?: number;
} {
  const url = step.url ? { url: step.url } : {};
  switch (step.kind) {
    case "pq2-lattice-ablation":
      return { path: step.lora, sha256: step.lora_sha256, ...url, ...sized(step.lora_bytes) };
    case "draft-head-splice":
      return { path: step.head, sha256: step.head_sha256, ...url, ...sized(step.head_bytes) };
  }
}

/** the leading [[derive]] steps whose assets are public: the steps `[public]` pins the output of */
export function publicSteps(derive: readonly Derive[]): Derive[] {
  const end = derive.findIndex((step) => !step.url);
  return derive.slice(0, end === -1 ? derive.length : end);
}

/** the draft's sidecar file pin, when it has one (an in-pack head has none) */
export function draftSidecar(speculative: Speculative): SidecarDraft | undefined {
  return "file" in speculative ? speculative : undefined;
}

/** whether a profile loads the head's draft: declared, and not opted out on this profile */
export function profileSpeculates(
  head: Pick<HeadConfig, "speculative">,
  profile: Pick<Profile, "speculative">,
): boolean {
  return !!head.speculative && (profile.speculative ?? true);
}

/** flags serve and the gates render themselves from the head's typed fields and the machine, in
 *  the spelling rig renders; a copy in a free-form arg list would win (llama-server takes the last
 *  occurrence) over the value the profile check charged, silently */
export const RENDERED = [
  "-m",
  "-ngl",
  "--jinja",
  "-fa",
  "-c",
  "-np",
  "--kv-unified",
  "--cache-ram",
  "--no-cache-idle-slots",
  "--metrics",
  "--host",
  "--port",
  "--spec-type",
  "--spec-draft-n-max",
  "--spec-draft-p-min",
  "--spec-draft-chain-p-min",
  "-md",
  "-ngld",
  "--cache-type-k",
  "--cache-type-v",
  "-cts",
  "--kv-mean-center",
  "-ctkd",
  "-ctvd",
  "-sm",
  "-ts",
] as const;

/** flags serve renders only from a head's own config, on an engine newer than the default pin, which does not declare
 *  them: recorded in each head engine's block of the manifest, never in the default pin's flat table. -ctki is a
 *  [cache] idx (cacheArgv) */
export const RENDERED_BY_HEAD_ENGINE = ["-ctki"] as const;

/** every spelling the pinned engine accepts for a rendered flag (--parallel for -np, --ctx-size for
 *  -c …), from its arg table: rendered-flag-aliases.json, regenerated by tools/rendered-flag-aliases.ts;
 *  a test holds it at the pin and equal to the table where the source is on the box */
export const RENDERED_FLAGS: readonly string[] = [
  ...new Set(
    [renderedFlagAliases.aliases, ...Object.values(renderedFlagAliases.engines)].flatMap((block) =>
      Object.values(block as Record<string, string[]>).flat(),
    ),
  ),
]; // a head engine's own flags (RENDERED_BY_HEAD_ENGINE) too, so a free-form copy of one is refused like the rest

/** whether `flag` is one serve renders, under any spelling the engine reads: it takes every "_" in a "--" flag as
 *  "-" before its table lookup (common/arg.cpp, parse_cli_args), so --ctx_size is --ctx-size to it, and to this */
export function renderedFlag(flag: string): boolean {
  return RENDERED_FLAGS.includes(flag.startsWith("--") ? flag.replaceAll("_", "-") : flag);
}

/** Invariants the schema cannot express. Returns every violation, not the first. */
export function headInvariants(head: HeadConfig): string[] {
  return [
    ...renderedFlagViolations(head),
    ...packViolations(head),
    ...profileViolations(head),
    ...sizingViolations(head),
  ];
}

/** the free-form lists carry no flag serve renders itself */
function renderedFlagViolations(head: HeadConfig): string[] {
  const violations: string[] = [];
  const lists: Array<[string, string[]]> = [
    ["runtime.args", head.runtime.args],
    ["runtime.extra", head.runtime.extra],
    ["runtime.sampling", head.runtime.sampling],
    ["runtime.lens.args", head.runtime.lens?.args ?? []],
    ["speculative.args", head.speculative?.args ?? []],
  ];
  for (const [name, list] of lists) {
    for (const flag of list) {
      if (renderedFlag(flag)) {
        violations.push(
          `${name} carries ${flag}, which serve renders from the head's typed fields and the machine; the copy would win silently`,
        );
      }
    }
  }
  if (head.runtime.lens?.enabled && !head.runtime.lens.args.includes("--lens-layers")) {
    violations.push(
      "runtime.lens.enabled = true but its args name no --lens-layers: the unit would be identical to lens off",
    );
  }
  return violations;
}

/** the packs: the source, and when the head derives one, the served and public packs the steps produce */
function packViolations(head: HeadConfig): string[] {
  const violations: string[] = [];
  const { source, served, context } = head;
  const derive = head.derive ?? [];
  if (head.derive && !served)
    violations.push("[derive] steps need a [served] pin for the pack they produce");
  if (!head.derive && served)
    violations.push("[served] pins a derived pack but the head declares no [derive] step");
  if (head.derive && source.files.length > 1)
    violations.push(
      `[derive] steps edit one GGUF, and the source is split over ${source.files.length} files`,
    );
  if (served && source.files.some((file) => file.sha256 === served.sha256))
    violations.push("a [derive] step must produce a different file than source");
  for (const step of derive) {
    const asset = deriveAsset(step);
    if (asset.url && !asset.bytes) {
      violations.push(
        `the [derive] asset ${asset.path} has a url but no size: \`rig up\` checks the disk has room for what it fetches`,
      );
    }
  }
  const leading = publicSteps(derive).length;
  if (derive.slice(leading).some((step) => step.url)) {
    violations.push(
      "a [derive] step with a url follows a private one: the public steps come first, so [public] is the prefix every machine can derive",
    );
  }
  const mixed = leading > 0 && leading < derive.length;
  if (mixed && !head.public) {
    violations.push(
      "[derive] mixes public and private steps but declares no [public]: a machine without the private assets would serve the source pack",
    );
  }
  if (head.public && !mixed) {
    violations.push(
      "[public] needs public [derive] steps (a url) followed by private ones: otherwise the served pack is the public one",
    );
  }
  const pub = head.public;
  if (
    pub &&
    [...source.files, ...(served ? [served] : [])].some(
      (p) => p.file === pub.file || p.sha256 === pub.sha256,
    )
  ) {
    violations.push(
      "[public] must be its own file and bytes, neither the source nor the served pack",
    );
  }
  const names = source.files.map((file) => file.file);
  if (new Set(names).size !== names.length) violations.push("[source] lists a file twice");
  if (context.advertise > context.model) {
    violations.push(
      `context.advertise (${context.advertise}) exceeds context.model (${context.model})`,
    );
  }
  return violations;
}

/** the profiles: a split over several cards, ordered so the first a machine satisfies is the most it can serve */
function profileViolations(head: HeadConfig): string[] {
  const violations: string[] = [];
  const { profiles, context } = head;
  if (head.cache.mean_center && !profiles.some((p) => profileCache(head, p).k === "q4_0")) {
    violations.push(
      "cache.mean_center is declared but no profile serves a q4_0 K, the only one it is passed with: it would never reach the engine",
    );
  }
  profiles.forEach((profile, index) => {
    const at = `profile ${index + 1} (${describeProfile(profile)})`;
    if (profile.devices > 1 && !profile.split)
      violations.push(`${at} spans ${profile.devices} cards but names no split`);
    if (profile.devices === 1 && profile.split)
      violations.push(`${at} names a split on one card, where there is nothing to split`);
    if (profile.cuda_graphs === 0 && profile.devices === 1 && head.sizing)
      violations.push(`${at} keeps CUDA graphs without a cap, which [sizing] cannot charge`);
    if (profile.speculative === true && !head.speculative)
      violations.push(
        `${at} asks for a draft head (speculative = true) but the head declares no [speculative]`,
      );
    if (profile.ctx < context.model)
      violations.push(
        `${at} pools ${profile.ctx} cells, below context.model ${context.model}: one conversation could not use its window`,
      );
    // a machine gets the first profile it satisfies: one satisfied by every machine an earlier one is never reached
    const shadow = profiles
      .slice(0, index)
      .findIndex((p) => p.devices === profile.devices && p.min_vram_mib <= profile.min_vram_mib);
    if (shadow !== -1)
      violations.push(
        `${at} is never reached: profile ${shadow + 1} takes every machine it would (list the profiles most capable first)`,
      );
  });
  return violations;
}

/** the memory model, when declared: every single-card profile fits its card, and nothing it charges is dead */
function sizingViolations(head: HeadConfig): string[] {
  const violations: string[] = [];
  const { sizing, speculative } = head;
  if (!sizing) {
    for (const profile of head.profiles)
      if (profile.runtime_mib !== undefined)
        violations.push(
          `profile ${describeProfile(profile)} sets runtime_mib, which only [sizing] charges, and the head declares none`,
        );
    return violations;
  }
  if (speculative && !sizing.draft)
    violations.push("[sizing] charges no draft ([sizing.draft]) but the head declares one");
  if (sizing.draft && !speculative)
    violations.push("[sizing.draft] charges a draft the head does not declare");
  if (sizing.draft && speculative && !speculative.cache)
    violations.push(
      "[sizing.draft] charges the draft's cache, and [speculative] names no cache formats to charge it at",
    );
  if (sizing.state_elements_per_copy > 0 && !head.cache.s)
    violations.push(
      "[sizing] charges a recurrent state, and [cache] names no state format (s) to charge it at",
    );
  for (const profile of head.profiles) {
    if (profile.devices > 1) continue; // [sizing] models one card; a split profile is served as measured
    if (profile.runtime_mib === undefined) {
      violations.push(
        `profile ${describeProfile(profile)} sets no runtime_mib, the driver's share [sizing] charges beside the buffers`,
      );
      continue;
    }
    const need = profileNeedMiB(head, sizing, { ...profile, runtime_mib: profile.runtime_mib });
    if (need > profile.min_vram_mib) {
      const draft = profileSpeculates(head, profile)
        ? " + draft weights, overhead, compute and rollback snapshots"
        : "";
      violations.push(
        `profile min_vram ${profile.min_vram_mib} MiB cannot hold slots=${profile.slots} ctx=${profile.ctx}: needs ${need} MiB (KV + weights + compute + graph slots + CUDA graphs + slot state${draft} + the driver's runtime)`,
      );
    }
  }
  return violations;
}

/** a profile as a refusal names it: its cards and floor, its slots and pool */
export function describeProfile(profile: Profile): string {
  const cards = profile.devices > 1 ? `${profile.devices} × ` : "";
  const split = profile.split ? ` ${profile.split} split` : "";
  return `${cards}${profile.min_vram_mib} MiB${split}, ${profile.slots} slots × ${profile.ctx}`;
}

/** MiB a single-card profile needs on the card, from the head's [sizing] constants: the check every such profile
 *  must pass */
export function profileNeedMiB(
  head: HeadConfig,
  sizing: Sizing,
  profile: Profile & { runtime_mib: number },
): number {
  const cache = profileCache(head, profile);
  const state = stateMiBPerCopy(sizing, cache); // one sequence's recurrent state, in the profile's format
  const perToken = (bytes: number) => Math.ceil((profile.ctx * bytes) / 1048576);
  const spec = profileSpeculates(head, profile) ? head.speculative : undefined;
  const draft = spec && sizing.draft;
  const draftMiB =
    spec && draft
      ? draft.weights_mib +
        draft.overhead_mib +
        perToken(draftKvBytesPerToken(draft, spec.cache) + draft.bytes_per_token) +
        profile.slots * sizing.compute_per_output_row_mib * spec.n_max +
        (rollsBackRecurrentState.has(spec.type) ? profile.slots * state * spec.n_max : 0)
      : 0;
  return (
    perToken(kvBytesPerToken(sizing, cache) + sizing.compute_bytes_per_token) +
    sizing.weights_mib +
    sizing.compute_mib +
    profile.slots * sizing.compute_per_output_row_mib + // each slot's own output row, drafting or not
    graphSlotsMiB(head, sizing, profile) +
    cudaGraphsMiB(head, sizing, profile) +
    profile.slots * state +
    draftMiB +
    profile.runtime_mib
  );
}

/** MiB the engine's graph slots can add on a profile: GRAPH_SLOTS in the target's context and in the draft's, each the
 *  context's largest graph of GRAPH_SLOT_ROWS rows, the bound the engine logs at load for each */
export function graphSlotsMiB(head: HeadConfig, sizing: Sizing, profile: Profile): number {
  const rows = (n: number) => Math.min(n, GRAPH_SLOT_ROWS);
  const slot = (s: GraphSlot) =>
    s.mib + s.per_seq_mib * rows(profile.slots) + (profile.ctx * s.bytes_per_token) / 1048576;
  const spec = profileSpeculates(head, profile) ? head.speculative : undefined;
  const draft = spec ? sizing.draft : undefined;
  // the target's graph has as many output rows as a batch of it can: n_outputs_max, slots × (1 + n_max) drafting
  const outputs = rows(profile.slots * (1 + (spec?.n_max ?? 0)));
  const target = slot(sizing.graph_slot) + outputs * sizing.compute_per_output_row_mib;
  return Math.ceil(GRAPH_SLOTS * (target + (draft ? slot(draft.graph_slot) : 0)));
}

/** the most CUDA graphs a context keeps on a profile (GGML_CUDA_GRAPH_MAX), 0 for no cap */
export function profileCudaGraphs(profile: Pick<Profile, "cuda_graphs">): number {
  return profile.cuda_graphs ?? CUDA_GRAPHS_MAX;
}

/** MiB the engine's CUDA graphs can hold on a profile: its cap of executables in the target's context and in the
 *  draft's, each the largest the engine logged instantiating in that context */
export function cudaGraphsMiB(head: HeadConfig, sizing: Sizing, profile: Profile): number {
  const draft = profileSpeculates(head, profile) ? sizing.draft : undefined;
  return Math.ceil(
    profileCudaGraphs(profile) * (sizing.cuda_graph_mib + (draft?.cuda_graph_mib ?? 0)),
  );
}
