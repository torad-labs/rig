// head.toml — everything rig knows about one model, as data. A head is a directory under heads/
// holding this file, an assets/ folder the runtime args may reference, a gates.toml with the
// head's probes, and evidence.md with the measurements behind its numbers. Adding a model is
// adding a directory; if it needs a code path, that is a bug in rig, not a feature of the head.
import * as v from "valibot";
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

// A draft head the served pack verifies (speculative decoding): exact output, more accepted tokens
// per weight read. Its footprint is charged to every tier that loads it — weights, a fixed overhead
// (its own cache, the target's larger verification graph), its compute buffer per pooled token
// (this engine has no --ctx-size-draft, so the draft's context follows the target's -c) and, for a
// draft the engine rolls the recurrent state back for, n_max state snapshots per slot.
const speculativeFootprint = {
  n_max: v.pipe(posInt, v.maxValue(64)), // --spec-draft-n-max: at most the head's block size
  // --spec-draft-p-min: a round stops drafting once the head's top-1 probability (over its top ten, the
  // engine's draft sampler) falls under this; absent, every round emits n_max whatever the confidence
  // below 1: the engine's top-1 does reach 1.0f — outright for a single candidate, and by rounding
  // once the other nine fall ~19 nats behind (llama-sampler.cpp, the size == 1 branch and the
  // p /= sum_cum pass) — so p_min = 1 is not "never drafts" but "drafts only at float saturation",
  // a head charged for its weights, KV and rollback snapshots on every tier to draft almost never
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
  weights_mib: posInt, // its model buffer (an in-pack head: its share of the pack's)
  overhead_mib: v.pipe(int, v.minValue(0)), // fixed: its cache, the target's larger graph, the pool — measured
  bytes_per_token: v.pipe(int, v.minValue(0)), // per pooled token: its compute buffer, its own KV rows
  args: v.optional(v.array(v.string()), []), // the draft's own flags (-ctkd/-ctvd …)
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
      ...speculativeFootprint,
    }),
    // a multi-token-prediction head carried inside the served pack: pinned by the pack's own sha
    v.strictObject({
      type: v.literal("draft-mtp"),
      ...speculativeFootprint,
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

export const TierSchema = v.strictObject({
  min_vram_mib: posInt,
  slots: v.pipe(posInt, v.maxValue(64)),
  ctx: posInt,
  speculative: v.optional(v.boolean()), // load the head's draft on this tier; unset = yes when the head declares one
});

export const HeadSchema = v.strictObject({
  name: v.pipe(v.string(), v.regex(/^[a-z0-9][a-z0-9.-]*$/, "lowercase, digits, dots, dashes")),
  title: v.string(),
  port: v.pipe(int, v.minValue(1024), v.maxValue(65535)),
  gpu: v.optional(v.pipe(int, v.minValue(0)), 0),
  source: v.strictObject({
    repo: hfRepo,
    rev: hex(40),
    file: relPath,
    sha256: hex(64),
    bytes: posInt, // its size: `rig up` checks the disk has room for every file it writes before it fetches
  }),
  served: v.strictObject({ file: relPath, sha256: hex(64), bytes: posInt }),
  // the pack the public steps alone produce (the leading [[derive]] steps with a url): what a
  // machine without the private assets serves, instead of the source pack
  public: v.optional(v.strictObject({ file: relPath, sha256: hex(64), bytes: posInt })),
  derive: v.optional(v.pipe(v.array(DeriveSchema), v.minLength(1))), // [[derive]] steps, applied in order
  speculative: v.optional(SpeculativeSchema),
  context: v.strictObject({ model: posInt, advertise: posInt }),
  geometry: v.strictObject({
    kv_bytes_per_token: posInt,
    compute_bytes_per_token: v.pipe(int, v.minValue(0)), // the compute buffer's share per pooled token (the attention mask)
    weights_mib: posInt,
    state_per_slot_mib: v.pipe(int, v.minValue(0)),
    compute_mib: v.pipe(int, v.minValue(0)), // the compute buffer's fixed part, at no output rows
    // per output row: the engine sizes the target compute buffer for n_outputs_max = slots × (1 +
    // n_max) rows, n_max 0 on a tier that does not draft (common_speculative_get_output_limits);
    // 167.22 MiB at 12 rows, 239.22 at 36 on the 5080, 2026-09-21
    compute_per_output_row_mib: v.pipe(int, v.minValue(0)),
    tiers: v.pipe(v.array(TierSchema), v.minLength(1)),
  }),
  runtime: v.strictObject({
    args: v.array(v.string()), // how the pack is loaded: cache types, template, bias, checkpoint grid
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
export type Tier = v.InferOutput<typeof TierSchema>;
export type Speculative = v.InferOutput<typeof SpeculativeSchema>;
export type SidecarDraft = Extract<Speculative, { file: string }>;

/** the draft's sidecar file pin, when it has one (an in-pack head has none) */
export function draftSidecar(speculative: Speculative): SidecarDraft | undefined {
  return "file" in speculative ? speculative : undefined;
}

/** whether a tier loads the head's draft: declared, and not opted out on this tier */
export function tierSpeculates(
  head: Pick<HeadConfig, "speculative">,
  tier: Pick<Tier, "speculative">,
): boolean {
  return !!head.speculative && (tier.speculative ?? true);
}

/** flags serve and the gates render themselves from the head's typed fields and the machine, in
 *  the spelling rig renders; a copy in a free-form arg list would win (llama-server takes the last
 *  occurrence) over the value the tier check charged, silently */
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
] as const;

/** every spelling the pinned engine accepts for a rendered flag (--parallel for -np, --ctx-size for
 *  -c …), from its arg table: rendered-flag-aliases.json, regenerated by scripts/rendered-flag-aliases.ts;
 *  a test holds it at the pin and equal to the table where the source is on the box */
export const RENDERED_FLAGS: readonly string[] = Object.values(renderedFlagAliases.aliases).flat();

/** Invariants the schema cannot express. Returns every violation, not the first. */
export function headInvariants(head: HeadConfig): string[] {
  const violations: string[] = [];
  const { source, served, context } = head;
  const lists: Array<[string, string[]]> = [
    ["runtime.args", head.runtime.args],
    ["runtime.extra", head.runtime.extra],
    ["runtime.lens.args", head.runtime.lens?.args ?? []],
    ["speculative.args", head.speculative?.args ?? []],
  ];
  for (const [name, list] of lists) {
    for (const flag of list) {
      // the engine reads every "_" in a "--" flag as "-" before its table lookup (common/arg.cpp,
      // parse_cli_args), so --ctx_size is --ctx-size to it, and has to be to this check
      const read = flag.startsWith("--") ? flag.replaceAll("_", "-") : flag;
      if ((RENDERED_FLAGS as readonly string[]).includes(read)) {
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
  const differs =
    served.file !== source.file || served.sha256 !== source.sha256 || served.bytes !== source.bytes;
  if (!head.derive && differs) {
    violations.push("served must equal source when there is no [derive] step");
  }
  if (head.derive && served.sha256 === source.sha256) {
    violations.push("a [derive] step must produce a different file than source");
  }
  const derive = head.derive ?? [];
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
  if (
    head.public &&
    [source, served].some((p) => p.file === head.public?.file || p.sha256 === head.public?.sha256)
  ) {
    violations.push(
      "[public] must be its own file and bytes, neither the source nor the served pack",
    );
  }
  if (context.advertise > context.model) {
    violations.push(
      `context.advertise (${context.advertise}) exceeds context.model (${context.model})`,
    );
  }

  const tiers = head.geometry.tiers;
  const descending = tiers.every((tier, index) => {
    const previous = tiers[index - 1];
    return previous === undefined || tier.min_vram_mib < previous.min_vram_mib;
  });
  if (!descending)
    violations.push("geometry.tiers must be listed from the largest min_vram_mib down");

  for (const tier of tiers) {
    if (tier.speculative === true && !head.speculative) {
      violations.push(
        `tier min_vram ${tier.min_vram_mib} asks for a draft head (speculative = true) but the head declares no [speculative]`,
      );
    }
    const need = tierNeedMiB(head, tier);
    if (need > tier.min_vram_mib) {
      const draft = tierSpeculates(head, tier)
        ? " + draft weights, overhead, compute and rollback snapshots"
        : "";
      violations.push(
        `tier min_vram ${tier.min_vram_mib} MiB cannot hold slots=${tier.slots} ctx=${tier.ctx}: needs ${need} MiB (KV + weights + compute + slot state${draft})`,
      );
    }
    if (tier.ctx < context.model) {
      violations.push(
        `tier ctx ${tier.ctx} is below context.model ${context.model}: one conversation could not use the trained window`,
      );
    }
  }
  return violations;
}

/** MiB a tier needs on the card, from the head's constants: the check every tier must pass. */
export function tierNeedMiB(head: HeadConfig, tier: Tier): number {
  const geometry = head.geometry;
  const perToken = (bytes: number) => Math.ceil((tier.ctx * bytes) / 1048576);
  const draft = tierSpeculates(head, tier) ? head.speculative : undefined;
  const draftMiB = draft
    ? draft.weights_mib +
      draft.overhead_mib +
      perToken(draft.bytes_per_token) +
      tier.slots * geometry.compute_per_output_row_mib * draft.n_max +
      (rollsBackRecurrentState.has(draft.type)
        ? tier.slots * geometry.state_per_slot_mib * draft.n_max
        : 0)
    : 0;
  return (
    perToken(geometry.kv_bytes_per_token + geometry.compute_bytes_per_token) +
    geometry.weights_mib +
    geometry.compute_mib +
    tier.slots * geometry.compute_per_output_row_mib + // each slot's own output row, drafting or not
    tier.slots * geometry.state_per_slot_mib +
    draftMiB
  );
}
