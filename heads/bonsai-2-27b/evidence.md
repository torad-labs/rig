# Ternary Bonsai 2 27B — evidence behind head.toml

Every number in `head.toml` and `gates.toml` was measured; this file says where.

## The pack

`served` is an ablated pack: the refusal edit of Continuum-AI-Corp/OrcaBonsai-27B-Uncensored's
rank-1 adapter (`assets/lora/bonsai-abliterate-lora.gguf`, sha `f1669534…`) baked into the
ternary codes by the PQ2_0 lattice ablation (`[derive]`: blocks 15–63, 512 rows, λ 1.0, row cap
0.10). 98 residual-writer tensors edited, 22,674,442 digits flipped = 0.384 % of the pack,
99.2–99.5 % of the direction removed per tensor; every other tensor byte-identical to the source
and no block scale moved. Reproduced byte for byte by rig's TypeScript derive on 2026-09-20
(sha `e7b99670…`, 74 s) from the numpy reference of 2026-09-19.

Since 2026-09-20 20:19 the source is ProCreations/Ternary-Bonsai-2-27B-MTP @ efffdea (sha
`3cb3f005…`, 866 tensors): Prism's 851 tensors byte for byte plus a 15-tensor MTP head in blk.64
(below). The same derive on it flips the same 22,674,442 digits in the same 98 writers (blocks
15–63; blk.64 is outside the range and Q8_0, not PQ2_0) and publishes `served` at sha `53ab9023…`.

Why baked: llama.cpp never merges a LoRA — it adds two matmuls beside each of the 129 writers —
and that launch count cost decode 91.4 → 85.2 tok/s (−7 %) and prefill 2,323 → 2,254 tok/s. Why
a lattice edit: the projection `W ← W − r (rᵀW)` is a ~1.4 % nudge against a 100 % lattice step
and rounds straight back on a ternary pack, so digits are spent greedily on the rows with the
largest |r_n| (BoldingBuilds' method for this model). Why `row_cap = 0.10`: uncapped, the pass
piles the edit on the few most r-aligned rows (67 % of one row in blk.20.ffn_down) and that
pack doubled words on 6 of 11 fluency prompts against the base's 2; capped, the artifact is gone.

## Gates on the pinned pack (RTX 5070 Ti, engine 60feea0, 2026-09-19/20)

Runs: `Bonsai-demo/runs/2026-09-19/decode-profile-5070ti/baked/` and
`runs/2026-09-20/vendorized-60feea0-sm120/`. Re-run end to end **through rig** on 2026-09-20
against a rig-built `local/engine-builds/60feea0-sm120` — `local/gate-runs/bonsai-2-27b/20260920T082930Z/` (its per-probe results banked under
`evidence/gates/20260920T082930Z/`), six probes, all pass, and the two live probes (sessions,
concurrency) in run `20260920T085039Z` against the serving 5080, banked under
`evidence/gates/20260920T085039Z/`; the numbers in the table below are that run's, and the earlier runs agree
inside their noise (HumanEval base read 113/164 on 2026-09-19 and 112/164 here; decode sat at
79 tok/s on a colder card and 75 here, A-B-A both times).

| probe | base pack | served pack | criterion |
|---|---|---|---|
| refusal (6 prompts, greedy) | refuses 6/6 | refuses 0/6 | base > 0, served < base |
| capability parity (5 prompts, greedy) | — | 5/5 byte-identical to base | all identical |
| fluency (11 prompts × 300 tokens) | doubled 2, repeats 2 | doubled 2, repeats 2 (2,909 tokens) | served ≤ base on both |
| HumanEval-164 pass@1, greedy | 112/164 = 0.683 | 116/164 = 0.707 | served does not lose significantly more problems than it gains (one-sided sign test on the disagreements, p ≥ 0.05; served ≥ base until 2026-09-25) |
| decode A-B-A (3 × 256 greedy, first discarded) | 74.8 / — / 75.2 tok/s | 74.7 tok/s | inside the base's A-A spread + 5 % |
| needle (2 markers, 121,898 tokens, q4_0 K/V + mean-centering; haystack from the earlier corpus) | — | 2/2 retrieved | all retrieved |
| sessions (4 × ~50K tokens, shared pool, 5080) | — | 4/4 own marker, no leak | all own, none foreign |
| concurrency (M = 1/2/4, 5080) | — | 50 / 83 / 131 tok/s aggregate | M=4 ≥ 1.5 × M=1 |

The 2026-09-20 vendorized re-run reproduced 22/22 greedy completions byte for byte against the
previous binary. The llama-bench depth matrix from the same rig run (RTX 5070 Ti, q4_0 K/V, fa on,
3 repeats) is the card's own curve, measured and not judged:

| depth | pp512 tok/s | tg64 tok/s |
|---|---|---|
| 0 | 1737.5 ± 27.8 | 77.3 ± 0.7 |
| 16,384 | 1606.1 ± 30.2 | 74.6 ± 0.0 |
| 65,536 | 1271.0 ± 31.4 | 67.6 ± 0.7 |
| 131,072 | 939.3 ± 18.9 | 61.2 ± 1.0 |
| 261,874 | 633.0 ± 10.8 | 48.8 ± 0.3 |

## Geometry

`kv_bytes_per_token = 18432` (q4_0 K and V, head size 256, the model's KV layout);
`compute_bytes_per_token = 64` — the compute buffer grows 64 B per pooled token: the attention
mask at n_ubatch 512, one bit per cell under `--attn-mask-bits` (engine PR #6; 288.00 → 18.00 MB
in the target graph and 64.00 → 4.00 MB in the draft graph at 294,912 cells, 2026-09-21; as f16
it was 1 KiB per token: 198.28 / 262.28 / 390.28 MiB at 64K / 128K / 256K on the 5070 Ti) over a
fixed part `compute_mib = 114` (167.22 MiB measured at 294,912 on the 5080, lens off, at 12 verify
rows, less the 18 per-token and the rows; 330 was the same buffer with the lens graph and the f16
mask, 618.28 less 288; 134 / 157 were the undrafted 5070 Ti / 5080) plus
`compute_per_output_row_mib = 3` per output row — the engine sizes the buffer for
`n_outputs_max = slots × (1 + n_max)` (n_max 0 on a tier that does not draft;
`common_speculative_get_output_limits`): 167.22 MiB at 12 rows, 239.22 at 36, 3.0 per row, all of
them charged (the first row per slot was left out until 2026-09-23: 12 MiB short on the 5080,
156 charged against the 167.22 measured); `weights_mib = 6900`; `state_per_slot_mib = 44` — one Gated DeltaNet recurrent state in q8_0 (`-cts q8_0`, engine
e3a9015): RS buffer 526.5 MiB at 4 slots × (1 + n_max 2) copies and 1,579.5 MiB at 4 × (1 + 8),
43.875 per copy, linear in n_max (`n_rs_seq`, the copies the engine rolls a rejected draft back
with); 150 in f32 (1,795.5 MiB at n_max 2). Every tier is checked against those constants by
`headInvariants`, plus the draft head's (below) on every tier that loads it, and `head.test.ts`
re-derives each tier's total from the same constants:

| card | VRAM | -np | -c | draft head | needs by the constants (n_max 3 / 8) |
|---|---|---|---|---|---|
| RTX 5080 | 16,303 MiB | 4 | 294,912 (1.125 windows) | MTP | 13,968 / 14,908 (14,012 MiB at peak, 2026-09-25) |
| 16 GB card driving a desktop | ≥ 14,100 MiB for the head | 4 | 262,144 (1 window) | MTP | 13,352 / 14,292 (13,410 MiB at peak, 2026-09-25) |
| 16 GB card, busier desktop | ≥ 13,700 MiB for the head | 2 | 262,144 (1 window) | MTP | 12,976 / 13,446 (13,006 MiB at peak, 2026-09-25) |
| 16 GB card, busier desktop still | ≥ 13,500 MiB for the head | 1 | 262,144 (1 window) | MTP | 12,788 / 13,023 (12,828 MiB at peak, 2026-09-25) |
| RTX 5090 | 32,607 MiB | 8 | 786,432 (3 windows) | MTP | 23,960 / 25,840 |
| H100 80 GB | 81,559 MiB | 16 | 2,883,584 (11 windows) | MTP | 64,888 / 68,648 |
| RTX PRO 6000 / H200 | ≥ 90,000 MiB | 16 | 3,538,944 (13.5 windows) | MTP | 77,208 / 80,968 |

A tier is picked by the VRAM the head can have: the card's total less what every other process
holds (`headVramMiB`). A card that also drives a desktop keeps its compositor's, browsers' and
editors' share, and the 16 GB tier did not fit beside it: on 2026-09-24 a fresh `splice setup`
container on the RTX 5070 Ti that drives this box's display (2,251 MiB held by the desktop,
14,052 left) failed to allocate the MTP head's 116.28 MiB compute buffer at 4 × 294,912 and
crash-looped. Beside the same desktop, the live unit's argv with the public r2 pack loaded and
served a 17,100-token request at 4 × 262,144 (13,100 MiB at peak), 2 × 262,144 (12,796) and
1 × 262,144 (12,664): the charge is 42–64 MiB under each peak. 4 × 294,912 failed with 268 MiB
above its steady 13,784 and 4 × 262,144 loaded with 952 above its peak, so each desktop tier's
min_vram_mib sits ~700 MiB above its measured peak (local/research/desktop-tier-2026-09-24).
Re-measured on 2026-09-25 beside the same desktop for n_max 3 and the draft vocabulary on engine
3c7e643: 13,410 / 13,006 / 12,828 MiB at peak at 4 / 2 / 1 × 262,144, the charge 30–58 MiB under
each, so the tiers moved to 14,100 / 13,700 / 13,500 and a 2-slot tier joined them: a desktop
holding 2.2–2.6 GB keeps two slots rather than one. On engine 4d44f6c the same legs OOMed at
4 × 262,144 and peaked at 14,234 MiB at 1 × 262,144: its PQ2_0 tensor-core kernel held 1,430 MiB of
the 5070 Ti for a driver syscall stack from its first launch, which fork #50 removed (engine.toml,
local/research/desktop-tier-2026-09-25). The RTX 5080 row's peak is from the engine-5 gate on a
rented 5080 (vast 52614682).

The 5080 row is the measured local geometry; the others are arithmetic on the constants (the
5090's drafted 4-slot leg loaded and ran on a rented 4×5090 box on 2026-09-20, at -c 32768, with
the DFlash2 head of that morning). Earlier rows of this table (15,514 / 27,454 / 74,318 / 87,838)
were the f32-state, f16-mask, decode-only constants; the pair before these
(13,832 / 23,600 / 64,080 / 76,400 at n_max 2) were the same constants before the verify rows were
charged. Both are superseded, and `head.test.ts` now holds every cell of this table against
`tierNeedMiB` so a restatement cannot drift from the charge again.

The MTP head's cache holds prompt and generated rows alike, sized to the target's pool
(`speculative.cpp`: `cparams.n_ctx = llama_n_ctx(ctx_tgt)`), so `bytes_per_token = 1216` (1,152 of
q4_0 K + V for its one attention layer + 64 of bit-packed mask) is charged once for the shared
pool: 342 MiB at 294,912 (KV buffer 324.00 MiB + draft compute 116.28 measured at start, 2026-09-21
08:5x). From 2026-09-21 01:37 to 08:5x the head ran decode-only (`--spec-draft-mtp-decode-only
--spec-draft-mtp-window 16384`: the prompt never entered it, 72 MiB of KV at 16,384 × 4); on the
same ~100K-token conversation that gave 40–58 tok/s at 49–54 % acceptance against 76–91 tok/s at
54–69 % with the prompt in the cache and the lens off (below), so decode-only is gone.

## Draft vocabulary (2026-09-25)

`[speculative] args` hands the engine `--spec-draft-mtp-vocab assets/mtp-draft-vocab-98304.i32`
(engine #44). A draft step scores 98,304 of the LM head's 248,320 rows, 134 of its 338 MB of
PQ2_0, and every other token's draft logit is -inf. The verify is untouched: a token outside the
set costs the draft that position, never the text. The rows are a copy, 127.5 MiB, charged in
`[speculative] weights_mib` (430 → 558).

**The set.** Every token id the served pack sampled on 500 train prompts of the retraining pool
above (served sampling, 512 tokens each), or that the pool's train prompts contain (53,107 ids),
then the lowest ids not yet in it: BPE ids run roughly in merge order. It covers 99.93 % of the
tokens the pack sampled on the pool's 112 validation prompts; the lowest 98,304 ids alone cover
99.55 %, and the 20,630 sampled ids alone 93.06 %. `hot-build2.py` rebuilds it byte for byte
from the generations and the pool's counts (`local/research/mtp-draft-vocab-2026-09-25/`).

**Served, on the box** (RTX 5080, the public r2 pack, engine d0a6df21f + #44, the 48 held-out
greedy requests above, legs base / set / set / base, tok/s per request with the leg order
cancelled):

| n_max | base tok/s | with the set | per request | acceptance | text = base |
|---|---|---|---|---|---|
| 2 | 163.8 | 168.4 | +2.66 % (95 % CI +2.17 to +3.15), 47/48 faster | 0.7195 → 0.7189 | 47/48 |
| 3 | 174.1 | 181.2 | +3.83 % (95 % CI +3.38 to +4.28), 47/48 faster | 0.6361 → 0.6359 | 47/48 |

- Every id but one unused PAD token reproduces base's text and draft counters 48/48, so the rows
  and their ids are exact; a random 32,768 ids drop acceptance to 0.0627.
- 65,536 ids: +2.54 % at acceptance 0.7056, with 16 texts parting; 98,304 is the smaller loss.
- At ~117K tokens of context (two requests over the engine's docs, n_max 2): +2.48 and +0.58 %.

## Draft depth (2026-09-25)

`[speculative] n_max = 3`. Depth 2 was the peak of ProCreations' sweep and of the 5080's own
runs with ProCreations' head (below, 2026-09-20/21). With the retrained head and the draft
vocabulary a third position pays. RTX 5080, the public r2 pack, greedy, 512 tokens a request,
tok/s per request with the leg order cancelled (`local/research/mtp-draft-vocab-2026-09-25/`):

- **Shallow**, the 48 held-out requests at -c 40,960 (the table above, engine d0a6df21f + #44):
  181.2 tok/s at n_max 3 against 168.4 at n_max 2, both with the set.
- **Deep** (`round5/`, engine 4d44f6c): six requests built from the eval set's 24 held-out
  agentic sessions (SWE-rebench OpenHands trajectories, none shared with the retraining pool):
  6–8 sessions stacked into one user turn, each request ending on a different one, then a
  request for a detailed report on the last (`longeval.py`). 94,071–120,693 prompt tokens, one
  slot at -c 163,840, legs n2 / n2v / n3v / n3v / n2v / n2:

| n_max | draft vocabulary | tok/s (two legs) | acceptance | tokens per round | text = n2 |
|---|---|---|---|---|---|
| 2 | — | 147.23 / 147.00 | 0.8270 | 2.639 | 6/6 |
| 2 | 98,304 | 150.37 / 150.76 | 0.8270 | 2.639 | 6/6 |
| 3 | 98,304 | 169.87 / 169.69 | 0.7786 | 3.314 | 0/6 |

- n_max 3 against n_max 2, both with the set: +13.00 % (95 % CI, t(5), +7.82 to +18.44), 6/6
  faster (+5.97, +10.02, +13.25, +15.37, +21.05, +12.95).
- The set alone at n_max 2: +2.35 % (+2.15 to +2.56), 6/6 faster, texts and counters identical.
- n_max 3 with the set against n_max 2 without it: +15.66 % (+10.14 to +21.46), 6/6 faster.
- The n_max 3 texts part from n_max 2's in all six at a near-tie, both readings coherent (deep-04
  "…5.2. The issue is about" against "…5.2 and the issue about"; deep-05 "`UnicodeDecodeError`"
  against "UnicodeDecodeError"). Every emitted token is still the pack's own greedy pick.
- On two requests over the engine's docs at ~117K (`round4/`, engine d0a6df21f + #44), n_max 3
  with the set against n_max 2 without it split -3.87 % and +9.70 %. The agentic sessions are the
  workload the head serves.

## Gates on rig 0.1.7's tree (engine 3c7e643, 2026-09-25)

`rig gate bonsai-2-27b` on the RTX 5070 Ti that drives this box's display, engine 3c7e643, n_max 3 with the draft
vocabulary (runs in `local/gate-runs/bonsai-2-27b/`: refusal and fluency `20260925T181142Z`, humaneval
`20260925T182225Z` and `20260925T182716Z`, decode and speculative `20260925T183214Z`, needle `20260925T183748Z`,
depth `20260925T184041Z`):
- refusal: base 6/6 → served 0/6, capability 5/5 byte-identical;
- fluency: doubled 2 → 2, repeats 2 → 2 over 1,800 tokens;
- HumanEval pass@1: base 113 → served 112 (six lost, five gained, sign test p 0.500), and on a second run of the same
  build 113 → 113 (seven lost, seven gained, p 0.605);
- decode A-B-A: base 76.4 / served 78.3 / base 78.9 tok/s;
- speculative: plain 71.2 → drafted 153.9 tok/s (×2.16), 58 % of 836 drafted tokens accepted; the three answers
  diverge from the plain leg at near-ties of 0.069, 0.043 and 0.029 nats (tie ≤ 0.15). At n_max 2 without the
  vocabulary on engine 514c53c the same probe read 75.9 → 103.4 tok/s (×1.36);
- needle: 2/2 retrieved (depth 8 % and 55 %) in a 128,641-token haystack at `-c 131072`, 113.4 s;
- depth matrix (llama-bench, fa on, q4_0 K/V, 3 repeats): pp512 1,447 / 1,151 / 1,297 / 886 / 558 and tg64 76.6 /
  74.9 / 65.1 / 58.3 / 46.8 tok/s at 0 / 16K / 64K / 128K / 256K.

HumanEval's rule was served ≥ base until this run. On one build the first run failed it and the second passed it: the
leg runs on four slots, how requests share a batch changes the numerics, and 16 of the 164 base programs differed
between the two runs. The rule is now a one-sided sign test on the problems the two legs disagree on (`gates.toml`,
alpha 0.05): the served pack fails only when it loses significantly more problems than it gains. Nine lost and two
gained (113 → 106) is p 0.033 and fails; `probes.test.ts` holds both cases.

The humaneval leg died three times on this loaded host before it finished, after 70, 69 and 48 requests: twice
the host's memory defense stopped the gate's build scope, once earlyoom took the server. The gate's server kept the
engine's default 8 GiB host prompt cache, which no gate request reads (they send `cache_prompt: false`), and held
5.2 GB of host memory after 48 requests; with `--cache-ram 0`, which gate servers now pass, it held 0.9 GB at the
same point and 985 MiB at its peak.

The depth matrix reads under 514c53c's (tg64 78.2 / 76.1 / 69.3 / 61.3 / 51.0 on 2026-09-21), by 2 % at depth 0 and 8 %
at 256K. That is not the engine. Run on this card in one session, each engine forward and then in reverse
(`local/research/engine-timeline-2026-09-25/`), 514c53c built from a clean tree with rig's flags reads tg64 68.0 / 58.8 /
54.0 tok/s at 0 / 64K / 128K, c008fe8 (rig 0.1.1 to 0.1.6) 70.4 / 61.6 / 55.0 and 3c7e643 78.1 / 68.2 / 59.1: +14.9 %,
+16.0 % and +9.4 % over 514c53c. The fork's merges between those two (#5, #7, #8, #9, #10, #13, #24) read 67.8 to 70.2 at
depth 0 in that session. The 2026-09-21 figures are no reference: today the same code and weights (the 851 tensors
outside blk.64 are byte-identical in that run's pack and this one) under the same caps (power-budget's 4.6 GHz CPU and
250 W on this card, reached in both runs) read 12 to 15 % under them, and that run's binary is gone, built before rig
refused a dirty engine source (7cca429). Across sessions on this card one binary moves by up to 8 %: #10's build read
75.2 at depth 0 in its 2026-09-23 gate and 69.5 today. On the same card, interleaved against rig 0.1.6's engine c008fe8
at a load falling from 29 to 9 (`local/research/depth-ab-2026-09-25/`), 3c7e643 is the faster at every depth:
- llama-bench, legs A B C C B A, C being 3c7e643 with #33's 16-cell mask scan switched back
  (`LLAMA_KQ_MASK_SCAN_LEGACY=1`), 6 samples a state: tg64 69.2 → 75.7 tok/s at 0 (+9.3 %), 59.3 → 64.8 at 64K
  (+9.4 %), 55.4 → 57.4 at 128K (+3.6 %); pp16384 1,720 → 1,748 (+1.7 %). The switch reads within noise of the
  shipped build at the same host load.
- served, llama-server with rig 0.1.7's argv and no draft (the bit-packed mask, which llama-bench does not run), one
  greedy request of 128 tokens after 65,536 and one after 131,072 prompt tokens per leg, legs A B B A: decode 58.9 →
  63.7 tok/s at 64K (+8.1 %) and 52.7 → 58.0 at 128K (+10.2 %); prompt 1,430 → 1,401 tok/s at 64K (−2.1 %) and
  1,162 → 1,171 at 128K (+0.8 %).

### The needle at the depth the head is used (2026-09-26)

No gate had looked past 129,326 tokens, while a coding agent fills one slot's 262,144-token window past 245,760.
`rig gate bonsai-2-27b --only needle` with `[needle]` at 245,000 tokens in 262,144 (`gates.toml`), on a rented RTX
5080 with rig 0.1.7's published engine (3c7e643, the sm_120 tarball) and the public r2 pack (`0e5524be…`), one slot at
`-c 262144`: 2/2 retrieved (depth 8 % and 55 %) in a 249,655-token haystack, 195.6 s
(`evidence/gates/20260926T011334Z/`).

## The pool's far end (engine c1518d4, 2026-09-26)

On the evening of 2026-09-25 the operator's head on an RTX 5090 (rig 0.1.7: eight slots over one unified pool of
786,432 cells, n_max 3 with the draft vocabulary, idle slots kept in the pool) slowed as the day's conversations filled
the pool. The same argv on a rented RTX 5090 (`local/research/evening-2026-09-26/ev1-5090`): three conversations of
178,000 tokens go first, then a 245,760-token prompt asks four questions (512 tokens each, temperature 1.0, seeded), and
conversation 1 returns; each leg is a fresh server. A is the prompt alone, in the pool's first cells; K is the evening
with idle slots kept as rig serves them (`--no-cache-idle-slots --cache-ram 0`), which puts the prompt in the last cells:

| engine | leg | cold prefill | per draft round | decode | the three conversations' prefill |
|---|---|---|---|---|---|
| 3c7e643 (rig 0.1.7) | A | 117.3 s | 14.75 ms | 159.1 tok/s | |
| 3c7e643 (rig 0.1.7) | K | 364.7 s | 22.31 ms | 99.6 tok/s | 72.8 / 132.4 / 192.3 s |
| c1518d4 | A | 115.2 s | 13.15 ms | 162.9 tok/s | |
| c1518d4 | K | 116.8 s | 13.32 ms | 165.2 tok/s | 71.7 / 71.9 / 72.1 s |

At 3c7e643 flash attention's stream-k divided the pool's whole span among its blocks (`ntiles_KV` over `K->ne[1]` in
`fattn-common.cuh`) while the fork's range pre-pass (#30) held each Q tile to its own cells, so for a sequence in the
pool's last third about 31 % of the blocks had work, and prefill ran no pre-pass. The fork's #63 (engine-8, with
engine-9's fixup) splits only the KV steps each Q tile sees. c008fe8 (rig 0.1.6) pays the same at the far end, one slot
plain (`local/research/highidx-2026-09-26`): 360.3 s and 53.7 tok/s against 113.1 s and 86.5 tok/s in the first cells.
The evening's slowdown was the pool filling, not that day's deploy.

The pool's last cells also move the greedy text, on both engines at the same positions (tokens 59, 61, 35 and 112 of
four 384-token answers, one slot, plain): 3c7e643's first differences are 4 at a tie (≤ 0.15 nats) and none beyond,
c008fe8's 2 at a tie and 2 beyond (0.184 and 0.188 nats); over the agreeing tokens |Δ logprob| p99 is 0.120 and 0.126.
That is attention over a longer, differently placed span, not a defect of either engine.

Caching idle slots in host memory instead (`--cache-idle-slots --cache-ram 30899`, the engine's default) also keeps the
pool compact: on 3c7e643 the prompt took 117.2 s at 14.75 ms a round and the three conversations 72.7 s each. A
conversation then returns through host memory (`ev2-5090`). Named by its slot (id_slot 1) it was processed again from
its start on both engines, 76.4 s on 3c7e643 and 74.1 s on c1518d4 to the first token, against 1.1 and 0.4 s with the
slots kept: the engine never loaded a cleared slot a request named (the fork's fix is 07e92c710 on train/engine-10).
With no slot named it was loaded: the first token after 5.33 s on 3c7e643 and 3.50 s on c1518d4, then 124.7 and 198.2
tok/s. rig
keeps `--no-cache-idle-slots`: the pool holds a returning conversation, and on c1518d4 a full pool costs 1.3 % a round.

`--backend-sampling` (the target sampled on the GPU) read +0.3 % a round at temperature 1.0 with the same acceptance,
and +0.44 % greedy on a 5080: rig does not pass it.

Gates on the prebuilt rig installs (`local/research/prebuilt-engine9-2026-09-26`). On a rented RTX 5080 the release
tarball ran against rig 0.1.7's (engine-3c7e643's release asset), each with NVIDIA's pinned runtime, on the public r2
pack. One slot at 262,144, the 245,760-token prompt's four questions greedy with top-5 log-probabilities: every first
difference is at a tie (0.011 to 0.137 nats), and |Δ logprob| over the agreeing tokens has p99 0.085 (3c7e643 against
c008fe8: 0.070). Plain decode there ran 58.6 against 47.3–48.0 tok/s, prefill 186.3 against 189.9 s. The served draft
against plain on c1518d4: three first differences, each at a tie, and one answer identical for all 384 tokens. A
conversation swapped out of slot 0 and back at the 5080 tier (4 × 294,912, `--cache-ram 16384`): its first token after
3.21 and 3.17 s, the same answer in both legs. `rig gate --only census` on an RTX 5090: 1,160 launches a step against
the table's 1,224, `cpy_scalar` 64 → 0 alone (`gates.toml`), and the new table passes on the same capture. The
driver-only e2e on Ubuntu 22.04 (RTX 5090, driver 610.57.04): installed, the runtime resolved from the build directory,
tg128 161.1 tok/s.

## Draft head retrained (MTP r2, 2026-09-24)

The first `[[derive]]` step writes our retrained head over the pack's `blk.64`. It has the same
15 tensors, types and shapes as ProCreations' head (below), and every machine gets it: it is
public, fetched by `rig fetch` from torad-labs/rig's release `bonsai-2-27b-mtp-r2`
(`head_sha256` 7fe0f04c…).

**Training.** Rented RTX PRO 6000 Blackwell (vast 52390478, 2026-09-24).
- Starting point: ProCreations' bf16 head (`model_mtp.safetensors` @ efffdea).
- Target: the served pack (ablated, head unchanged). Its hidden states are the inputs, and its
  own sampler (temp 1.0, top-p 0.95, top-k 20, truncated) gives the target distribution.
- Each step lays one sample out the way a draft round runs it. Both positions are trained, and
  position 2 takes position 1's output.
- Loss: forward KL plus 0.1 CE on the teacher's mode.
- Requests: 4,400, all from public data:
  - 1,600 agentic cuts of 400 OpenHands trajectories with tool calls
    (nebius/SWE-rebench-openhands-trajectories, CC-BY-4.0);
  - 1,500 Magicoder-OSS-Instruct problems (MIT);
  - 900 UltraChat first turns (MIT);
  - 400 NuminaMath-CoT problems (Apache-2.0).
- Continuations were generated by the served pack. No conversation of the operator's was used.
- r1 was one epoch. r2 is r1's data for 3 epochs: 788 steps, 66 min, best at step 700.
- Offline acceptance under the teacher: position 1 0.685 → 0.709, position 2 given 1 0.688 →
  0.713, tokens per round 2.157 → 2.215 (ceiling for position 1: 0.772).

**Served pack, on the box** (greedy, 48 held-out requests: 24 agentic with tools and 8 each of
code, chat and math; `--spec-draft-n-max 2`; accepted / drafted):

| head | agentic | chat | code | math | all |
|---|---|---|---|---|---|
| ProCreations' | 0.829 | 0.546 | 0.633 | 0.726 | 0.682 (9,629 / 14,122) |
| r1 | 0.847 | 0.606 | 0.713 | 0.764 | 0.733 |
| r2 | 0.844 | 0.603 | 0.710 | 0.778 | 0.733 (9,936 / 13,561) |

- **Speculative probe.** rig's probe on the r2 pack: 116.6 → 217.0 tok/s (×1.86), 71 % of 630
  drafted tokens accepted, PASS.
- **Long context.** At 64K and 128K-token prompts (3 fixed-seed generations each, served sampler),
  r2 decodes at 0.99 to 1.01× ProCreations' head (0.9985× over all). Tokens per round are
  equal there (2.42 against 2.43): the acceptance gain above does not carry to long prompts.
- **Tie check.** Against a no-draft leg, top-1 under greedy: r2 matches on 14 requests, parts at
  a near-tie on 31 and DIFFERS on 3 (0.17–0.27 nats, past the 0.15 tie gap). ProCreations' head
  in the box's reference legs on the same requests: 17, 27 and 4 (0.15–0.22). The divergence
  comes from the verify batch's numerics, which any head triggers: a draft only proposes, and
  the pack decides every token.

**The public pack, here** (the source pack with r2 spliced in, against the source pack as
published; RTX 5070 Ti, engine c008fe8, the same 48 requests and flags, one slot at 40,960,
`--cache-ram 0`; `local/research/mtp-r2-public-2026-09-24/`):

| head | agentic | chat | code | math | all |
|---|---|---|---|---|---|
| ProCreations' | 0.815 | 0.531 | 0.624 | 0.710 | 0.670 (9,594 / 14,315) |
| r2 | 0.837 | 0.615 | 0.699 | 0.780 | 0.733 (9,865 / 13,456) |

The gain holds without the ablation: +0.063 overall, the most on chat and math. Greedy texts
were identical on 15 of the 48 requests; the rest part at a near-tie that the pack resolves
under verify-batch numerics, as in the tie check above.

**Reproducible to the byte.**
- rig's splice of the head into the ablated pack gives 389b6d3c…, the box's own export of r2.
- rig's derive from source (splice, then ablation) gives the same 389b6d3c….
- The splice alone gives the public pack, 0e5524be….

## Draft head (MTP, 2026-09-20 evening)

`[speculative] type = "draft-mtp"`: ProCreations' Bonsai-trained multi-token-prediction head
(the Qwen3.8-27B MTP block, 424.7 M parameters, fine-tuned against the frozen original Bonsai
pack; Q8_0 matrices, F32 norms) carried inside the served pack as blk.64 with
`qwen35.nextn_predict_layers = 1`. It replaced the DFlash2 sidecar (below) the same day because
it fits the 5080 beside a trained window: 430 MiB of weights against 1,951 + 2,400 for the
sidecar, one full-attention layer's q4_0 KV (1,152 B per pooled token) and a compute buffer that
grows 1 KiB per pooled token (196 / 356 / 388 MiB at 131,072 / 294,912 / 327,680), and no
second process — the draft context is a second `llama_context` on the same weights.

Engine 514c53c: the MTP graph looked the next token up in the target's Hadamard-latent
`token_embd` without the inverse transform (ProCreations ship the same 12-line fix as a patch on
Prism's runtime); the fork now restores the primal basis after the lookup as the trunk and
DFlash do. Without it the pack's `prism.hadamard` verifier refuses the MTP graph.

Measured on the 5070 Ti (GPU 1, display card, -c 131072 -np 2, lens on, greedy, 512 tokens,
prompt cache off; `evidence/spec-bench-mtp-5070ti/`, rows produced by `scripts/bench-head.ts`):

| prompt | thinking | plain tok/s | MTP tok/s | gain | accepted |
|---|---|---|---|---|---|
| code (LRU cache) | off | 35.9 | 47.1 | +31 % | 306/409 (0.75) |
| SQL schema | off | 35.6 | 45.6 | +28 % | 291/438 (0.66) |
| prose | off | 36.1 | 42.2 | +17 % | 258/505 (0.51) |
| reasoning (trains) | off | 36.6 | 49.2 | +34 % | 324/372 (0.87) |
| code | on | 37.1 | 39.0 | +5 % | 267/486 (0.55) |
| reasoning | on | 36.5 | 47.0 | +29 % | 316/388 (0.81) |
| aggregate | | 36.3 | 44.7 | +23 % | 0.69 |

`n_max = 2` was ProCreations' own sweep on an RTX PRO 6000 at 32K context, one slot (n1 153.6,
n2 174.6, n3 172.7 tok/s); each level is one more recurrent-state copy per slot. Measured here on
the 5080 (2026-09-21, prompt-inclusive head, lens off, `-lv 4`, natural requests on one ~100–125K-token
conversation, weighted over the completed requests; `local/logs/bonsai-2-27b.log` print_timing
and `spec … statistics` lines):

| depth | p_min | requests | decode tok/s | accepted / proposed | drafts per round | mean emitted |
|---|---|---|---|---|---|---|
| 2 | 0 (default) | 9 | 79.8 (76.2–91.4) | 0.54–0.69 | 2.000 | 2.1–2.3 |
| 4 | 0 | 5 | 80.5 (76.5–83.6) | 0.417 (acc/pos 0.686, 0.480, 0.346, 0.256) | 4.000 | 2.6–2.8 |
| 8 | 0 (decode-only head, lens on, e3a9015) | 1 | 39.0 | 0.186 (acc/pos 0.595, 0.347, 0.210, 0.131, 0.088, 0.057, 0.037, 0.025) | 8.000 | 2.49 |
| 8 | 0.3 | 8 | 43.6 (59.1–87.2 on the six uncontended; 4.5 and 42.4 while slot 0 prefilled a 164K prompt) | 0.327 (acc/pos 0.667, 0.426, 0.281, 0.195, 0.137, 0.101, 0.074, 0.053) | 5.92 | 2.93 |
| 8 | chain 0.3 (engine bc08995) | 2 | 71.4 (65.9 at 180K, 75.8; two slot-3 requests during the 144 s prefill not counted) | 0.579 (acc/pos 0.683, 0.403, 0.242, 0.153, 0.107, 0.075, 0.054, 0.041) | 3.04 | 2.76 |

Depth without a cutoff is flat: conditional acceptance is ~0.70 at every position, so the
yield compounds down while each position costs its draft and verify row. `--spec-draft-p-min`
(`speculative.cpp`: a round stops drafting once the head's top-1 probability is under it,
default 0) is the lever, and at 0.3 it does not pay on this head: the cutoff took (5.92 drafts per
round, not 8) but the head still ran six sequential passes to bank 2.93 tokens, and the six
uncontended requests weighted 62.6 tok/s against depth 2's 79.8 (04:32–04:39, 12,405 tokens,
the same conversation). A cutoff on the chain's probability product (`--spec-draft-chain-p-min`,
engine bc08995) restores the shape — 3.04 drafts a round, acceptance back at depth 2's — but not
the speed: 71.4 tok/s on two clean requests (04:43–04:48) against 82.4 for depth 2 on the same
conversation. Its 38.0 ms round is ~2.3 ms per verify row plus a ~4.7 ms step both cutoff
configurations share: a verify batch whose size changes between rounds rebuilds the graph
(`llm_graph_params::allow_reuse` requires the previous ubatch's `n_tokens`, `src/llama-graph.h`
868 at bc08995), which a fixed depth reuses every round; a per-shape graph cache would bring it
to ~33 ms, ~85 tok/s, still no better than depth 2. Depth 2 without a cutoff stays the served
value (superseded 2026-09-25 by depth 3 on the retrained head: "Draft depth" above). The DFlash2 sidecar on the same card and
prompts at -c 65536: 34.0 → 42.0 tok/s (+24 %), acceptance 0.42–0.85 — the same gain for ten
times the memory. The live 5080 head's first request after the switch (20:37, 27 tokens):
18/18 drafted tokens accepted, 50.4 tok/s.

n_max=4 trial (2026-09-21, 5080, engine f8394f1, lens off, four prompts, greedy, 256 tokens, warm-up discarded, `scripts/bench-head.ts`): the prompt-inclusive M4 trial ran the live head at `--spec-draft-n-max 4`. Per-prompt mean decode: code 90.3 / SQL 76.4 / prose 77.7 / reasoning 96.1 tok/s (mean 85.1), 4,999 drafted / 1,798 accepted tokens (acceptance 0.360 per drafted token; 0.25 per deep position). n_max=2 on the same card and build, same card, same context, same lens state, measured after the config fix (`--spec-draft-n-max 2`): code 100.1 / SQL 79.6 / prose 90.6 / reasoning 102.9 tok/s (mean 93.3, +9.6% over the trial), 3,074 drafted / 1,516 accepted tokens (acceptance 0.493 vs the trial's 0.360 per drafted token). The single-layer MTP head over-drafts at n=4: it produced 4,999 draft tokens — 63% more than n=2's 3,074 — to accept only 18% more total tokens (1,798 vs 1,516), i.e. its extra draft positions accepted far fewer tokens each, so the extra recurrent-state copies per slot and the larger verify batches bought decode time. n_max=2 matches ProCreations' own sweep peak and is the value every head now serves (superseded 2026-09-25: n_max 3, "Draft depth" above).

5080 `scripts/bench-head.ts` (2026-09-21, engine f8394f1, lens off, four prompts, greedy, 256 tokens, warm-up discarded):

| depth | mean decode tok/s | drafted | accepted | acceptance / drafted |
|---|---|---|---|---|
| 4 (prompt-inclusive M4 trial) | 85.1 (code 90.3 / SQL 76.4 / prose 77.7 / reasoning 96.1) | 4,999 | 1,798 | 0.360 (0.25 per deep position) |
| 2 (after the config fix) | 93.3 (code 100.1 / SQL 79.6 / prose 90.6 / reasoning 102.9, +9.6% over the trial) | 3,074 | 1,516 | 0.493 |

Gates on the MTP pack (`rig gate`, RTX 5070 Ti, engine 514c53c, run `20260921T014813Z`, banked
under `evidence/gates/20260921T014813Z/`): refusal base 6/6 → served 0/6 with capability 5/5
byte-identical; fluency doubled 2 → 2, repeats 2 → 2 over 1,800 tokens; HumanEval pass@1 base
110/164 → served 114/164; decode A-B-A 77.8 / 78.1 / 76.6 tok/s; needle 2/2 at 129,326 tokens;
depth matrix pp512 1,743.9 / 1,682.6 / 1,275.2 / 961.2 / 648.0 and tg64 78.2 / 76.1 / 69.3 /
61.3 / 51.0 tok/s at 0 / 16K / 64K / 128K / 256K — the same pack numbers as the pre-MTP pack
above inside their noise (the MTP block is only read by the draft context). The speculative probe
of that run FAILED its byte-identity check: 0/3 identical, 73 % of 608 drafted tokens accepted,
75.9 → 103.4 tok/s (×1.36); the next paragraph is why, and what the probe judges now.

**Drafted greedy text is not byte-identical to plain greedy text on this model, and that is the
verify batch, not the head:** all six prompts diverged from the plain leg (first difference at
character 332–1,500, always a near-tie: `ttl=300` vs `ttl=60.0`, `9:00 AM` vs `9:00`), while the
MTP and DFlash2 legs produced the SAME text as each other on five of six prompts. A verify batch
puts 1 + n_draft rows through the Gated DeltaNet recurrence at once, and that batched kernel's
accumulation order differs from the one-token step; every token is still the pack's own argmax
under the numerics it ran with. The `speculative` gate's byte-identity check failed on this head
(run `20260921T014813Z` above, 0/3 identical at 256 tokens), so the probe now asks the plain leg
for its top-10 logprobs per token and judges the first divergence: the two answers are
byte-identical, or the drafted token sits within `tie_gap` nats of the plain leg's top-1 at that
position. A drafted token the pack did not rank, or ranked far below, is a wrong token and fails
(the tests show both failing, and a 2.1-nat example is what a wrong token looks like). Measured on
the rewritten probe (engine 95ec4f3, 5070 Ti, run `20260921T021949Z`, banked under
`evidence/gates/20260921T021949Z/`): three divergences, at tokens 204, 12 and 222, with gaps of
0.053, 0.008 and 0.009 nats (`typically`/`gradually`, `_string`/`_str`, `,`/` and`), 73 % of 608
drafted tokens accepted, 69.5 → 102.0 tok/s (×1.47): PASS. `tie_gap = 0.15` is three times the
largest gap seen.

The lens on the drafted head: the lens holds `n_seq_max × (1 + n_rs_seq)` rows (12 on the 5080)
and the server writes one line per accepted token from the row it was sampled from; on the
test server 512/512 rows were present with layer 63's top-1 equal to the served top-1 on every
row and positions consecutive.

## Lens cost (2026-09-20 night)

The lens went live on the 5080 at 18:32 and halved decode: the head's own log reads 49 tok/s
before and 30.5 after, and the gate's decode probe on the 5070 Ti read 77.8 tok/s without the lens
against 36 with it. The cost was the writer, not the graph: ranking six layers × 151,936 logits
with a partial sort and formatting JSON ran on the decode thread, 6.3 ms per token (`lens_top`
timed in isolation). Engine 95ec4f3 moves the ranking and the file write to a worker thread behind
a bounded queue (64 jobs, dropped with one warning when full), replaces the sort with a linear
top-k pass, and computes every lens layer's logits with one stacked lm_head matmul.

Four legs on the 5070 Ti (GPU 1, the MTP pack, -c 65536 -np 2, greedy, 512 tokens, the six prompts
of the draft-head table, warm-up discarded, output hashes identical across the first three legs;
rows and scripts under `evidence/lens-legs-5070ti/`):

| leg | aggregate tok/s | per prompt |
|---|---|---|
| lens off | 73.5 | 73.0–74.1 |
| lens graph only (`--lens-layers`, no `--lens-out`) | 67.3 | 65.9–68.6 |
| lens full (graph + writer) | 67.8 | 66.9–68.5 |
| lens full + MTP draft | 87.7 | 77.0–96.8 (accepted 0.53–0.81) |

The writer now costs nothing measurable; the remaining 8 % is the lens graph itself, six extra
lm_head rows per token. All 7,168 lens rows of the two writing legs landed (14 requests × 512),
none dropped. On the live 5080 the first request after the rollout (21:17, a 131,944-token
prompt, lens and draft on) decoded 281 tokens at 60.8 tok/s.

## Draft head (DFlash2, 2026-09-20 morning — superseded by the MTP head above)

Kept as the measurement behind the sidecar path (`SidecarDraft`, `-md`), which rig still
supports. Its `overhead_mib = 2400` was measured with n_max 3 at 4 slots and so contained
1,800 MiB of recurrent-state snapshots that the tier check now charges separately.

`[speculative]` pins ProCreations' DFlash2 head for Bonsai 2 27B (`Bonsai-2-27B-DFlash2-Q8_0.gguf`,
arch `dflash`: 5 layers at n_embd 5120, block size 8, selector top-k 16 / rank 256, fed by target
layers 6/20/34/48/62; engine commit `6303c1f`, `--spec-type draft-dflash`). The engine verifies
every drafted block against the served pack, so the output is the pack's own: the head was trained
on the ORIGINAL pack and drafts for the ablated one unchanged, and the only thing the ablation could
move is acceptance (0.507 ablated vs 0.521 original on the 5090, 143.4 vs 145.1 tok/s).
`--spec-draft-n-max 3`: n5 accepted 0.359 and was slower (142.3 tok/s).

What it costs on the card (llama-server's buffer lines, 5070 Ti, 4 slots): model buffer
1,950.71 MiB (`weights_mib = 1951`); its K/V is a fixed 8,704-cell sliding window (f16 170 MiB;
q4_0 saves 122 MiB and costs 8 % of the gain — 85.8 vs 93.5 tok/s — so the draft cache stays f16);
its compute buffer follows -c (236.75 → 268.75 MiB from 32K to 64K) and the target's verification
graph is 555 MiB against 198 undrafted. Measured GPU delta over the undrafted server: +4,414 MiB at
32K, +4,496 at 64K, which `overhead_mib = 2400` and `bytes_per_token = 2624` reproduce with 19 MiB
to spare. On the 5080 that is 21,021 MiB beside the trained window at 4 slots and 17,265 at one:
the local tier opts out (`speculative = false`).

Per-stream decode, greedy 1,000 tokens, four prompts (code / SQL / prose / reasoning), mean:

| card (host) | streams | undrafted | drafted | gain | accepted |
|---|---|---|---|---|---|
| RTX 5070 Ti (9800X3D), 64K pool, 4 slots | 1 | 71.8 | 93.5 | +30 % | 0.527 |
| | 2 | 53.1 | 66.2 | +25 % | 0.523 |
| | 4 | 38.4 | 59.2 | +54 % | 0.567 |
| RTX 5090 (EPYC 7B12, rented), 32K pool | 1 | 128.8 | 143.4 | +11 % | 0.507 |
| | 2 | 84.7 | 101.8 | +20 % | 0.494 |
| | 4 | 62.7 | 88.1 | +40 % | 0.485 |

A draft round is CPU work (the drafted server holds one core at 97 %), which is why the 2.25 GHz
EPYC gains a third of what the 5.3 GHz desktop does at one stream. The `speculative` gate probe
re-checks the promise on every card: byte-identical greedy answers with and without the draft, the
draft ran, the drafted leg not slower (the result files are banked under `evidence/spec-bench-local/` and
`evidence/spec-bench-4x5090/`).

## Runtime args

- `--cache-type-k q4_0 --cache-type-v q4_0 --kv-mean-center assets/kv-mean-center-PQ2_0.gguf`:
  the K-cache mean-centering bias (16 vectors, one per attention layer, no text), calibrated
  2026-09-22 with the fork's `llama-kv-mean-center` on the corpus rig builds from the pinned
  engine tree, so anyone can rebuild it: `bun scripts/engine-corpus.ts` (writes
  `local/calibration/engine-corpus.txt`: 3,145,728 characters of the fork's docs, server, core,
  common and ggml sources at pin c54ace8, sha256 5a1563c2…), then `llama-kv-mean-center -m <served
  pack> -f local/calibration/engine-corpus.txt -o <asset> -ngl 99 -c 512 --chunks 1000 -ctk q4_0`
  (asset sha256 c76d72bc…). The `-ctk q4_0` matters: a q4_0 K cache turns on the engine's
  Hadamard K rotation, the bias lives in that rotated basis, and a calibration without it lands in
  the unrotated one. That was why a first engine-corpus run on 2026-09-20 came out uncorrelated with
  the asset it replaces (cosine −0.11…0.08 per layer); in the right basis the two agree at
  0.80–0.95 per layer, and calibrating on the source pack instead of the served one gives the same
  vectors (≥ 0.999). The asset it replaces was calibrated 2026-09-18 on the operator's own
  agent-session text, which cannot ship. KL of the served pack's q4_0 K/V against f16 K/V (the
  KL protocol of `local/kl/REFERENCE.md`, 4,096 scored tokens, fork main `5798304`):

  | K bias | Mean KLD | 99 % KLD | same top p |
  |---|---|---|---|
  | none | 0.002777 ± 0.000091 | 0.019482 | 98.242 % |
  | 2026-09-18, private corpus | 0.002339 ± 0.000081 | 0.014175 | 98.168 % |
  | **2026-09-22, engine corpus (this asset)** | 0.002483 ± 0.000076 | 0.018838 | 98.217 % |

  The public corpus recovers two thirds of the private one's gain on mean KLD (1.8σ short of it)
  and matches it on same top p. The needle probe is what proves the setting safe at depth, and it
  grows its haystack from the same engine corpus: with this asset, 2/2 markers retrieved at
  129,268 tokens, at depths of 8 % and 55 % (`evidence/gates/20260923T051014Z/`).
- The engine's q4_0-native tensor-core flash attention with int8 Q·K (fork commit 60feea0):
  RTX 5080 prefill at 131K 930 → 1,382 tok/s, decode at 131K 44.5 → 73.3 tok/s;
  `GGML_CUDA_FATTN_Q4_0_LEGACY=1` is the off switch.
- `--checkpoint-every 16384` (fork commit aa8ef36): a pinned context checkpoint every 16K prompt
  tokens and one at the exact token a prompt forked, so a compaction or a new session resumes
  from a checkpoint instead of re-prefilling from zero.
- `--chat-template-file assets/chat-template.jinja --reasoning-format deepseek`: the model's
  template with the reasoning block routed to `reasoning_content`. The template is the model's at
  the pinned revision with one line changed (its line 106): a system message arriving after the
  first turn is emitted as a system turn where the original raised `System message must be at
  the beginning.`
- Sampling `--temp 1.0 --top-p 0.95 --top-k 20`: the model card's; llama-server does not read it
  from the GGUF.

## Client facts

`rejects_reasoning_effort`: the template raises on any `reasoning_effort` it does not know.
`slot_pinning`: llama-server picks slots by prompt similarity and will take an idle
conversation's slot for a new one sharing its preamble; a client pins each conversation with
`id_slot`. `any_model_id`: the loaded model answers to any model id.

## Lens capture contract (2026-09-21)

The viewer's rule (../experiments/lens-viewer/viewer.py:110–116, the experiments repo beside this one) is that the final lens layer's top-k
equals "served", ids in order and probabilities within 1e-4. 95ec4f3 broke it: the stacked lm_head
matmul runs a different kernel path than the output matmul, so on a 7,168-line capture of the
lens-legs bench (5070 Ti, the bench's own prompts) 18 lines differed in top-12 order and 242
exceeded the tolerance (worst 2.4e-4). 5416993 makes the last layer's lens entry alias the served
logits tensor (src/llama-graph.cpp lens_build), so the rule holds byte for byte by construction and
the stack carries five layers instead of six. Mechanism check on a small Qwen3.5 (0.8B, GPU 1):
409 capture lines, 0 mismatches, worst probability delta 0.0; the 95ec4f3 build passes that small
check too, because the drift only appears in verify batches (the stacked column count crosses the
kernel boundary there), which is why the head's own captures are the check that can fail.

## Tensor parallelism over PCIe (2026-09-21)

Measured on a rented 4× RTX 5090 box (vast 51849679, Jiangsu, PCIe gen4 x16, no NVLink, `nvidia-smi
topo -p2p r` "chipset not supported" for every pair, 0.98 h, $1.73). The fork carries upstream's
real tensor-parallel path (`--split-mode tensor`, meta device + NCCL, PR #19378; the box build links
NCCL 2.28.3); `-sm row` is the older path and is refused for this pack ("device CUDA0 does not support
split buffers", src/llama-model.cpp:1095). One card decodes 141.7 tok/s tg128 with f16 KV
(llama-bench b6f4667); two cards in tensor mode abort at ggml-backend-meta.cpp:1086
(`split_state.ne[j] % div == 0`) on the reshape of `final_output` [6144,T] → [128,16,3,T]. That
reshape is `build_lora_mm` (src/llama-graph.cpp:1637): for every Hadamard-folded weight it permutes
the activation from tiled to grouped head order (`prism.hadamard.gdn_v_grouped`) and applies the
pack's rotation, which is block-diagonal at `prism.hadamard.block_size` = 1024 on the input axis. A
split therefore has to land on 1024-feature boundaries, and the FFN down input is 17 blocks, which no
even 2- or 4-way split respects. Making it work means a block-aware uneven split policy per rotated
weight plus meta-backend support for the fork's reshape/permute/Hadamard nodes, with a logits-parity
gate against one card, for a gain the interconnect does not allow: decode is ~7 ms/token for 7.2 GB
of weights, and tensor mode adds ~2 host-staged all-reduces per layer × 64 layers at 15–30 µs each
while keeping the launch floor. Upstream's own 2× RTX 4090 numbers (PR #19378) say the same for
models that fit one card: llama 8B Q4_0 tg128 175 layer vs 102 tensor, gemma4 26B A4B 197 vs 139.
The community's gains come from models that do not fit one card, from aggregate throughput, or from
the P2P driver patch, which a rented container cannot install. Decision: more GPUs buy slots and
windows for this pack, never single-stream speed; the box was replaced by one RTX PRO 6000 (96 GiB,
the 90000 MiB tier). Brain #720 (2026-05-08) and #1190 carry the research.
