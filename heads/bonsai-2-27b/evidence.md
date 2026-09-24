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

## Gates on the pinned pack (RTX 5070 Ti, engine da69dc5, 2026-09-19/20)

Runs: `Bonsai-demo/runs/2026-09-19/decode-profile-5070ti/baked/` and
`runs/2026-09-20/vendorized-da69dc5-sm120/`. Re-run end to end **through rig** on 2026-09-20
against a rig-built `local/engine-builds/da69dc5-sm120` — `local/gate-runs/bonsai-2-27b/20260920T082930Z/` (its per-probe results banked under
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
| HumanEval-164 pass@1, greedy | 112/164 = 0.683 | 116/164 = 0.707 | served ≥ base − 0 |
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
f5d4f29): RS buffer 526.5 MiB at 4 slots × (1 + n_max 2) copies and 1,579.5 MiB at 4 × (1 + 8),
43.875 per copy, linear in n_max (`n_rs_seq`, the copies the engine rolls a rejected draft back
with); 150 in f32 (1,795.5 MiB at n_max 2). Every tier is checked against those constants by
`headInvariants`, plus the draft head's (below) on every tier that loads it, and `head.test.ts`
re-derives each tier's total from the same constants:

| card | VRAM | -np | -c | draft head | needs by the constants (n_max 2 / 8) |
|---|---|---|---|---|---|
| RTX 5080 | 16,303 MiB | 4 | 294,912 (1.125 windows) | MTP | 13,652 / 14,780 (14,104 MiB used at n_max 4, 2026-09-21) |
| RTX 5090 | 32,607 MiB | 8 | 786,432 (3 windows) | MTP | 23,456 / 25,712 |
| H100 80 GB | 81,559 MiB | 16 | 2,883,584 (11 windows) | MTP | 64,008 / 68,520 |
| RTX PRO 6000 / H200 | ≥ 90,000 MiB | 16 | 3,538,944 (13.5 windows) | MTP | 76,328 / 80,840 |

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

## Draft head (MTP, 2026-09-20 evening)

`[speculative] type = "draft-mtp"`: ProCreations' Bonsai-trained multi-token-prediction head
(the Qwen3.8-27B MTP block, 424.7 M parameters, fine-tuned against the frozen original Bonsai
pack; Q8_0 matrices, F32 norms) carried inside the served pack as blk.64 with
`qwen35.nextn_predict_layers = 1`. It replaced the DFlash2 sidecar (below) the same day because
it fits the 5080 beside a trained window: 430 MiB of weights against 1,951 + 2,400 for the
sidecar, one full-attention layer's q4_0 KV (1,152 B per pooled token) and a compute buffer that
grows 1 KiB per pooled token (196 / 356 / 388 MiB at 131,072 / 294,912 / 327,680), and no
second process — the draft context is a second `llama_context` on the same weights.

Engine a0af7ec: the MTP graph looked the next token up in the target's Hadamard-latent
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
| 8 | 0 (decode-only head, lens on, f5d4f29) | 1 | 39.0 | 0.186 (acc/pos 0.595, 0.347, 0.210, 0.131, 0.088, 0.057, 0.037, 0.025) | 8.000 | 2.49 |
| 8 | 0.3 | 8 | 43.6 (59.1–87.2 on the six uncontended; 4.5 and 42.4 while slot 0 prefilled a 164K prompt) | 0.327 (acc/pos 0.667, 0.426, 0.281, 0.195, 0.137, 0.101, 0.074, 0.053) | 5.92 | 2.93 |
| 8 | chain 0.3 (engine 9579ba8) | 2 | 71.4 (65.9 at 180K, 75.8; two slot-3 requests during the 144 s prefill not counted) | 0.579 (acc/pos 0.683, 0.403, 0.242, 0.153, 0.107, 0.075, 0.054, 0.041) | 3.04 | 2.76 |

Depth without a cutoff is flat: conditional acceptance is ~0.70 at every position, so the
yield compounds down while each position costs its draft and verify row. `--spec-draft-p-min`
(`speculative.cpp`: a round stops drafting once the head's top-1 probability is under it,
default 0) is the lever, and at 0.3 it does not pay on this head: the cutoff took (5.92 drafts per
round, not 8) but the head still ran six sequential passes to bank 2.93 tokens, and the six
uncontended requests weighted 62.6 tok/s against depth 2's 79.8 (04:32–04:39, 12,405 tokens,
the same conversation). A cutoff on the chain's probability product (`--spec-draft-chain-p-min`,
engine 9579ba8) restores the shape — 3.04 drafts a round, acceptance back at depth 2's — but not
the speed: 71.4 tok/s on two clean requests (04:43–04:48) against 82.4 for depth 2 on the same
conversation. Its 38.0 ms round is ~2.3 ms per verify row plus a ~4.7 ms step both cutoff
configurations share: a verify batch whose size changes between rounds rebuilds the graph
(`llm_graph_params::allow_reuse` requires the previous ubatch's `n_tokens`, `src/llama-graph.h`
868 at 9579ba8), which a fixed depth reuses every round; a per-shape graph cache would bring it
to ~33 ms, ~85 tok/s, still no better than depth 2. Depth 2 without a cutoff stays the served
value. The DFlash2 sidecar on the same card and
prompts at -c 65536: 34.0 → 42.0 tok/s (+24 %), acceptance 0.42–0.85 — the same gain for ten
times the memory. The live 5080 head's first request after the switch (20:37, 27 tokens):
18/18 drafted tokens accepted, 50.4 tok/s.

n_max=4 trial (2026-09-21, 5080, engine 2ba9821, lens off, four prompts, greedy, 256 tokens, warm-up discarded, `scripts/bench-head.ts`): the prompt-inclusive M4 trial ran the live head at `--spec-draft-n-max 4`. Per-prompt mean decode: code 90.3 / SQL 76.4 / prose 77.7 / reasoning 96.1 tok/s (mean 85.1), 4,999 drafted / 1,798 accepted tokens (acceptance 0.360 per drafted token; 0.25 per deep position). n_max=2 on the same card and build, same card, same context, same lens state, measured after the config fix (`--spec-draft-n-max 2`): code 100.1 / SQL 79.6 / prose 90.6 / reasoning 102.9 tok/s (mean 93.3, +9.6% over the trial), 3,074 drafted / 1,516 accepted tokens (acceptance 0.493 vs the trial's 0.360 per drafted token). The single-layer MTP head over-drafts at n=4: it produced 4,999 draft tokens — 63% more than n=2's 3,074 — to accept only 18% more total tokens (1,798 vs 1,516), i.e. its extra draft positions accepted far fewer tokens each, so the extra recurrent-state copies per slot and the larger verify batches bought decode time. n_max=2 matches ProCreations' own sweep peak and is the value every head now serves.

5080 `scripts/bench-head.ts` (2026-09-21, engine 2ba9821, lens off, four prompts, greedy, 256 tokens, warm-up discarded):

| depth | mean decode tok/s | drafted | accepted | acceptance / drafted |
|---|---|---|---|---|
| 4 (prompt-inclusive M4 trial) | 85.1 (code 90.3 / SQL 76.4 / prose 77.7 / reasoning 96.1) | 4,999 | 1,798 | 0.360 (0.25 per deep position) |
| 2 (after the config fix) | 93.3 (code 100.1 / SQL 79.6 / prose 90.6 / reasoning 102.9, +9.6% over the trial) | 3,074 | 1,516 | 0.493 |

Gates on the MTP pack (`rig gate`, RTX 5070 Ti, engine a0af7ec, run `20260921T014813Z`, banked
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
the rewritten probe (engine a0bacec, 5070 Ti, run `20260921T021949Z`, banked under
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
timed in isolation). Engine a0bacec moves the ranking and the file write to a worker thread behind
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
layers 6/20/34/48/62; engine commit `e7c67e0`, `--spec-type draft-dflash`). The engine verifies
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
  common and ggml sources at pin 3520147, sha256 5a1563c2…), then `llama-kv-mean-center -m <served
  pack> -f local/calibration/engine-corpus.txt -o <asset> -ngl 99 -c 512 --chunks 1000 -ctk q4_0`
  (asset sha256 c76d72bc…). The `-ctk q4_0` matters: a q4_0 K cache turns on the engine's
  Hadamard K rotation, the bias lives in that rotated basis, and a calibration without it lands in
  the unrotated one. That was why a first engine-corpus run on 2026-09-20 came out uncorrelated with
  the asset it replaces (cosine −0.11…0.08 per layer); in the right basis the two agree at
  0.80–0.95 per layer, and calibrating on the source pack instead of the served one gives the same
  vectors (≥ 0.999). The asset it replaces was calibrated 2026-09-18 on the operator's own
  agent-session text, which cannot ship. KL of the served pack's q4_0 K/V against f16 K/V (the
  KL protocol of `local/kl/REFERENCE.md`, 4,096 scored tokens, fork main `bb2a26f`):

  | K bias | Mean KLD | 99 % KLD | same top p |
  |---|---|---|---|
  | none | 0.002777 ± 0.000091 | 0.019482 | 98.242 % |
  | 2026-09-18, private corpus | 0.002339 ± 0.000081 | 0.014175 | 98.168 % |
  | **2026-09-22, engine corpus (this asset)** | 0.002483 ± 0.000076 | 0.018838 | 98.217 % |

  The public corpus recovers two thirds of the private one's gain on mean KLD (1.8σ short of it)
  and matches it on same top p. The needle probe is what proves the setting safe at depth, and it
  grows its haystack from the same engine corpus: with this asset, 2/2 markers retrieved at
  129,268 tokens, at depths of 8 % and 55 % (`evidence/gates/20260923T051014Z/`).
- The engine's q4_0-native tensor-core flash attention with int8 Q·K (fork commit da69dc5):
  RTX 5080 prefill at 131K 930 → 1,382 tok/s, decode at 131K 44.5 → 73.3 tok/s;
  `GGML_CUDA_FATTN_Q4_0_LEGACY=1` is the off switch.
- `--checkpoint-every 16384` (fork commit 4651ce7): a pinned context checkpoint every 16K prompt
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
equals "served", ids in order and probabilities within 1e-4. a0bacec broke it: the stacked lm_head
matmul runs a different kernel path than the output matmul, so on a 7,168-line capture of the
lens-legs bench (5070 Ti, the bench's own prompts) 18 lines differed in top-12 order and 242
exceeded the tolerance (worst 2.4e-4). b6f4667 makes the last layer's lens entry alias the served
logits tensor (src/llama-graph.cpp lens_build), so the rule holds byte for byte by construction and
the stack carries five layers instead of six. Mechanism check on a small Qwen3.5 (0.8B, GPU 1):
409 capture lines, 0 mismatches, worst probability delta 0.0; the a0bacec build passes that small
check too, because the drift only appears in verify batches (the stacked column count crosses the
kernel boundary there), which is why the head's own captures are the check that can fail.
