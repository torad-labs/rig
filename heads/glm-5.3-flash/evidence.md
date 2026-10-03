# glm-5.3-flash — where the numbers in head.toml come from

## The pack

- `[source]`: the Hugging Face API at revision `1bc0afad…` lists the three IQ3_XXS shards with
  these sizes and LFS sha256s (`/api/models/…/tree/<rev>/IQ3_XXS`, read 2026-09-28); the same
  values are in `docs/glm53-iq3-qualification.json`. The model card: MIT, not gated.
- The GGUF header of shard 1 (read by range request, 2026-09-28): `general.architecture =
  glm5next`, 46 blocks, 288 experts (8 used), MLA (`kv_lora_rank` 512) on every fourth block and
  KDA recurrent blocks between (`attention.head_count_kv` per block), `nextn_predict_layers = 1`,
  `context_length = 1048576`, `general.sampling.temp = 1.0`, `top_p = 0.95`, `penalty_repeat =
  1.1`, `penalty_last_n = 256`, `split.count = 3`.
- The engine reads `general.sampling.*` as its defaults (`common/common.cpp:1371` at the fork),
  so the head passes no sampling flags.

## The qualified run (`docs/glm53-iq3-qualification.json`, Sep 26, 2026, 6:56 PM CT)

Two RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB each, driver 615.71.09), engine
`a29b719`, `-c 524288 -np 1`, f16 K and V, `-b 4096 -ub 512`, `--spec-type draft-mtp
--spec-draft-n-max 2`, split by layer, `-ts 1/1`, the chat template in `assets/` (sha256
`a4fddbbf…`, the model's own with its indexing rewritten to what the engine's jinja supports).

- A 436,077-token cold prompt answered correctly (612.9 prompt tok/s, 34.6 generated), and again
  from the prompt cache in 3.6 s with 436,073 tokens cached: `context.model` is the served window
  of 524,288, half the trained 1,048,576.
- `-ub 512` against 128, A/B/A, three uncached requests each at 14,029 prompt tokens: 1,862.8 and
  1,869.5 prompt tok/s against 1,100.7, generation 85.4 and 84.8 against 86.4.
- Short-context generation 94.2 tok/s; a streamed and a non-streamed tool call round-trip.

## What the head changes from the qualified run

- `[engine]`: `76df278` (fork main at PR #78; its tree is `c988f052a`'s). Over the qualified
  run's engine it adds the routed experts' ring (`b8d44d4b8`) and the SwiGLU limit's fused clamp
  (`640a12c41`) for glm5next, as `a786bcd` did (the rental below ran that), and then: `ssm_a` loads
  on the card (`bb0e005fd`; on the CPU, its 34 tensors were copied into the card's split every token,
  one split more: rig-lead's 44-layer proxy of this model, same metadata, shapes and types, random
  weights, `-sm layer` on an RTX 5080, 52.8 tok/s in 2 splits against 50.3 in 3); the CUDA graphs a
  context keeps capped by their nodes, 32,768 by default (`83f230ebb`; read on the proxy, one card,
  `-sm layer`, `draft-mtp` at `n_max = 2`, 512 greedy tokens, `10bc46150`'s tree with rig-lead's
  uncommitted `ggml-cuda.cu` kernel work, capture and cap code as at `76df278`: the target context
  held at most 3 graphs, decode and the verify batches, 12,637 nodes and 36 MiB, and the draft
  context 6 graphs of 489 nodes; the real model's 45 layers come to about 12,900, and a layer split
  halves each card's. If that work's alpha/beta fold also matched glm5next's KDA layers, `76df278`
  runs on the order of 100 nodes a graph more; the first real run prints its own figure in the
  "CUDA graphs held at most" line, and the template's first box (below) read 5,020 nodes a card at
  `n_max = 2`. Each further draft length adds a target graph of about 4,300
  nodes, so an `n_max` of 7 or more, or more slots, would need the cap raised, to about
  (n_max + 1) × 4,400.
  `local/research/glm-moe-ring-2026-09-28/glm53/mtp-v1-layer-5070ti-server.txt`, rig-lead); and a
  meta tensor's device tensors kept until its address is initialized again
  (`dfa3319fc`, `-sm tensor` only). The real model runs on `76df278` from the template's first box
  (below). Its sm_120 portable build:
  `local/prebuilt/engine-builds/engine-sm120-76df278.tar.gz`, sha256 `8497437b…`, compiled Sep 28, 2026,
  8:02 AM CT (`tools/build-prebuilt.sh --sha`).
- `min_vram_mib = 92000`: the validation rental's per-card peak, 90,817 MiB on the second card, and
  a GiB of margin. A layer split at `-ts 1/1` is uneven: the first card peaks at 75,455 MiB.
- `split = "tensor"`, and `[engine]` at `ba0e6ec26` (`0eb0464ad`, PR #81, and the PDL restrict fix: no raw
  `__restrict__` on the hyper-connection front's two kernels or the rank-1 LoRA kernel, which a PDL launch
  could read before the previous kernel's writes; nothing else). The engine runs `-sm tensor` on glm5next
  from `1f004c9e4` (fork main at PR #79: the KDA and DSA layers split by head, 32 of each a card), and
  `0eb0464ad` adds the whole-evaluation CUDA graph under tensor, the hyper-connection kernels, the shared
  q8_1 input and KDA's q, k and v as one launch. Its real-model figures (the template's first box, below):
  `llama-bench` tg64 113.0 tok/s against layer's 81.6 on the same engine, and through `llama-server` at the
  served argv 134.5 tok/s at `n_max = 2` against layer's 102.6, 1.31 times. Its logits differ from layer's
  by a mean KLD of 0.052 (F32 all-reduce) to 0.056 (BF16), which the layer split's own batch-shape floor
  (layer against layer at another `-ub`: 0.055) does not separate it from: 4 chunks of 512, so no error
  bars tell the three apart (KLD 0.0551 ± 0.0024 for layer `-ub 256`, 0.0559 ± 0.0026 tensor BF16 wire,
  0.0524 ± 0.0026 tensor F32 wire; paired ln(PPL(Q)/PPL(base)) +0.010 ± 0.012, -0.009 ± 0.012 and
  -0.004 ± 0.012; KLD 99.9 % 0.568, 0.738, 0.858; 99 % 0.367, 0.393, 0.332; the tails of 2,048 tokens are
  a handful of tokens). Per layer tensor is 1.0 to 1.2 times that floor at each of the 45 layers. Open: the
  BF16 wire the head serves is tensor's own rounding term and is not yet shown inside the floor (a served
  tok/s A/B of the F32 wire and a 32-chunk paired run settle it; a peer review). The floor is the engine's
  arithmetic, q8_1 activation rounding and the routing flips it seeds, measured against a float64 reference
  in the section after them. The served environment sets `GGML_CUDA_ALLREDUCE=internal` under a tensor
  split (the all-reduce the capture is made over and every tensor figure here was measured with).
  `min_vram_mib` stays 92,000: a tensor split holds half the pack a card (64,036 MiB) and the same
  KV, compute and graph buffers on each, about 80 GB a card at the served `-c`, under the layer split's
  second card at 90,817 MiB.
- `cuda_graphs = 0`: under `-sm tensor` the meta backend runs about two subgraphs per layer on
  each card, each its own CUDA graph, and the default cap of 8 evicts every one before its reuse.
  rig-lead, 5080 + 5070 Ti, qwen3-0.6b q4_0, `llama-bench tg128 -ts 1/1`, three reps: tensor
  257.0 ± 20.0 tok/s at cap 8 against 558.0 ± 13.9 uncapped (layer 803.6 and 803.5). The rental
  ran the layer split uncapped and did not measure a cap against it.
- `[engine]` at `9c506a40d` (Sep 30, 2026, 8:34 PM CT): the engine the rerun box ran (`2e699cceb`, below) and the 10
  commits after it on the same line (head.toml names them). rig-lead's gate on that sha, built in a detached tree and run
  from its own bin: 13 `test-backend-ops` suites (MUL_MAT 1475/1475, MUL_MAT_ID 1076/1076, FLASH_ATTN_EXT 3229/3229 and
  the glm5next fusions), GATED_DELTA_NET_CACHE_FUSION with the state persisted at PDL on and off, `test-llama-archs`,
  `test-backend-meta-{split,views,capture,sourceless}`, and the 44-layer proxy under `-sm tensor -ts 1/1 -fa 1` with
  graphs on and off: 4 of 4 rows, no asserts. Not yet on the cards. One commit in it costs decode: `4a88ac993`, which
  gives each filler slot of the sparse mask's scatter its own dump column. On the proxy over the 5080 + 5070 Ti,
  `llama-bench` tg32 at depth 32768, `-r 6`, medians of reps 2-6, two interleaved pairs against its parent `0b0e97b5d`:
  85.79 / 80.11 and 84.92 / 84.13 tok/s (-3.6 % pooled). In an `nsys` node capture of the same run its op chain is 128
  more launches and about 146 us of kernel time a token on each card (about 1.1 % of the token), and the decode runs as
  two CUDA graphs a token, 30 to 46 us apart, where its parent ran one. Its logits at `-ub 512` differ from its parent's
  (the proxy on one card, `-c 8192`, one chunk: mean KLD 5.63e-4, same top token 94.07 %; the parent against itself 0)
  and at `-ub 3` and `-ub 1` sit at the parent-against-itself floor (`allreduce-ll/kld-fix.out`).
  Folding the liveness into the pool scores' top-k removes the chain and is the next change on that line.
- `[engine]` at `e11e67c29` (Sep 30, 2026, 10:48 PM CT): `9c506a40d` and the 8 commits after it on the same line (head.toml names
  them), the last of them that fold. Its gate on the 44-layer proxy (`local/research/glm-dsa-sparse-2026-09-30/scratch-gate-e11e67c29.log`, 10:11-10:47 PM CT): the build, 23 op and meta suites and the proxy under `-sm tensor` with graphs on and off, 25 legs, every one exit 0. The fold against its parent `12ef697e8` on the proxy over the 5080 +
  5070 Ti under `-sm tensor` (`local/research/glm53-tp-rental-2026-09-28/allreduce-ll/fold-cuda/`, its TORAD.md row):
  logits at `-ub 3` and `-ub 1` at the parent-against-itself floor and at `-ub 512` the 5.63e-4 of `4a88ac993`, the same
  fusion layout; the meta split's inputs 30 to 15 (the parent at the scheduler's cap); 1561.1 to 1465.4 launches a token
  on each card, kernel time -153 / -92 us a token (the `ffn_moe_gate` control -3.8 / -5.9). tg32 at `-d 32768`, graphs
  on, two interleaved pairs, +5.2 % and +2.8 %, is directional only: the second pair's reps spanned 61-84 tok/s on both
  binaries. On the cards: the box below (vast 53637080), +9.5 % at 32,768 cached tokens and -0.5 % at none.
- The served decode below, 8.9 ms a token against the bench's 7.05, is the indexer's context gate, not the server: the
  head scores the indexer whenever `n_ctx` exceeds 2051 (`glm5next.cpp`, gated on the context so the graph never changes
  topology mid-run), so the server at `-c 524288` with 512-token prompts ran it where every cell is selected anyway, and
  a bench with its own 128-token context did not. On the proxy (`local/research/glm53-image-2026-09-30/serve-gap-notes.md`):
  served at `-c 256` 10.33-10.89 ms a token, at `-c 8192` 11.47-12.06, and at equal depth served and bench agree (a
  6,228-token prompt 11.81-12.53, `-d 4096` 11.23-12.59). A Claude Code session is past 2051 tokens with its system
  prompt, so the bench to hold a served head to is one at the served depth.

## The template's first box (vast 53268003, Sep 28, 2026, 5:55-6:18 PM CT)

Two RTX PRO 6000 Blackwell Server Edition (97,887 MiB, 188 SMs, a 450 W limit, driver 595.71.05,
CUDA 13.2; PCIe through the host bridge, P2P not supported), rented from template 743917 as it
stands: image `glm-5.3-flash-sm120-e2354c3-14b1ab0b`, engine `76df278`, the layer split. The
template's on-start fetched, verified and served the head with no step of ours, 340 s after the
box's first command. Then, over the same pack, the candidate `0eb0464ad` (rig-lead's
`train/engine-10`, PR #81: the whole-evaluation CUDA graph under `-sm tensor`, the hyper-connection
kernels, the shared q8_1 input, KDA's q, k and v in one launch), built as the portable tarball with
`llama-perplexity` beside it, `GGML_CUDA_ALLREDUCE=internal`. One script ran every leg:
`local/research/glm53-tp-rental-2026-09-28/box-run.sh`, dry-run first in the template's image over
rig-lead's 44-layer proxy; the logs are under `box-53268003/` there. $1.15 of credit.

- Served as the template serves it (`76df278`, layer, `n_max = 2`), three prompts of 512 tokens at
  temperature 0: 96.9, 97.5 and 93.0 tok/s, draft acceptance 0.638.
- `llama-bench tg64 -fa 1`, five rounds alternated, three reps each: `0eb0464ad` tensor 113.02 tok/s
  (112.94-113.08), layer 81.59 (81.57-81.60); `76df278` layer 72.53.
- Through llama-server at the served argv (`-c 524288` fits both splits), the same three prompts,
  tok/s and acceptance per draft length:

  | `n_max` | layer | tensor |
  |---|---|---|
  | 1 | 96.7 (0.760) | 125.9 (0.769) |
  | 2 | 102.6 (0.611) | 134.5 (0.611) |
  | 3 | 97.6 (0.495) | 130.5 (0.505) |
  | 4 | 89.3 (0.387) | 126.0 (0.431) |

  2 stays the fastest under either split.
- CUDA graphs a context held at most: layer at `n_max = 2` 5,020 nodes a card; tensor 8,193 at
  `n_max = 2` and 12,738 at 4. The 32,768 cap holds.
- KL divergence against layer's logits, 4 chunks of 512 tokens (the fork's `TORAD.md`): layer again
  0.000000 (100 % the same top token); tensor 0.0559 ± 0.0026 mean, 0.029 median, 0.84 max, 86.9 %
  the same top token, perplexity 13.29 against 13.41; tensor with an F32 all-reduce
  (`GGML_CUDA_AR_BF16_THRESHOLD=0`) 0.0524, 87.4 %. The proxy measured 0.0059 and 0.0027. What
  the next box measured (below) is that layer against itself at another `-ub` differs as much, so
  the difference is not the split's, and that difference is itself too large.
- nsys over tensor decode, 30 tokens, against the 113.02 wall (8,848 µs): a floor of 4,297 µs a
  card, 48.6 % of the wall; the routed experts' gate and up (1,081 µs) and down (624 µs), KDA's q,
  k and v (1,285 µs) and the attention output (787 µs) lead.

## Where the KLD comes from (vast 53284921, Sep 28, 2026, 8:23-9:08 PM CT)

The same template and offer again, with three engines side by side: `76df278` (tensor with the
attention mirrored), `1f004c9e4` (the head split, PR #79) and `0eb0464ad`. The KLD legs were widened
to controls, and rig-lead's `layerdiff` (each layer's input for 96 tokens and the final logits, the
relative L2 error per layer, F32 all-reduce) ran over the served pack. Tables:
`local/research/glm53-tp-rental-2026-09-28/box2/`. $2.51 of credit, about 15 minutes of it my upload
of the engines.

- KL divergence over the text, 4 chunks of 512, base layer `-ub 512` on `0eb0464ad`: layer against
  itself 0.000000; layer `-ub 256` 0.0551 (86.2 % the same top token); tensor 0.0524, F32 wire, and
  0.0559 BF16; tensor at `-ub 1` 0.0526, at `-ub 3` 0.0545; with the whole-evaluation capture off
  (`GGML_META_CAPTURE_LEGACY=1`) and with every all-reduce by the chunked kernel
  (`GGML_CUDA_AR_COPY_THRESHOLD=0`) the same figures to the digit. Layer of `76df278` and of
  `1f004c9e4`, against `0eb0464ad`'s layer at the same split and `-ub`: 0.0555 for both, identical.
- `layerdiff`, prefill: layer `-ub 32` against layer `-ub 512` on one engine, the floor: after block
  0 (dense) 4.8e-7 median relative error, after block 1 (dense too: blocks 0-2 are dense, block 3 is the
  first MoE) 3.8e-3, then rising to
  0.17 by block 24 and staying between 0.15 and 0.18 to the last; logits KLD 0.0360, 92.7 % the same
  top token. Tensor against layer: 1.0 to 1.2 times that floor at every one of the 45 layers, KLD
  0.0359, 92.7 %.
- Decode (`-ub 1`): tensor against layer 0.0308, 91.7 % the same top token, 0.9 to 1.0 times the
  floor of layer `-ub 1` against `-ub 32` (0.0358, 96.9 %). That floor pair includes the BF16 projection
  path switch (below), which tensor against layer at `-ub 1` does not, so tensor sitting under it is
  expected, not a result.
- Across engines at one split and one `-ub` (layer, 512): `76df278` against `0eb0464ad` KLD 0.0278,
  92.7 %; the output of block 0 (dense, no router) differs by 6.3e-3 median where the same engine at
  another batch shape differs by 4.8e-7. `76df278` and `1f004c9e4` agree bit for bit in layer mode.
  On the 44-layer proxy the same comparison leaves block 0 unchanged (2.4e-7) and starts at block 1.
- The router's choices were not read: `layerdiff --router` aborts on the real model
  (`ffn_moe_topk-3 is not a contiguous I32 tensor`), so whether routing flips carry the growth is
  not known.

The generated text is coherent (the openings of two of the three served answers were read; draft
acceptance 0.6 to 0.8), so what is measured is a sensitivity, not garbage. Two runs of one engine that
differ only in batch size disagree on the top token in 7 % of positions and move the residual stream by
17 % by block 24; a change between `1f004c9e4` and `0eb0464ad` moved block 0 by 6e-3. The next section
explains block 0; the growth after block 9 is still open. The head serves by tensor regardless: the floor
is the same for both splits.

## Against exact arithmetic (local, real weights of blocks 0-3, Sep 28, 2026, 9-11 PM CT)

`mini/make-real-mini.py` range-fetches the real tensors of blocks 0-3, the embedding and the output
head from the pinned pack revision (4.9 GiB, fits the two 16 GB local cards), and `ref-block0.py`
runs block 0 in float64 on those very weights, teacher-forced on the same 96 tokens, and records every
intermediate under the engine's own tensor names. `layerdiff-v3 --dump-layer 0` dumps the engine's.
Tools and tables: `local/research/glm53-tp-rental-2026-09-28/mini/` (`ref-block0.py`, `ref-trace.py`,
`ref-replay.py`). The reference can fail: a mutant weight (`wq[100,200] += 0.05`) moves its output by
1.4e-4, and row 0 of every dump matches its embedding bit for bit.

- Every engine at every batch shape is 1.35e-2 (per stream; 2.6e-2 on the mean-of-streams row that
  layerdiff compares) from exact arithmetic after block 0, more than they differ from each other
  (3.4e-3 to 8.4e-3 where they differ at all). That distance is q8_1 activation rounding: each of block 0's five Q8_0 mat-muls
  (`kda_qkv`, `kda_out`, `ffn_gate`, `ffn_up`, `ffn_out`), replayed in float64 from the engine's own
  dumped input with the activation rounded to q8_1 per 32, matches the engine at 1.5e-7 (`-ub 512`
  and `32`, float scale) and 6e-8 (`-ub 1`, half scale); without the rounding the replay is 2.7e-3 to
  9e-3 off. No mat-mul is wrong; each rounds its activation by 3e-3 to 9e-3 (outlier blocks).
- Prefill and decode differ in the BF16 projections. At `-ub` 32 and 512 `hc_mixes`, `kda_gate` and
  `kda_beta` are 2.5e-4, 7.2e-4 and 9.8e-5 from exact (cuBLAS's BF16 GEMM rounds the F32 streams to
  bf16); at `-ub 1` they are 6e-8. Decode is the exact path, prefill the one that deviates (rig-lead
  is declaring F32 precision on them).
- The first split between two engines or two batch shapes is `kda_qkv`, not the HC ops. Even from its
  own dumped `attn_norm`, `kda_qkv` is 2.8e-4 off the q8_1 replay (the other four ops: 1.5e-7): 331 of
  393,216 input elements (90 of 96 tokens) are quantized one quantum away from the replay, each within
  8e-6 quanta of a rounding tie. Block 0's input is a Q8_0 embedding row times a scalar, so exact ties
  are structural there and the last bit of the normalisation decides them. Between `0eb0464ad` and
  `76df278` (hc_mixes bit-equal, `attn_norm` 2.8e-8 apart by rig-lead's read) that moves 16 tokens by up
  to 3.3e-4 at `kda_qkv`, and each later q8_1 mat-mul multiplies it (`-ub 32` against `-ub 1`: kda_out
  2.9e-3, SwiGLU 1.3e-2, block output 8.4e-3 median). The engine change is a rounding cascade, not a
  regression.
- So the floor is q8_1 rounding plus a cascade that reaches the noise level within one block. At the
  first MoE block (3, `flip-rate.py`, all 15 pairs of six runs: two engines at `-ub` 512, 32, 1; 96
  tokens) routing flips are the only thing that grows the divergence: 5 to 9 of 96 tokens change exactly
  one expert of eight, at 4.7e-3 to 6.7e-3 entering; a flipped token leaves at 1.6e-2 to 2.1e-2
  (median), a fixed chunk in every pair; a token whose route did not change leaves at 0.80 to 0.96 times
  what entered, so the block contracts the rest. A flip rate of about 12 times the entering
  divergence and a fixed injection of 2e-2 per flip, with that contraction, put the fixed point of one
  block near 2e-2; the whole-model curve reaches 0.17, so later blocks flip more or move more than
  block 3 does. That part is not measured: the mini ends at block 3, and the router of the real model
  has not been read (v3's `--router` fixes the abort; a box with it would give flips per block).
- What would remove it is arithmetic that does not depend on the batch shape (F32 activations, F32 in the
  BF16 HC and KDA projections, batch-invariant kernels), which is engine work and moves layer and tensor
  alike. The tensor split is no worse than layer's own batch-shape floor, so the head serves by tensor.

## The all-reduce as LL packets (local proxy, engine `1cbc66f49` and after, Sep 29, 2026, 12-2 PM CT)

Not in the pinned engine (`ba0e6ec26`) and not measured on the rental cards: the 44-layer proxy
(`glm53-proxy-L44-E8.gguf`, served shapes and types, random weights) over the local RTX 5080 (PCIe x8)
and RTX 5070 Ti (Gen5 x4), `-sm tensor`, 88 reductions a token. The x4 link makes bytes on the wire dear
here, so the size where LL stops paying moves up on x16 cards; the rental's own sweep of
`GGML_CUDA_AR_LL_MAX_BYTES` sets it.

- What it is: a 4 B word and the call's token in one 8 B packet, polled by the peer in place of the chunked
  kernel's write, fence, arrival int, fence and read. It runs the reductions up to `GGML_CUDA_AR_LL_MAX_BYTES`
  (16 KB on the wire, 0 disables); larger ones keep the chunked kernel and the copy engines. The first commit
  set 32 KB: the isolated cycle time crosses at about 20 KB (7.4 us against 7.9 chunked at 16 KB, 12.7 against 9.8
  at 32 KB), and a 4-token batch (32 KB on the wire) read 1 to 4 % below the chunked kernel.
- Same bits: byte-equal to the chunked kernel and to a CPU reference over 163 cases (F32, F16, BF16; odd and even
  counts; BF16 and F32 wire; 900 stress calls a wire with fresh data), and the logits of four 512-token chunks at
  decode batch 1 have KLD 0 (Max 2e-6, the tool's own floor, as the chunked run against itself) against the
  chunked run's, both wires. The same instrument reads 5.8e-3 (Same top p 80.7 %) for F32 wire against BF16 wire,
  so it can tell them apart.
- Speed: the later card's kernel median (nsys, graph nodes) 6.08 to 3.84 us; tg64 86.99 to 89.45 tok/s (+2.8 %,
  six alternating pairs, LL ahead in all six). At 2 to 4 tokens a batch (`llama-bench -p N`, not in a graph) LL
  and chunked read alike inside the +-3 % spread of the rounds.
- The wire type (alternating rounds, tok/s; the box swings +-2 % round to round, and by 20 % in whole runs when
  other work shares it): tg64, five rounds, BF16 wire 89.63 with LL and 87.82 chunked (86.5-89.1), F32 wire 87.57 with LL
  and 87.77 chunked (87.4-88.3). Batches of 2, 3 and 4 tokens, four rounds, chunked: BF16 139.2, 181.2, 218.1; F32
  138.5, 180.4, 207.1. So the F32 wire costs nothing measurable at decode and verification sizes up to 3 tokens.
  Prompt processing is where it costs (the copy-engine path moves twice the bytes over the x4 link): pp512 2,233
  with the BF16 wire, 1,809 with F32; pp2048 2,110 and 1,765; the layer split reads 1,761 (1,431-1,958) and 2,413.
  On the rental's x16 links the bytes are cheaper and this loss smaller, not measured. The KLD figures above are
  prompt-shaped (512-token chunks), and prompt processing writes the cache decode reads. The head serves the BF16
  wire until the box measures both, with the paired KLD. (Measured on Sep 30, below: BF16 stays.)

## LL under the paced L2 issuer, and the conv's weights as segments (local proxy, engine `f1709b6a9`, Sep 29, 2026, 2:30-4:20 PM CT)

Same proxy and cards as above, `-sm tensor`, tg64, three runs each, six rounds with the four arms interleaved so drift
cancels (the first attempt of this pair, taken while another job thrashed the box's memory, was discarded).

- The issuer (`345065068`, rig-lead's, on by default) adds 5.8 tok/s to the chunked kernel, +6.7 % (87.46 to 93.70
  median, six of six rounds ahead); its off switch is `GGML_CUDA_L2_ISSUE_LEGACY=1`.
- LL against chunked is not separable on this link: with the issuer on 93.53 against 93.70 (paired difference -0.6 +-
  1.3 tok/s, one LL round at 86.07), with it off 89.17 against 87.46 (+1.2 +- 0.9). The reduction itself is not slowed by
  the issuer: rig-lead's node trace has `ggml_cuda_ar_ll_kernel` at 349 us a token with it on and 341 off. The x4 link
  is the local cap on LL's win; the rental's own limit sweep settles the default.
- The fused conv reading KDA's three conv weights itself (`831bac6a7`): glm5next builds the SSM_CONV's weights every token
  as `CONCAT(CONCAT(q, k), v)`, 132 launches a token over the two cards, and the matcher declined a conv when a node after
  the state's CONCAT had its output over `x`, which the weights' CONCAT did on the proxy over one card and in 68 of 1,122
  conv launches on the two cards. Now the kernel reads the three weights and the CONCATs are skipped: 56,024 launches in
  the traced run (34 decoded tokens) become 53,576 on the two cards, and logits of four 512-token chunks are byte-equal with the switch off
  (`GGML_CUDA_SSM_CONV_W_SEGMENTS_LEGACY=1`) at batch 1, 4 and 512 under `-sm tensor`, `-sm layer` and one card. tg64 on
  one card in `-sm tensor` over CUDA0 twice, alternating, 16 runs each: 62.94 to 63.69 tok/s (+1.2 %).

## The rerun box (vast 53547560, Sep 30, 2026, 10:41 AM-12:15 PM CT)

Two RTX PRO 6000 Blackwell Server Edition (97,887 MiB, 188 SMs at 2430 MHz, L2 131,072 KiB, DRAM 1,598 GB/s; one PCIe
switch apart, P2P ok, `canAccessPeer` 1, `nativeAtomics` 0, driver 580.95.05), offer 33665866 at $3.47 an hour (the
download rate of the offer I refused the day before was $0.051 a GB; this one $0.0027), the template's image at engine
`2e699cceb` and the pack it serves (glm5next 313B.A17B, IQ3_XXS, 125.07 GiB). Every figure is one `llama-bench` process of
three repetitions, `-sm tensor -ts 1/1 -fa 1`, CUDA graphs on, tg64 unless said, from `box3/box-run3.sh`; a pair is two
processes alternating. Raw results: `local/research/glm53-tp-rental-2026-09-28/box3/results-53547560/`. The bill was
$5.78 (credit $99.77 to $93.99); the box the day before cost $7.41 and produced nothing.

- Decode by split: tensor 142.8, 141.6, 141.7 against layer 95.8, 95.7, 96.1 (+48 %). Prefill, pp512 and pp2048: layer
  1475.8 and 1871.7, tensor with the BF16 wire 2360.5 and 2206.9, with the F32 wire 2203.9 and 2056.8 (-6.6 %).
- Served, the template's head (tensor, MTP `n_max` 2), three prompts of 512 tokens: 138.2, 133.5, 128.2 tok/s (draft
  acceptance 0.68, 0.66, 0.61). The same head without speculation: tensor 113.0, 112.4, 112.5; layer 78.9, 80.1, 80.6. MTP
  at 2 is +18 % over plain under tensor. `n_max` under tensor, prompts 1-3: 1: 114.3, 115.9, 110.6; 2: 136.5, 132.0, 123.2;
  3: 102.5, 109.7, 102.7; 4: 118.2, 125.4, 102.2; 5: 93.2, 100.2, 83.5; 6: 103.9, 113.5, 85.7; layer at 2: 96.1, 100.6,
  102.4. A verify batch is dear on the MoE: one token 7.1 ms, two 12.5, three 13.9, four 16.0, six 19.4 (the
  `-p 2..6` rows below), since each token brings its own experts.
- Served decode is 8.9 ms a token against the bench's 7.05, under both splits (layer 12.4-12.7 against 10.4). It is not
  the server's log: at verbosity 0 the plain tensor head gives 114.2, 112.5, 113.1; nor its sampler chain: with `top_k` 1,
  `top_p` 1, `min_p` 0 it gives 111.1, 109.2, 109.1. It is the context size. `glm5next.cpp:647` turns the indexer's scoring
  on when `n_ctx` exceeds `indexer_top_k + kpool - 1` (`llama-graph.cpp:608` the same), fixed per context, so a served
  context (the head's `-c 524288`) scores on every token however few are cached, and a `llama-bench` at depth 0 (a context
  of 64 tokens) does not score at all. On the 44-layer proxy over the 5080 + 5070 Ti, `-sm tensor`, the server's plain
  decode is 9.98 and 9.98 ms a token at `-c 1024` and 10.78 and 10.58 at `-c 4096` (alternating, the same 8 to 264 cached
  tokens), and a capture of the two contexts has the lightning indexer (114 us a token), its top-k (38 us), and about
  235 us of extra matrix-vector launches only above the gate; device 0's whole-token graph runs 10,002 us in the bench
  and 10,893 us in the server. The bench at depth 2048, which is over the gate, is the decode the server sees: 126.6 and
  119.2 tok/s at 2048 and 8192 in the issuer rows, against 139.3 at depth 0. Every depth-0 row in this section
  therefore leaves the indexer out. The rest of the gap (7.9 ms at depth 2048, 8.9 served) is the host: on the proxy's
  9800X3D at 5.3 GHz the device idles 283 us a token in the bench and 671 us in the server between two graphs, and each
  device's graph launch takes 286 us (393 served) and is issued after the other's, so device 1 starts 274 us (404) later;
  launching both at once removed that skew (to 4 us) and moved the token by nothing (10,254 us serial, 10,227 to 10,276
  concurrent), so the skew is not on the critical path there. The box's Xeon 6767P is slower per core.
- Decode at depth, same proxy and cards (single runs while another seat's jobs loaded the host, so the ratios, not the
  levels): `llama-bench` tg32 at depth 2048, 16384, 32768, 65536 is 84.3, 82.6, 76.6, 62.5 tok/s (11.9, 12.1, 13.1, 16.0 ms
  a token). A graph-level capture of the last 31 tokens, graph plus the idle before the next: depth 0 10,870 us, 2048
  10,666, 32768 11,587, 65536 13,264; the idle alone 769, 482, 628, 1,619 us a token, the graph 10,101, 10,183, 10,959,
  11,645. Node ranges, card 0, per token: flash attention 116, 197, 410, 599 us at depth 0, 2048, 8192, 32768 (its live-step
  path scans the mask of the whole cache: it is not flat under the indexer's top-k); the indexer's own kernels stay small
  (lightning 21 us and top-k 27 us at the turn-on, projections 159 us); the rest of the turn-on is the paced issuer's
  launches, gathers and copies. The prediction I declared (lightning +110 us at the turn-on; flash attention flat from
  2048 to 8192) was wrong on both. Host stacks of the server at 64K cached tokens (gdb, 400 samples): 86% of the active
  samples are the main thread in the device sync and 14% host work, the largest `llama_kv_cache_set_input_kpool`, which
  makes about eight passes over the cache's cells each token to rebuild the pool maps and the two masks; with a few
  hundred cached tokens (800 samples) it is 0.2% of them.
- The pool maps from the cells that changed (engine `0b7316af7`; `LLAMA_KPOOL_INPUT_LEGACY=1` is the old build from
  every cell). The host step alone, `llama_kpool_set_input` at 1 token, 1 sequence, per call, views against cells: 1.1
  against 12.9 us at 2048 cached tokens, 3.8 against 49.1 at 8192, 16.5 against 223.7 at 32768, 37.7 against 458.4 at
  65536 (3 tokens at 65536: 77.2 against 449.1). Output identical byte for byte (the unit test's 420 random rounds; the
  server on the proxy with `LLAMA_KPOOL_INPUT_CHECK=1` over prompt reuse, tail trims, a slot reset, a context shift and
  MTP verifies: more than 9,700 checked calls, no difference). End to end on the proxy under `-sm tensor`, `llama-bench` tg64 at
  depth 32768, three interleaved pairs, views against cells: 85.37, 84.22, 84.93 against 83.69, 83.12, 84.79 tok/s
  (+2.0, +1.3, +0.2 %; the 207 us saved of an 11.8 ms token is 1.7 %, worked out after the result). At depth 65536, four
  pairs: 81.0, 75.7, 66.5, 75.2 against 66.7, 77.7, 76.5, 74.1, and the device idle between graphs (nsys, the last 31
  tokens, device 0, mean, two runs each) 718, 937 against 1,051, 992 us at 65536 and 729, 491 against 581, 597 at 32768:
  no reading, the host's load average was 8 to 11 (other seats' builds). The box's Xeon 6767P, slower per core, is not
  measured with it.
- Pool alignment against upstream's glm5-next (ggml-org `f872b5911`, the same proxy GGUF with the arch renamed), which
  regroups pools from the sequence's first position where ours anchors at the absolute p/kpool. Which cells the
  indexer leaves unmasked for the last rows of a 6000-token prefill, 11 indexer layers, Jaccard of the two sets: no
  edit 0.93 mean (0.90 min; ours against itself at another ubatch size 0.91, upstream 0.90, top_k halved 0.50), so the
  selection agrees. After a head cut that is a multiple of kpool, a shift (`seq_rm` of the head then `seq_add` of the
  rest) or the server's context shift (n_keep 1, n_discard 1001): 0.94, 0.94, 0.94. After an API head cut `seq_rm(0,
  1002)` with no shift (first position 1002, not a multiple of 4): 0.38 (set 2048 against 2050 cells), and cells 1002
  and 1003, the head block's two cells, are selected in no layer by ours and in at least one layer by upstream. The server
  cannot reach it: the context shift (`server-context.cpp:3440`) and cache reuse (`:3767`) both keep the sequence
  starting at position 0. Left as is; the anchor stays absolute.
- The paced L2 issuer, one pass each in the order off, v1 (the default), busy (`GGML_CUDA_L2_ISSUE_BUSY_KB=1024`), at
  depth 0, 2048, 8192: 135.2, 119.5, 113.2; 139.3, 126.6, 119.2; 142.0, 127.2, 120.2. The default measured 141.1 to
  141.7 at depth 0 in five other steps, so off is about -4 %, busy is level with the default at depth 0 and +0.5 % deeper;
  off ran first in its step.
- The LL limit (`GGML_CUDA_AR_LL_MAX_BYTES`), tok/s at `-p 2`, `-p 3`, `-p 4`, `-p 6`, tg64: chunked (0) 152.5, 216.4,
  253.2, 309.6, 133.8; 16 KB 160.2, 219.2, 251.7, 312.9, 141.1; 32 KB 159.7, 224.6, 243.7, 307.1, 141.4; 64 KB 151.6,
  223.8, 243.2, 314.3, 141.4; 128 KB 157.4, 224.6, 251.2, 307.9, 141.1. LL is +5.5 % on the one-token reduction (12 KB of
  BF16 wire); from two to six tokens nothing separates from the +-3 % between identical runs (the 16 KB row is the
  chunked kernel at `-p 2` and reads 160.2 against 152.5). The default 16 KB stays.
- GDN state persist (`GGML_CUDA_GDN_STATE_PERSIST`), three interleaved rounds at g128: off 145.9, 146.6, 143.7 and on
  145.1, 145.4, 146.0 at depth 0; at 8192 off 123.8, 124.1, 121.6 and on 122.5, 123.2, 123.5. Neutral. The check shows the
  window on the one 46-node decode graph of each card (74,528 KiB, hit ratio 1.000) and none on the others.
- Pairs decoded once (3dbefa8fb), `-p 2,3,4 -n 0`, two rounds with the order reversed: default 162.3, 159.9 / 216.2, 216.5 /
  245.9, 248.5 against `GGML_CUDA_MMVQ_MOE_PAIRS_LEGACY` 160.4, 161.9 / 221.3, 221.9 / 250.7, 251.5: equal at two tokens,
  the legacy +2.4 % at three (the `n_max` 2 verify shape) and +1.4 % at four.
- Each switch once at tg64, between two default runs (141.3, 139.7): KDA cache 141.3, weighted sum 140.7, Q8_1 producer
  140.0, IDS early 137.9, issuer stop 140.0, top-k alias 139.6, conv weights as segments 138.4, pairs 138.9, hyper-connection
  comb side 145.5. The two that moved more than the drift, interleaved four times with the default: the comb side's
  legacy ahead in 4 of 4, 145.4/141.7, 144.0/140.4, 146.2/138.7, 147.1/141.3 (+3.7 %, so the default is the slower; rig-lead's
  plan is to flip it unless it wins locally), and the conv's segments (831bac6a7) ahead of their legacy in 4 of 4, 141.8/140.3, 142.4/137.8, 140.2/138.0,
  141.9/140.9 (+1.6 %).
- The state buffer under tensor is split across the cards: `RS buffer size` per device 145.56, 218.34, 291.12, 363.91,
  436.69, 509.47 MiB at `n_max` 1 to 6 (+72.8 MiB a draft token; the two cards of layer at 2 add to 436.69).
- KL divergence, 32 chunks of 512 tokens, the layer split's logits as the base (a `llama-perplexity` of the same commit
  built in rig's Ubuntu 22.04 image, `box3/build-kld-portable.sh`; the first box's build needed glibc 2.38):

  | arm | mean KLD | same top token |
  |---|---|---|
  | layer again | 0.000000 (max 0.000062) | 100 % |
  | tensor, F32 wire, again | 0.000000 (max 0.000066, read against its own first run) | 100 % |
  | layer `-ub 256` | 0.07139 | 85.03 % |
  | tensor, F32 wire | 0.06964 | 85.50 % |
  | tensor, BF16 wire | 0.07516 | 84.77 % |
  | tensor BF16 against tensor F32 | 0.07239 | 85.15 % |

  Paired over the 32 chunks (95 % CI): BF16 minus F32 against layer +0.00552 (+0.00153 to +0.00951) in KLD, +0.0066
  (-0.0055 to +0.0187) in the log perplexity ratio on the text's own tokens; layer `-ub 256` minus tensor F32 +0.00175
  (-0.00091 to +0.00441), BF16 minus layer `-ub 256` +0.00377 (-0.00116 to +0.00870), and on the log perplexity
  ratio `-ub 256` against tensor F32 +0.0142 (+0.0029 to +0.0255). The reviewer's declared rule (BF16 worse than F32 by more than
  0.002 with a CI off zero means F32 for prefill) is met on its letter; the control it did not declare, an arithmetically
  equal recompute at another micro-batch, lands as far from the base as either wire, and the wire read directly against F32
  is at that floor. The reviewer withdrew the rule after reading the tables: BF16 for every phase, the knob
  `GGML_CUDA_AR_BF16_MAX_BYTES` I had begun was dropped uncommitted. The layer `-ub 256` floor was 0.0551 on four chunks
  at `0eb0464ad` (above).
- Roofline of the capture (tg32 under `nsys` with the engine's NVTX node ranges, graphs off, 30 tokens; rig-lead's
  `roofline-proxy.ts`, wall 7,057 us): the bytes of a token at 1,598 GB/s take 4,255 us a card, 60.3 % of the wall. Over
  its floor, per token on card 0: the routed experts' gate and up 964 us against 684 (+281 by the median), down 511
  against 342 (+171); the two hyper-connection groups 264 and 261 us against 36 each (+452 together, 6.4 % of the wall:
  the `dsv4_hc` kernels); the router's top-k 155 us against none; attention 126; the shared experts' gate and up 330 and
  down 185; the conv 101. KDA's q, k, v (944 us) and the attention output (537) read below their byte floors, which the
  L2 persisting windows explain. The paced issuer's own kernel, outside any node, is 1,774 us a token on a side stream.
- Not done: the GDN state-persist arm at the newer engine, `MMVQ_MOE_PAIRS_LEGACY` at `2606ae528` (rig-lead asked; the
  box was not extended for it), the second KLD round (`R2`) and the nsys roofline of the layer split.

## The pin on the cards (vast 53637080, Sep 30, 2026, 11:27-11:52 PM CT)

Two RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB, 500 W, driver 615.71.09) in Norway, one NUMA node, P2P
between them (`cudaDeviceCanAccessPeer` 1 both ways, where the earlier boxes had none), rented by `rig vast up
glm-5.3-flash --template --hours 2 --budget 12.50` from template 743917 on image `glm-5.3-flash-sm120-2fbb839-c42b3166`
(881 MB, engine `e11e67c29`). $3.244/h and $0.040 a GB down; $6.90 of credit in all, most of it the pack's download.
Created at 11:27:34, ssh at 11:28:31, the pack fetched at ~270 MiB/s over two shards, serving at 11:34:52: 7 min 18 s
with no step of ours. One script ran the legs (`local/research/glm53-image-2026-09-30/box4/box-run4.sh`, dry-run first on
the proxy); the results are `box4/out4-53637080.tgz`.

- The pin against the engine the cards last measured, `llama-bench tg64 -sm tensor -fa 1`, three rotated rounds of
  three reps: at no cached tokens `2e699cceb` 171.8 / 171.3 / 171.7 tok/s and `e11e67c29` 170.6 / 170.8 / 170.8 (-0.5 %);
  at 32,768, 134.5 / 134.2 / 134.1 and 146.9 / 147.1 / 146.9 (+9.5 %). The pin stays: a Claude Code session decodes
  deep.
- Served as the head serves (tensor, MTP `n_max = 2`, `-c 524288`), each engine: three 512-token answers to short
  prompts, 169-186 tok/s on both; two answers of 256 tokens after a 27,642-token prompt, `2e699cceb` 156.0 / 163.4 tok/s
  (drafts 151/205, 153/203 accepted) and `e11e67c29` 186.6 / 196.8 (156/196, 161/187); that prompt's prefill 1,632-1,640
  against 1,677-1,685 t/s, 16.4 s before the first token.
- The template's own server, the box's first CUDA process, took 7.51 s to prefill its second request (43 tokens, 39 of
  them new after the warm-up's 10), where each later server, the same argv, env and slot reuse, took 0.28 s. Not a JIT:
  the prebuilt carries SASS for sm_120 and sm_120a and no PTX. One sample, its cause not found.
- The weights' upload, the server log's model buffer to its context: tensor 20.4 / 20.5 s, layer 6.5 / 6.4 s (3.2x), each
  load under `nsys -t cuda` in `out4/load/`.
- `-ub` under tensor, two rotated rounds, pp2048 / tg64: at no cached tokens 2,340 / 2,310 t/s at 512, 3,153 / 3,145 at
  1024, 3,724 / 3,725 at 2048 (+60 %); at 32,768, 1,809 / 1,785, 2,243 / 2,247, 2,533 / 2,523 (+41 %), the decode there
  152.1 / 152.2, 151.0 / 150.9, 149.8 / 149.4 (-1.7 %). The served buffers a card at 512: weights 64,036 MiB, KV 9,856,
  recurrent state 218, compute 3,204, the MTP context's KV 896 and compute 2,495; about 15 GB free. Compute buffers grow
  with `-ub` (the proxy's: 439 MiB at 512, 1,756 at 2048), so 2048 wants about 17 GB more a card and 1024 about 5.7.
  Whether the served `-c` and the MTP context still fit was the leg the box lost.
- `capture32k`, for rig-lead's lever attribution on headless cards: graphs on as served, tg32 at 32,768 cached tokens,
  `nsys -t cuda --cuda-graph-trace=node`, 314,845 kernels a card, 249,455 of them graph nodes, 105.2 tok/s under the
  trace (`out4/nsys/graphs-d32768.sqlite`).
- Not done: the box was destroyed during `-ts` (1/1 at 32,768 read 151.6), so the other splits, MTP at `n_max` 1-3, the
  NVTX capture and the served `-ub` leg did not run. The driver had been paused with SIGSTOP to add that leg; the
  harness that started it took the stop for its end and its teardown ran the trap's `vast down`. The driver now holds on
  a file (`box4/rent4.sh`), and `ubserved` is a step of `box-run4.sh`.

## The lost legs and the served `-ub` (vast 53720656, Oct 1, 2026, 11:00-11:50 AM CT)

Two RTX PRO 6000 Blackwell Max-Q Workstation Edition (97,887 MiB, 300 W, driver 610.43.02) in California, no P2P
(`cudaDeviceCanAccessPeer` 0 both ways), on an Intel Core Ultra 7 270K Plus, rented by `box4/rent4.sh` from the same
template under a $5 cap for an hour: $2.711/h and $0.003 a GB down. Created at 11:00:55, serving at 11:24:46 (the pack
over a 902 Mb/s link), destroyed at 11:50:02 after 0.82 h, ~$2.22. `box-run4.sh` ran `served ubserved ts mtp capture`;
the results are `box4/out4-53720656.tgz`.

- The served `-ub`: the head's argv with `-ub` alone changed, a fresh server each, two rotated rounds (512, 1024, 2048,
  then reversed), three 512-token answers to short prompts and two 256-token answers after a 27,642-token prompt. At
  512, 81,960 MiB a card in use and the deep prefill 20.93 / 20.71 s and 21.18 / 20.89 s. At 1024, 87,816 MiB a card,
  10,071 free, and 16.22 / 16.01 s and 16.31 / 16.01 s: 22.9 % less, 1,695-1,727 against 1,305-1,335 t/s. 2048 did not
  come up at `-c 524288` (`allocating 9970.57 MiB on device 0: cudaMalloc failed: out of memory`); at `-c 131072`,
  75,618 MiB a card and 14.20 / 14.15 s twice.
- Decode by `-ub`: each answered every prompt the same way in both rounds. The short prompts' answers are the same text
  at 512 and 1024 (drafts 283/454, 283/455, 263/495 at both), decoded at 160.5 tok/s on average at 512 and 154.7 at 1024
  (-3.6 %, lower at 1024 in 5 of the 6 pairs), two servers an arm; the box's other two servers at 512 (the template's
  own, and `mtp`'s `n_max = 2`) averaged 159.6 and 160.0. The bench on 53637080 put it at -0.8 % at 32,768 (above). The
  deep answers part at their 57th and 176th character between 512 and 1024, so their decode is not compared.
- So the head serves `-ub 1024`. A turn that adds N tokens at depth and generates G saves 0.107 (pp2048 at 32,768, the
  bench) to 0.173 ms (the 27,642-token turn) a token of N, and costs 0.05 (the bench's -0.8 %) to 0.23 ms (the served
  -3.6 %) a token of G: a Claude Code turn, a tool's output in and a call out, adds more than it generates, and a cold
  27.6k-token start is 4.8 s shorter. 1536 was not measured: about 93.7 GB a card by the step from 512 to 1024.
- `-ts` at 32,768 cached tokens, tg64 rotated: 1/1 134.6 / 134.9, 47/53 129.1 / 129.2, 44/56 128.0 / 128.0. The even
  split stays. The same engine on 53637080's 500 W cards read 146.9-147.1: the Max-Q's 300 W decode 8.3 % slower deep.
- MTP by `n_max` under tensor, a fresh server each: the short answers at 1 160.5 tok/s on average, 2 160.0, 3 143.8; the
  deep turns at 1 161.9 / 161.4, 2 169.0 / 181.3, 3 159.4 / 164.0. The layer split at 2: 112.2-119.1 short, 120.7 /
  131.7 deep, and its deep prefill 887 / 1,140 t/s against the tensor split's 1,304-1,329. `n_max = 2` stays.
- The NVTX capture, graphs off, tg32 under tensor: 88.93 tok/s under the trace, `out4/nsys/nvtx-tensor.sqlite` (32 MB).
- The template's own server prefilled its second request (43 tokens) in 0.35 s here: 53637080's 7.51 s did not recur.

## The validation rental (vast 53163829, Sep 28, 2026, 6:16-6:26 AM CT)

Two RTX PRO 6000 Blackwell Server Edition (97,887 MiB each, 188 SMs, 1,598 GB/s), rented from the
head's template: the image `rig image glm-5.3-flash --push` published at `a953e34`, engine
`a786bcd` (no NCCL: the internal all-reduce), the pack fetched and verified by `rig up`. Each leg is
llama-server with the argv of `rig serve glm-5.3-flash --plan`, one change at a time: three chat
prompts of 512 generated tokens at temperature 0, then an 11,414-token prompt.

- `-sm tensor` aborts at the first decode: `ggml-backend-meta.cpp:476
  GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer))`, with or without MTP, and in plain
  `llama-bench -sm tensor`. rig-lead has it.
- `-sm layer`, generated tok/s and draft acceptance: n_max 1 83.5 (0.733), n_max 2 85.9, 85.7,
  85.5 and 85.8 (0.554), n_max 3 84.3 (0.47), n_max 4 78.7 (0.38). Without speculation
  (`llama-bench tg128`) 69.7 ± 1.3.
- Prompt: 1,592-1,622 tok/s over the 11,414 tokens. The same prefix with another question after it
  computed 515 tokens in 0.41 s: the prompt cache serves it.
- Load: 15-18 s from the page cache.

> **SUPERSEDED — read "The ceiling" and "THE TOKEN BOUNDARY IS RETIRED" below before believing anything in the next
> five sections.** They chase a 444.6 us per-token "host boundary" that **does not exist**: it was manufactured by nsys,
> which inflates CUPTI RUNTIME durations 30-40x on this machine and slows the host until it stops running ahead of the
> device. In-process timing (`LLAMA_TIME_DECODE=1`, engine `acfa0196f`) shows `decode()` returning in 1.32 ms against a
> 10.08 ms token — the host is idle 87 % of the time. What survives from these sections is only what rests on DEVICE
> timestamps or on profiler-free walls: the exclusive-occupancy table, the all-reduce being fully exclusive, the decode
> window's identification, and the 4-bit tiering's **+10.93 %**. Every host-side share in them is void. They are kept
> because the sequence of errors is the useful part, and because two of the fixes they motivated are committed.

## Where the decode token goes, and what a graph launch costs (box 53637080's node trace + local proxy, Sep 30 - Oct 1, 2026, 11 PM-3 AM CT)

Read from `out4/nsys/graphs-d32768.sqlite` (the pin-on-the-cards box, headless, both cards identical, `cudaDeviceCanAccessPeer`
1 both ways, engine `e11e67c29`, `-sm tensor -p 0 -n 32 -d 32768`, `nsys -t cuda --cuda-graph-trace=node`). Scripts:
`local/research/glm-dsa-sparse-2026-09-30/{kernel-sum,span-share,idle-gaps,l2-nodes}.py|sh`.

**The decode window, stated because it is easy to get wrong:** the capture holds 97 bursts of 11 `flash_attn_ext_f16`
launches (11 DSA layers a token). 66 are spaced >40 ms apart -- those are prefill batches at `-ub` 512, which emit 11
each exactly like a decode token. Only the **30 bursts spaced 4-8 ms, median 6.909 ms (144.7 tok/s)** are decode. Taking
"the last 31x11 launches" as 31 decode tokens is wrong by one prefill burst; and nsys costs only 109 us a token here,
6.909 ms against the 6.80 ms clean 147 tok/s, which is what makes everything below provably not an artifact.

**A CUDA graph launch costs ~0.14 us a graph NODE, linear** (local, both cards, `-d 2048`): the 4-layer proxy at 147
kernels a token gives a 22.4 us median `cudaGraphLaunch`; the 45-layer proxy at 1466 gives 204.0 us. Nodes x10.0, cost
x9.09. On box4's Xeon it is 278.2 us over ~1481 nodes = 0.19 us a node -- the same law on a slower core. **Host
contention inflates it about 3x** (204 us quiet against 564-695 us with other seats at loadavg 30), so host-side numbers
from differently-loaded runs are not comparable.

**The token's largest single cost is the host boundary: 752 us, 10.9 %.** Exactly 30 gaps above 100 us over 30 decode
tokens, every one at the `mul_mat_vec_q -> k_bin_bcast` boundary (the output head's logits, then the next token's first
op), median **752.8 us** (707.0-1060.1). Inside it `cudaGraphLaunch` runs **2.00 times a token at 278.2 us median =
567.1 us**, with **no `cudaGraphExecUpdate` or `cudaGraphInstantiate` anywhere in the decode region** -- the graph is
reused unchanged, so this is intrinsic launch cost. `cudaStreamSynchronize` totals 6166 us a token at a 2.6 us median:
the host waiting on the device, not a cost. Two launches because `-sm tensor` issues one graph a device, and
`ggml-backend-meta.cpp:2607` runs them **serially on one host thread** while the device is idle for both. The same gap at
the same boundary appears on the dev pair: 551.4 us of an 11.044 ms token quiet (408 of it launch), 2009.0 us contended.

**EXCLUSIVE occupancy, which is what a lever can actually save** (`span-share.py`; the kernel-time sum is 8.627 ms
against a 6.883 ms span, so 1.744 ms already runs concurrently and any lever sized from the sum is overstated):

| kernel | total | EXCLUSIVE | % of span |
|---|---|---|---|
| mul_mat_vec_q | 2.616 | 2.192 | 31.9 % |
| mmvq_moe | 1.535 | 1.041 | 15.1 % |
| mul_mat_vec_f_vec | 1.256 | 0.110 | 1.6 % |
| ggml_cuda_ar_ll_kernel | 0.309 | 0.309 (all) | 4.5 % |
| flash_attn_ext_f16 + topk_radix + lightning_indexer | 0.347 | 0.347 (all) | 5.0 % |
| rms_norm + quantize_q8_1 + unary_gated + k_moe_weighted_sum + k_bin_bcast | 1.274 | **0.078** | **1.1 %** |

So the elementwise pool is worth 0.078 ms as kernel time, not the 1.274 ms its sum shows. **It is worth ~181 us as
LAUNCH time instead**, because its 481 launches are 32 % of the graph's 1481 nodes. That also explains the `ab5` e9
result retroactively: removing the remaining `quantize_q8_1` launches moved the wall -0.02 %, and those ~124 nodes are
worth 0.3 % of the token -- below that measurement's noise floor, not zero. **The pool only pays as a batch**; any single
fusion is ~0.3 % and unmeasurable. Its adjacencies are mechanical: `quantize_q8_1 -> mul_mat_vec_q` 127.7 a token,
`rms_norm -> unary_gated_op -> quantize_q8_1` 35.0, `mmvq_moe -> k_moe_weighted_sum -> mul_mat_vec_q` 42.3,
`k_bin_bcast -> k_bin_bcast` 14.3; 153.3 pool-to-pool adjacencies a token, and ~169 of the 481 run on grid <= 4.

**The all-reduce is 0.309 ms a token and 100 % exclusive** -- it serialises completely. `allreduce.cu:14-19` stages
through pinned host memory by design, for setups without NVLink, and `GGML_CUDA_P2P` only enables peer access for memcpy
and VMM, never the reduction. Box 53637080 is the first box to report peer access both ways, so a peer-direct reduction
is newly possible; the dev pair is PHB and cannot test it.

**A 4-bit role-based requantization of the non-expert tensors cuts per-token reads 11.860 -> 8.881 GB (-25.1 %)** and
measured **+10.93 %** on the dev pair (three rounds, one passed a pre-declared 4 %-spread gate: 90.68 -> 100.59 tok/s;
the two rejected rounds read +16.97 % and +24.41 % because contention depressed their Q8_0 arms). Predicted **+7.3 %** on
2x PRO 6000, from `mul_mat_vec_q`'s 2.192 ms exclusive occupancy x the measured -23 %. Tier A (`attn_output`, KDA
`attn_q`, shexp gate/up, MLA `q_a`/`q_b`, dense gate/up, `nextn.eh_proj`) -> IQ4_XS; Tier B, the tensors written INTO the
KDA recurrent state (`attn_k`, `attn_v`, `ffn_down_shexp`, dense `ffn_down`) -> Q6_K; Tier C, experts and BF16 indexer
untouched; head Q6_K. At the output head, the one shape exceeding this card's 67 MB L2 in every format, all five
candidates achieve **905-912 GB/s** (0.7 % spread, 95 % of peak) with time tracking bytes to 0.4 pp, so 4-bit carries no
kernel penalty at M=1 and format choice is purely bits-per-weight: Q4_K 393.7 us against NVFP4 393.4 us is a dead heat,
and IQ4_XS/MXFP4 at 4.25 bpw beat both by their byte ratio. **NOT a quality claim** -- measured on a random-weight proxy.
Quality needs the position-binned dKLD (`LLAMA_KLD_FIRST`, engine `a61c58f5c`) on real weights, binned 0-512/512-4k/4k+,
because 512-token chunks restart from cleared recurrent state and structurally cannot see KDA error accumulate.

### CORRECTION to the section above (Oct 1, 2026, 4:10 AM CT): the graph-launch numbers were an instrument artifact

**The "~0.14 us a graph node, linear" law is WRONG and must not be used. `cudaGraphLaunch` does not scale with node
count.** It was measured under `nsys --cuda-graph-trace=node`, whose own overhead is per graph node -- an instrument
whose cost scales with the quantity being varied. Re-measured with plain `-t cuda` (no node tracing), same models, same
binary, same cards:

| model | nodes a token | node-TRACED launch | NO node tracing |
|---|---|---|---|
| L4 proxy | 147 | 22.4 us | **22.9 us** |
| L44 proxy | 1466 | 204.0 us | **24.4 us** |

Flat across a 10x node swing. Node tracing inflates the call **8.4x** at 45 layers and not at all at 4.

**Consequences, all of them mine to own:**
1. **A graph launch costs ~24 us, not 204-278 us.** Two a token is ~49 us, not 567 us.
2. **The "752 us host boundary, 10.9 % of the token" is not established.** The gap was measured in the same node-traced
   capture, so it is inflated too. There IS a gap at the `mul_mat_vec_q -> k_bin_bcast` boundary -- it appears in every
   capture, 30 of 30 tokens -- but its real size is unknown and must be measured without node tracing.
3. **My argument that it could not be an artifact was wrong.** I compared box 53637080's clean 147 tok/s against the
   trace's own 144.7 and concluded nsys cost 109 us a token. Those were DIFFERENT RUNS. rig-glm separately reported
   **105.2 tok/s under nsys** -- 9.506 ms, about 2.7 ms of overhead a token -- which I dismissed because it disagreed
   with my burst-spacing measurement. The 105.2 was the better evidence and I argued it away.
4. **Node count is NOT a lever**, so fusing the elementwise pool has no launch-cost justification. Its EXCLUSIVE kernel
   time, 0.078 ms, stands as its whole value. The `ab5` e9 null result needs no new explanation.
5. A `GGML_BACKEND_META_PARALLEL_LAUNCH` experiment (launching each device's graph from its own thread) measured
   **worse**: launch median 24.4 -> 61.5 us, total host launch time 47.0 -> 94.0 ms, tg32 93.48 +- 7.99 -> 88.12 +- 9.77.
   The launches do overlap as intended (151 of 525 overlapping against 0 of 525), so the mechanism works and simply does
   not pay -- driver contention costs more than the serialisation it removes. The change is deleted, not committed.

**What in the section above still stands**, because kernel timestamps are device-side and node tracing inflates the host
side and the GAPS, not kernel durations: the EXCLUSIVE-occupancy table, the all-reduce being 100 % exclusive, the decode
window's identification, and the whole 4-bit tiering result including the **+10.93 %** wall measurement, which used no
profiler at all.

**The rule this earns, and it is the one the whole campaign kept rediscovering:** never vary a quantity whose instrument's
overhead is a function of that quantity. Where that is unavoidable, measure the same comparison with the instrument
removed before believing any slope.

### The host boundary, now MEASURED without node tracing (local proxy, Oct 1, 2026, 4:30 AM CT)

The correction above left the gap's real size unknown. It is now measured, from the same untraced capture
(`nsys -t cuda`, no `--cuda-graph-trace`), by joining `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` to its host
`cudaGraphLaunch` on `correlationId` and taking the capture's TAIL, where the last 128 launches are exactly the 64
generated tokens x 2 devices:

| | device 1, L44 proxy, `-sm tensor -d 2048 -r 2` |
|---|---|
| graph executions isolated | 63 (64 tokens, one graph a device a token) |
| graph execution duration, median | **10.266 ms** |
| gap between consecutive executions, median | **444.6 us** (p25 404.7, p75 546.5, n=61) |
| reconstructed token | 10.71 ms = **93.4 tok/s** |
| **measured wall, same arm** | **93.48 tok/s** |

**Exec + gap reconstructs the wall to 0.1 %.** That closure is what the node-traced numbers never had, and it is the
reason to trust this decomposition: nothing is unaccounted for.

**So the host boundary is REAL, at 444.6 us = 4.2 % of the token** -- not the 10.9 % the node-traced capture reported,
and not zero. Node tracing inflated the GAP only 1.24x (551.4 -> 444.6 going the other way), while inflating the
`cudaGraphLaunch` CALL 8.4x. Both can be true: tracing makes the launch call expensive without making the interval
between graph executions much longer, because the interval is dominated by something else.

**And the cause is NOT graph launch.** Two launches at 24.4 us are 48.8 us, **11 % of the 444.6 us gap**; the other
~396 us (3.7 % of the token) is unattributed host time -- sampling, the logits D2H, batch preparation, the meta
backend's own per-step work. That is why the parallel-launch experiment could not have paid even in principle, and
measuring it was the cheaper way to learn that than reasoning about it.

**What this makes of the lever, stated as a band for the target:** 444.6 us against the dev pair's 10.70 ms token.
On 2x PRO 6000 the token is 6.80 ms, and host work does not shrink with faster cards -- if the gap is the same in
absolute terms it is **6.5 % of the target token**, and if it scales with the slower Xeon's host cost (0.19 against
0.14 us a launch, ~1.36x) it is worse. Either way it is now the single largest non-kernel item and **the next
measurement is an untraced capture on target hardware, decomposed exactly this way**, before any attempt to fix it.
The 396 us needs attribution before a lever is proposed: this is the third time this campaign that a boundary was
explained before it was decomposed, and twice the explanation was wrong.

### The retraction was WRONG. The node-count law stands. (Oct 1, 2026, 4:45 AM CT)

**Read this section before either of the two above it.** The correction that called the node-count law an instrument
artifact was itself an error, from a denominator mistake of mine, and the original finding is right.

I compared the **median over every `cudaGraphLaunch` in each capture**, and the two captures are polluted by prefill to
very different degrees -- which is the actual mechanism that made the comparison look flat. The **L44** capture holds
**526** launches of which only **128 are decode**, so its 398 cheap prefill launches dominate its median and hide a
261.5 us decode cost behind 24.4 us. The **L4** capture holds **166**, of which the same 128 are decode, so its median is
barely distorted at all. One median moved 237 us and the other moved 0.2 us, and comparing them reported a flat line.
(Launch counts re-derived independently by the reviewer at 605ea35ca, which also corrected my earlier prose here
that had generalised 526 to both captures.) Isolating the decode tail (the last 128 launches = 64
generated tokens x 2 devices):

| capture | nodes a token | median over ALL launches | **DECODE-only median** | decode mean |
|---|---|---|---|---|
| L4 proxy | 147 | 22.9 us | **22.7 us** | 26.0 us |
| L44 proxy | 1466 | 24.4 us | **261.5 us** | 290.3 us |

**x11.50 over a 10.0x node swing.** The law is real; untraced slope (261.5-22.7)/(1466-147) = **0.181 us a node** on this
box. And node tracing never inflated it: **traced 204.0 us against untraced 261.5 us** -- if anything the traced number
was low. The "8.4x inflation" was the 24.4 us polluted median standing in for a 261.5 us decode cost.

**So, the standing account of the token boundary**, all from the untraced capture whose exec+gap reconstructs the wall
to 0.1 % (10.266 ms + 444.6 us = 93.4 against a measured 93.48 tok/s):

| in the 444.6 us gap | a gap | us a gap | share |
|---|---|---|---|
| `cudaGraphLaunch` | 2.0 | **294.6** | 66 % |
| `cudaMemcpyAsync` | **28.1** | 69.4 | 16 % |
| `cudaStreamSynchronize` | 2.2 | 33.1 | 7 % |
| `cudaEventSynchronize` + `cudaEventRecord` | 4.0 | 7.6 | 2 % |
| no CUDA call at all -- pure host CPU | | 39.9 | 9 % |

91 % attributed to CUDA API calls. The launches' own duration (261.5 us each, 523 us for two) exceeds the gap, so they
extend into the device-busy period as well; 294.6 us of them falls inside it. **Graph launch dominates the boundary, so
node count IS a lever**, and the original 181 us estimate for the elementwise pool is back in force at the untraced
slope: 481 nodes x 0.181 us x 2 launches = **174 us a token**.

**Why parallelising the launches still failed**, which is not a contradiction: the driver appears to serialise graph
submission regardless. Launching from two threads made the overlap real (151 of 525 against 0 of 525) while **doubling**
total host launch time, 47.0 -> 94.0 ms, and the wall went 93.48 -> 88.12. Concurrency does not help a driver-level
lock; **fewer nodes** does. That change stays deleted, for a better reason than before.

**The error class, which is the one this campaign keeps paying for and the fourth distinct costume it has worn:** a
median over a MIXED population reports the dominant subpopulation, not the one under study. Prefill launches outnumber
decode launches 3:1 here, so every statistic over "all launches" is a prefill statistic. Alongside the earlier three --
a bake-off whose denominator was L2-resident, a span that contained the gap it was used to disprove, a check whose list
was its own denominator -- the rule is the same each time: **name the population a number describes before believing
what it says about a different one.**

### The boundary's memcpy component, so the decomposition is closed (local proxy, Oct 1, 2026, 5:15 AM CT)

The 28.1 `cudaMemcpyAsync` calls a boundary, never looked at before, are **29.0 device-side copies moving 416.7 KiB in
34.9 us**:

| direction | bytes each | a gap | us a gap | achieved |
|---|---|---|---|---|
| **D2H** | **309,760** | 1.0 | **22.1** | **14.0 GB/s** |
| H2D | 17,440 / 16,384 / 8,704 | 2.0 each | 1.6 / 1.5 / 2.2 | 22 / 22 / 8 GB/s |
| H2D | 4,608 / 2,312 / 2,048 | 4.0 / 4.0 / 2.0 | 2.4 / 1.3 / 0.7 | 7.6 / 6.9 / 6.2 GB/s |
| H2D | 8 / 4 / 16 | 6.0 / 4.0 / 2.0 | 1.5 / 1.1 / 0.5 | ~0 |

The one D2H is **the logits**: 309,760 bytes is this device's half of the vocab in F32 under TP, and at **14.0 GB/s it
runs at about half of what PCIe Gen4 x16 gives**, which is the signature of an unpinned destination. The other 28 are
batch setup -- token ids, positions, KV metadata, per-layer state -- in copies of 4 to 17 KiB plus eight copies of 4 to
16 BYTES, where the time is entirely per-call overhead.

**So the boundary is closed, and it is a launch problem, not a transfer problem:**

| component of the 444.6 us | us | share |
|---|---|---|
| `cudaGraphLaunch` x2 | **294.6** | **66 %** |
| memcpy, API-side (34.9 us of it device-side) | 69.4 | 16 % |
| `cudaStreamSynchronize` x2.2 | 33.1 | 7 % |
| pure host CPU | 39.9 | 9 % |
| events | 7.6 | 2 % |

Everything outside the launches is **~150 us**, and the realistic recovery inside it is small: pinning the logits
destination buys ~11 us, batching the 28 setup copies into one buys perhaps 20-30 us of call overhead. **Worth doing, not
worth planning around.** The launches are the boundary, and at 0.181 us a node the only two ways at them are **fewer
nodes** a graph or **fewer graphs** a token.

The second of those is the stronger version and it is the same change as `next-levers.md` item 1b: sampling on the device
removes this 310 KiB D2H, the host sample inside the 39.9 us, and the H2D of the chosen token -- and once the host is out
of the loop between tokens, several tokens can share one graph, which removes whole boundaries rather than shrinking one.

### Graph launch cost tracks KV DEPTH, not node count or device time -- so the node lever is UNSIZED (Oct 1, 2026, 5:45 AM CT)

The node-count slope above is confounded, and the confound is not the one I chased. Varying KV depth holds the graph's
node count EXACTLY constant (same model, same ops, only the KV extent differs), and it moves the launch cost more than
changing the model does:

| `-d` | decode launch | graph exec | boundary gap |
|---|---|---|---|
| 0 | **179.9 us** | 9.577 ms | 670.9 us |
| 2048 | 261.5 us | 10.266 ms | 444.6 us |
| 8192 | 272.9 us | 10.381 ms | 461.0 us |
| 24576 | **384.2 us** | 10.517 ms | 654.8 us |

**Launch x2.14 while device execution x1.10, at identical node count.** A least-squares fit of launch against device time
returns a NEGATIVE intercept (-1523 us), which is the arithmetic refusing the model rather than a small misfit.

So all three explanations I have offered for this number are wrong or unproven:
1. "0.14 us a node, linear" (node-traced L4 vs L44) -- confounded: node count, device time and nothing else varied together.
2. "an instrument artifact, the law is flat" -- wrong, from a median over a mixed prefill/decode population.
3. "0.0993 us a node + 11.30 us a ms of exec" -- fails its held-out point by **+27 %**, and this depth sweep refutes the
   device-time term outright.

**What is solid, and it is less than I have been claiming:** launch cost is a large real part of the boundary (294.6 us of
444.6 us at d=2048); it grows with model size at fixed depth (22.7 us at 4 layers against 261.5 us at 45); and it grows
with depth at fixed model and near-fixed device time. **Nothing I have measured licenses a prediction of what removing N
nodes would save**, so the elementwise-pool lever is UNSIZED -- not 174 us, not 96 us, unknown. `next-levers.md` item 3
should be read as "unknown, plausibly tens of us" until the experiment below runs.

**The decisive experiment, which is a mutant and not a model:** add a temporary macro-guarded branch that SKIPS a counted
batch of pool nodes (~100), accepting numerically wrong output, and measure the launch cost against the unmutated binary
at the same depth and model. That measures d(launch)/d(nodes) directly with everything else fixed, which no amount of
cross-model or cross-depth regression can do, because in this engine node count never varies alone. Remove the guard
afterwards and verify removal by diff hash.

**And a free finding worth more than the lever:** if launch cost grows with KV depth, it grows along the serving session,
so a long-context seat pays a boundary that widens as the conversation does -- 179.9 us at depth 0 against 384.2 us at
24.6k, on a graph that never changed shape. Worth confirming on target hardware at serving depths, where it would show up
as latency creep that no kernel measurement explains.

### What the launch cost actually is, settled with a microbenchmark (Oct 1, 2026, 6:30 AM CT)

Five attempts to size this from engine captures failed because in llama.cpp node count, device work and KV depth never
vary independently. A standalone CUDA microbenchmark varies them one at a time
(`local/research/glm-dsa-sparse-2026-09-30/launch-cost/launch-cost.cu`, empty nodes via `cudaGraphAddEmptyNode`, graph
pre-uploaded with `cudaGraphUpload` so first-use cost is excluded, host duration of `cudaGraphLaunch` timed with the
device idle, median of 50 after 10 warmups):

| what varies | range | launch cost |
|---|---|---|
| **node count**, empty nodes, zero device work | 1 -> 3000 | **0.14 us, FLAT** |
| **node count**, kernel nodes, 4 KiB each | 1 -> 1500 | 1.08 -> **1.79 us** (0.0005 us a node) |
| **allocation size**, 400 nodes, fixed 4 KiB of work each | 1 MiB -> 192 MiB | 1.34 -> **1.40 us, FLAT** |
| **allocation count**, 400 nodes, fixed work | 1 -> 64 allocations (16 MiB -> 1 GiB) | 1.38 -> **1.36 us, FLAT** |
| **work a kernel** (grid scaled with the buffer) | 4 KiB -> 192 MiB a kernel | 1.34 -> **22.37 us** |

**Node count does not drive launch cost. Neither does allocation size or count. Only the WORK the graph enqueues does.**
A synthetic 1500-node graph submits in 1.79 us while the engine's 1466-node graph takes 246-469 us -- a **143x gap at the
same node count** -- so the cost was never the count, and **the elementwise-pool fusion lever is dead for a third and
final time, now with a mechanism instead of an estimate.** `next-levers.md` item 3's launch-cost justification is void;
its 0.078 ms of exclusive kernel time is all it was ever worth.

**And the boundary is the submission itself.** A single-card run does not collapse it, which rules out TP
synchronisation:

| arm | launch | graph exec | boundary gap |
|---|---|---|---|
| tiered, TP 2 cards | 246.0 us | 8.891 ms | 433.6 us |
| tiered, SINGLE card | **469.4 us** | 12.539 ms | **496.9 us** |

**The gap equals the launch duration** (496.9 against 469.4 single-card; 433.6 against two serialised 246.0 us launches).
The device is idle at the token boundary **because the host is inside `cudaGraphLaunch` enqueuing the next token's work**,
and submission cost scales with the work enqueued. That one relation explains every earlier observation without any of
the mechanisms I invented: L4 at 22.7 us against L44 at 261.5 (10x the work), single-card above TP (1.41x the work a
card), deeper KV above shallower (more attention blocks), and the tiered model 5.9 % below Q8_0 (less work).

**What it means for the levers, and it is good news for the one that already works:**
- **Anything that reduces device work reduces the boundary too.** The byte lever is worth more than its device-side
  number: -24 % of bytes also bought -15.5 us a launch, **-31 us a token** on top of the kernel-time saving. Device-side
  optimisations pay twice here.
- **The structural fix is to overlap submission with execution**, not to shrink the graph: submit token N+1 while token N
  runs. The host cannot do that today because it waits for the logits, samples, and only then builds the next batch --
  so this is `next-levers.md` item 1b (on-device sampling) arriving by a second independent route, now with two
  mechanisms behind it rather than one.
- Fusing nodes, parallelising launches across threads, and trimming graph size are all dead, each now for a measured
  reason.

**The methodological cost, recorded because it was high:** this one number got six accounts in one night -- 0.14 us a
node, an instrument artifact, 0.0993 us a node plus a device-time term, depth-driven, unsized, and finally
work-driven -- and five were wrong. Every wrong one came from regressing across engine configurations where the variables
move together. **The 90-line microbenchmark that settled it should have been the first move, not the sixth**: when a
quantity cannot be varied alone in the real system, no amount of careful measurement of the real system will separate it.

### n_copies > 1 under a tensor split does NOT shrink the boundary — null result (Oct 1, 2026, 6:25 AM CT)

A tested fix for the 444.6 us boundary, measured and rejected. Recorded because the reasoning behind it was sound, the
experiment was cheap, and the answer was no.

**The premise, read from the decode path rather than assumed.** `llama_context::decode` ends with `//synchronize();`
commented out and `graph_compute` calls `ggml_backend_sched_graph_compute_async`, so decode returns without waiting.
Submission costs 246-507 us against 8.9-12.5 ms of execution, so the host should run far ahead and the boundary should not
exist. It does, and **the gap equals the launch duration** — the host is released exactly when the device goes idle. That
left the synchronous input copy in `set_inputs`: it cannot write the buffer the running graph is still reading, and
`ggml-backend.cpp:1763-1766` does wait on `sched->events[backend][cur_copy]` before staging an input. With n_copies = 1
that event is the immediately preceding graph; with n_copies > 1 it is several tokens back and the wait disappears.

**The switch.** `src/llama-context.cpp:559` gates n_copies > 1 on `split_mode == LAYER && n_devices() > 1`. A tensor split
presents ONE meta device and fails both, yet the meta backend reports `async` and `events` (`ggml-backend-meta.cpp:130-144`)
and implements `set_tensor_async` (`:2724`), which is the path the scheduler actually takes. `LLAMA_PIPELINE_PARALLEL=1`
overrode the gate; confirmed live by `llama_context: pipeline parallelism enabled` at `-v`, absent with it off.

**The result: no effect.** First pass, 4 arms interleaved 0-1-1-0, read **-34.4 %** — and failed its own pre-declared
20 % spread gate at 60 % and 30 %, so it was discarded rather than reported. Ten interleaved PAIRS, each arm about a
minute, differences taken within a pair so drift cancels:

| statistic | value |
|---|---|
| mean per-pair difference | **-77.8 us** |
| sd / stderr | 213.8 / 67.6 |
| 95 % interval | **-210.3 to +54.7 us** |
| pairs favouring the switch | **5 of 10** |

**The interval includes zero and half the pairs went the wrong way, so the -34.4 % was arm ordering, not an effect.**
The switch is deleted; an env var with no measurable effect is dead code. Its correctness gate (greedy `llama-cli`,
byte-identical output required, since `llama-bench` feeds `std::rand() % n_vocab` and never reads a logit so it cannot
detect corruption) was cancelled unrun when the performance result came back null.

**What this leaves.** The boundary is real and its cause is submission — that much survives, from the gap equalling the
launch duration and from the microbenchmark showing submission cost tracks **enqueued work**. What is now also known is
that the scheduler's input-copy rotation is NOT the serialiser, so the remaining candidate is that the same `graphExec`
cannot be enqueued concurrently with its own prior execution, which no amount of input buffering addresses. That points
back to the only two levers the microbenchmark left standing: **less enqueued work** (which the byte lever already
delivers, worth -31 us a token of submission on top of its kernel saving) and **fewer graphs a token**, which needs the
device-resident sampled token in `local/research/glm-dsa-sparse-2026-09-30/submit-ahead.md`.

### SUPERSEDES every launch-cost attribution above: it is ~15 us a token, not 294.6 (Oct 1, 2026, 8:00 AM CT)

**Read this section instead of the launch-cost shares in any section above.** Every one of them came from nsys, and nsys
cannot measure this call.

**The instrument, because no profiler could do it.** `GGML_CUDA_TIME_LAUNCH=1` (engine `c95678d9b`) times
`cudaGraphLaunch` with `clock_gettime` inside the engine and prints a running mean a device. It exists because the
standalone microbenchmark showed nsys inflating the call **39x**: a 400-node graph that its own process times at
**1.3 us** reads **50.6 us by that same clock** once nsys is attached, and nsys reports 51.1 us. The instrumentation
CONSUMES that host time; it does not merely misreport it.

**The measurement, no profiler attached, tensor split on two cards, `-d 2048`:**

| | mean a launch |
|---|---|
| device 0 | **8.0 - 13.6 us** |
| device 1 | **5.9 - 9.1 us** |
| nsys's figure for the same launches | 246 - 294 us |

**So two launches cost about 15 us a token, 3 % of the boundary -- not the 66 % a trace reported.**

**The boundary itself is still real, and this is why it survives.** Device-side graph-execution timestamps are reliable
(they are device records, not host API records): execution is **10.266 ms**, and the **untraced** wall is 93.48 tok/s =
10.70 ms, so the gap is **~430 us** with no profiler in the picture. What changes is entirely its composition: **~415 us
of it is engine host work, not CUDA.** The 28.1 `cudaMemcpyAsync`, the `cudaStreamSynchronize` and the "pure host CPU"
rows in the table above are all host-API measurements from the same trace and are inflated the same way, so that whole
decomposition is void except for its total.

**This is why all three fixes measured null, and the nulls were correct.** Fusing nodes to shrink the graph, launching the
two devices' graphs from separate threads, and the scheduler's extra input copies each attacked the launch -- 3 % of the
boundary. Each one measured no effect because there was no effect to find. The experiments were right; the target was
wrong, and it was wrong because a profiler said so.

**What is now the open question, and it is a better one:** what consumes the other ~415 us? It is host work inside
`llama_context::decode` outside the launch -- candidates are `set_inputs`, the graph build-or-reuse decision, the
scheduler's split bookkeeping, and the batch preparation. **The technique that settled the launch question answers this
one too**: time those phases with `clock_gettime` inside the engine, no profiler. That is the next measurement, and no
further lever should be proposed for this boundary until it runs.

**The methodological rule, which cost this campaign most of a night:** on this machine, at this scale, **nsys may be used
for DEVICE timestamps and must not be used for HOST attribution.** Kernel and graph-execution records are trustworthy;
every CUPTI RUNTIME duration is inflated roughly 30-40x and will invent a bottleneck that does not exist. The 90-line
microbenchmark and the 20-line in-engine timer that found this should both have come before the first trace-based lever.

### THE TOKEN BOUNDARY IS RETIRED: the host runs 7.6x ahead of the device (Oct 1, 2026, 8:30 AM CT)

**This retires the "444.6 us host boundary" and everything built on it, including sections of this file above.** It was
mostly the profiler's own cost.

`LLAMA_TIME_DECODE=1` (engine `acfa0196f`) times decode's host phases with `clock_gettime`, no profiler attached. Decode
only (`-p 0 -n 128 -d 0`), tensor split on two cards, 99.25 tok/s = **10.08 ms a token**:

| phase | mean a call | calls | share of decode |
|---|---|---|---|
| `build_graph` | 812.7 us | 2 | 1.0 % |
| `alloc_graph` | 18,445.8 us | 2 | 21.9 % |
| `set_inputs` | **2.7 us** | 128 | 0.2 % |
| `graph_compute` | 944.8 us | 128 | 71.6 % |
| **`decode()` TOTAL** | **1319.1 us** | 128 | 100 % |
| unattributed | 70.6 us | | 5.4 % |

**`decode()` returns in 1.32 ms while a token takes 10.08 ms.** Over 128 tokens the host spends **169 ms against the
device's 1290 ms** — idle **87 %** of the time. If the host were blocking on the device, decode would take about 10 ms,
not 1.32. **The host runs 7.6x ahead and is not the bottleneck.** `cudaGraphLaunch` in the same run: 18.0 us on device 0,
10.9 us on device 1.

**So the whole host-boundary thesis was an artifact**, and the chain of reasoning that produced it failed at its first
link: a trace said the device idled 444.6 us between graph executions with 66 % of it inside `cudaGraphLaunch`. The launch
is really ~15 us, and the idle is largely nsys's own host cost — nsys slows the host enough that it can no longer stay
ahead of the device, manufacturing exactly the gap it then reports. **The measurement created the phenomenon.**

**It also explains, correctly, every null this campaign recorded against that boundary:** fusing the elementwise pool to
shrink the graph, launching the two devices' graphs from separate threads, and the scheduler's extra input copies
(n_copies > 1) each measured no effect. All three were sound experiments aimed at recovering host time **that was never
being lost.** The nulls were right and the hypothesis was wrong, which is the opposite of how I read them at the time.

**`set_inputs` at 2.7 us also kills the specific mechanism** I had reasoned my way to from the code — that a synchronous
input copy blocked the host against the running graph. It costs 2.7 microseconds.

**What this leaves standing, and it is the thing that was measured without a profiler all along:** the decode token is
**device-bound**, so only device-side work pays. That is the byte lever — **-25.1 % of per-token reads, +10.93 % measured**
— and it is the one result of this campaign that never depended on a trace. `next-levers.md` items 1b, 3 and the
`submit-ahead.md` design are all withdrawn as boundary levers; if on-device sampling is still wanted it has to be
justified by the speculation loop's own measurement, not by a per-token host gap that does not exist.

**The rule, stated once and for the whole machine:** nsys is valid for DEVICE timestamps and invalid for HOST attribution
here. Worse than inflating its own numbers 30-40x, it perturbs the system enough to create host-side bottlenecks that are
absent without it. **Any claim that the host is the bottleneck must come from in-process timing**, which is 20 lines
(`LLAMA_TIME_DECODE`, `GGML_CUDA_TIME_LAUNCH`) and should precede the first trace, not follow the sixth.

### How close decode is to the ceiling, measured per kernel against DRAM peak (local, Oct 1, 2026, 9:00 AM CT)

The answer to the campaign's question, from the one instrument that is valid for it: ncu measures DEVICE counters, so
unlike nsys's host attribution it can be trusted here. `--replay-mode application` (the decode holds CUDA graphs),
`--cache-control all` (without it a shape that fits the 67 MB L2 reports L2 bandwidth as if it were DRAM -- this campaign
already published a 2816 GB/s "achievement" against a 960 GB/s peak that way), and `--kernel-name regex:` to select the
DECODE kernels, after a first attempt profiled prefill by launch index and mislabelled compute-bound tensor-core matmuls
as "below roofline".

Card 0, one card, `-p 0 -n 24 -d 0`, 600 launches:

| decode kernel | launches | **% of DRAM peak** | total us | share of kernel time |
|---|---|---|---|---|
| `mul_mat_vec_q` | 242 | **85.7 %** | 10,421 | 64 % |
| `mmvq_moe` | 92 | **82.3 %** | 4,500 | 28 % |
| `mul_mat_vec_f_vec` (BF16 indexer) | 218 | 41.9 % | 827 | 5.1 % |
| `gated_delta_net_cuda` | 36 | 55.8 % | 298 | 1.8 % |
| `flash_attn_ext_f16` | 12 | 4.7 % | 123 | 0.8 % |

> **PARTIALLY CORRECTED.** The **`% of DRAM peak` column is valid** — it is a per-kernel hardware counter. The **`total us`
> and `share of kernel time` columns are NOT**: they are ncu durations under `--replay-mode application`, which inflates
> them, and the 600-launch window is about ONE token rather than the 24 requested. **So "92 % of decode kernel time" is an
> artifact.** The same two matmuls hold **2.192 + 1.041 = 3.233 ms of EXCLUSIVE occupancy in a 6.88 ms span, about 47 %**
> (nsys, node-traced, from the table further up). That corrected figure is corroborated independently: the byte A/B puts
> **f = 0.545**, i.e. 54.5 % of token time scaling with bytes, which agrees with ~47 % and not with 92 %. **The surviving
> conclusion rests on f and on the per-kernel percentages, never on the inflated shares** — and the "2.6 % from lifting the
> BF16 indexer" below is likewise wrong: that kernel's own exclusive time is 0.110 ms, so **1.6 % is its ceiling**, and it
> is already ~91 % PDL-overlapped, which is why it was dropped.

**The two matmuls are 92 % of decode kernel time and run at 82-86 % of DRAM peak. Decode is AT the bandwidth roofline.**

**So the ceiling is bytes, and that is now a measurement rather than an argument.** Everything with real headroom totals
**1,248 us, 7.7 %** of kernel time: the BF16 indexer at 41.9 % (its scalar-FMA path, `next-levers.md` item 6 -- lifting it
to the matmuls' 86 % would save ~419 us, **2.6 %**), the KDA kernel at 55.8 %, and flash-attention at 4.7 % of peak but
only 123 us, which is what a correctly-sized attention looks like at depth 0 with nothing to read. Perfect optimisation of
all three yields about **3 %**, once.

**This closes the loop on the whole campaign.** Every scheduling lever tried -- fusing the elementwise pool, launching the
two devices' graphs from separate threads, the scheduler's extra input copies, branch concurrency -- is dead, and this
table is why: 92 % of the time is kernels already streaming at 82-86 % of what the memory system can deliver. There is no
schedule that makes DRAM faster. **The only lever that moves a bandwidth-bound decode is fewer bytes**, which is exactly
the one that measured **+10.93 %** at **-25.1 %** of per-token reads, and the only one here that never needed a profiler.

**What that makes the remaining work, in order:**
1. **Validate the byte lever's QUALITY on real weights** -- the only thing between a measured +10.93 % and shipping it.
   It needs a rental, which needs the operator. The instrument is built and verified (`LLAMA_KLD_FIRST`, position-binned
   dKLD, engine `a61c58f5c`).
2. **Then cut more bytes**, since the roofline says that is the whole game: the Q6_K tier and the Q8_0 MLA tensors
   (`attn_k_b`, `attn_v_b`, `attn_kv_a_mqa`) are the next candidates, each gated on the same quality measurement.
3. The BF16 indexer's WMMA path, worth ~2.6 % on its own.
4. Prefill, which this census says nothing about and where `mul_mat_q` at 15-19 % of DRAM peak is expected and correct
   (compute-bound on tensor cores) -- prefill needs an FLOP roofline, not this one.

### The byte lever re-measured as 10 pairs: +15.94 %, not +10.93 % (local, Oct 1, 2026, 11:31 AM CT)

The campaign's headline rested on **one round**: three were run and two failed a pre-declared 4 % within-arm spread gate.
That is thinner evidence than was used to kill several levers here, so it was re-measured with the paired design that
settled the pipeline-parallel question — baseline and tiered back to back, ten times, no profiler, difference taken within
each pair so host-load drift cancels.

| | |
|---|---|
| mean per-pair gain | **+15.94 %** |
| sd / stderr | 1.70 / 0.54 |
| **95 % interval** | **+14.88 % to +17.00 %** |
| pairs | 10, at loadavg 7-17 |
| previous single-round figure | +10.93 % — **outside this interval** |

**The lever is better than recorded, and the old number was pessimistic rather than merely uncertain.** The paired design
is why the spread collapsed: sd 1.70 here against +-4 to +-8 % within the unpaired rounds, on a noisier box. The one
round that survived the original gate was simply unlucky.

**What the gain means against the ideal, which is the more useful framing.** Per-token reads go 11.860 -> 8.881 GB (the
trunk only: the graph stops at `n_layer()`, and the NextN block's 0.280 / 0.199 GB is not read; corrected Oct 1 from
12.139 -> 9.079, which counted it), so a perfectly bandwidth-bound token would gain 11.860/8.881 = **+33.5 %**. The measured
+15.94 % is **48 % of that ideal**,
which is what it should be: the reduction touches the non-expert tensors (8.4 GB of the 12.1), the experts' 3.4 GB are
untouched at IQ3_XXS, and the all-reduce (fully exclusive), attention, KDA and the BF16 indexer are not byte-proportional
at all. **About half the token moves with bytes, so a -25.1 % byte cut buys about +16 %.** That ratio is the thing to
carry to the target rather than the +7.3 % predicted from `mul_mat_vec_q`'s exclusive occupancy alone, which understated
it for the same reason the single round did.

**Still UNMEASURED: quality.** Every number above is a random-weight proxy. The speed claim is now solid and the quality
claim does not exist, and those are independent. `e28-bytes/candidates.md` holds five recipes in priority order with their
override files for the funded box; this result raises the stakes on A's quality run rather than changing its design.

### The byte-proportional fraction f, and a DECLARED band for the target (local, Oct 1, 2026, 12:06 PM CT)

Declared BEFORE any target run, so the box can falsify it rather than confirm it.

A byte cut only buys time on the part of the token that scales with bytes. Call that fraction **f**, and the rest
(all-reduce, attention, KDA, the BF16 indexer, launch and sync) does not move when weights shrink:

    speedup = 1 / (1 - (1 - r) * f),   r = new bytes / old bytes = 8.881 / 11.860 = 0.749

Inverting the measured +15.94 % gives **f = 0.545** (0.547 with the trunk-only bytes, inside the interval, so 0.545 is
kept below), and the 95 % interval on the speedup gives **f = 0.514 to 0.576**.
So **about 55 % of this token scales with bytes** on the dev pair. That one number explains the whole session: it is why
-25.1 % of reads bought +15.94 % rather than the ideal +33.5 %, and it is why every scheduling lever measured null -- they
all attacked the other 45 %, which the roofline census then showed is mostly kernels already at 82-86 % of DRAM peak.

**Prediction for 2x RTX PRO 6000, with its mechanism: +16 % to +21 %, i.e. MORE than the dev pair, not less.**
The reasoning is that f should RISE on the target, because the largest non-byte-proportional item here is the all-reduce
-- 0.309 ms of a 6.88 ms span, **100 % exclusive**, and serialized over PCIe with `cudaDeviceCanAccessPeer` false both
ways. The target has P2P. Removing non-byte-proportional time raises f, and f is the only term that matters:

| f on target | predicted gain |
|---|---|
| 0.545 (unchanged from here) | +15.9 % |
| 0.60 | +17.8 % |
| 0.65 | +19.6 % |
| 0.70 | +21.4 % |

**What falsifies this:** a target gain BELOW +15.9 % means f fell, and the only honest reading would be that the target's
faster memory (~1.75 TB/s a card against 960/896 GB/s here) moves the bandwidth-bound part so much that it stops
dominating -- which would mean the target is NOT at its roofline and the whole "bytes are the only lever" conclusion is
local to this pair. That is the single measurement that could overturn this session's main finding, so it is worth taking
first on whatever box comes up. It needs no real weights and no quality run: the proxy answers it.

**Note on the earlier +7.3 % prediction: it was wrong and this supersedes it.** It came from `mul_mat_vec_q`'s exclusive
occupancy alone (2.192 ms x -23 %), which counts only the time that kernel holds the device by itself and so ignores every
byte-proportional millisecond it spends overlapped with other work. Same error as the single-round headline: both
understated the lever by looking at too narrow a slice.

### The recipe file did not describe the model it was named for (local, Oct 1, 2026, 12:34 PM CT)

`e28-bytes/tiers.txt` was handed to the L5 box as "recipe A, the +15.94 % model". It was not. It omitted
**`output.weight`**, the logits head, which falls through to the base type without a rule. The measured file has it at
**Q6_K, 520 MB**; the recipe as written produces **Q8_0, 674 MB**, a file **1.57 % larger**. The type had been set by a
`--output-tensor-type q6_k` flag on a command line that was never recorded anywhere.

**With `^output\.weight=q6_k` appended, the pipeline reproduces the measured file at 9762864448 bytes — diff 0.000 %.**

**The consequence was not 1.57 % of bytes, it was the quality run testing the wrong thing.** The logits head is the most
quality-sensitive tensor in the model: it decides token selection directly. The recipe silently described a *more
conservative* model than the one whose speed is being shipped, so a dKLD on it would have come back cleaner than reality
while never touching the tensor most likely to hurt. Byte accounting and quality accounting were pointed at different
files.

**Two verified traps in the one line that fixes it:**
1. **The `^` anchor is load-bearing.** `llama-quantize` matches with `std::regex_search`, so an unanchored `output\.weight`
   also matches `attn_output.weight` as a substring, and first-match-wins would then hand every attention output
   projection the logits head's type, decided by line order. Checked against `attn_output.weight`, `output.weight` and
   `blk.3.attn_output.weight`.
2. **`token_embd.weight` stays Q8_0** and must not be "made consistent" with the head: it is a row gather, not a
   per-token read, so quantizing it costs quality and buys no bandwidth. It is also why every byte figure in this campaign
   excludes it.

**How it was found, and the generalisation.** The candidate harness was built to re-quantize recipe A FIRST and abort
unless it matched the existing file within 0.5 % — a pipeline that cannot reproduce a known answer cannot be trusted on
unknown ones. That gate fired twice: once on the missing rule (+1.57 %), and once on my first fix, which guessed
`iq3_xxs` and came out **2.84 % too small**. The guess came from a type-id table written from memory in which 23 was
labelled `iq3_xxs`; 23 is **IQ4_XS** and 18 is IQ3_XXS, per `gguf-py`. The same probe also misreported the head's type
because it matched tensor names with `endswith`, and `attn_output.weight`.endswith(`output.weight`) is true — the identical
substring trap, in the measuring code rather than the measured config.

**The rule this earns: a recipe that cannot rebuild its own artifact byte-for-byte is not a recipe, it is a note.** Three
hand-written tables (ggml type ids, block layouts, tensor-role patterns) were wrong in this one investigation, and all
three were caught by deriving from the source instead: GGUF tensor offsets for bytes, `gguf-py` for type names, and the
regex engine's own behaviour for matching.

**Root cause, and the structural fix.** `local/*` is `.gitignore`d, so every recipe in this campaign lived outside version
control — which is exactly how the `--output-tensor-type q6_k` flag was lost: nothing was tracking the command that built
the artifact. Documentation would not have stopped this; the recipe being untracked is what made the loss possible. The
recipes now live at **`heads/glm-5.3-flash/recipes/`**, version-controlled beside the head that serves the model, with the
byte-exact reproduction requirement written into their README as the acceptance check.

**Correction to the alarm above, from rig-glm with evidence: the L5 box was never quantizing the un-fixed model.** Its tier
step passes `--output-tensor-type q6_k` on the command line, and its dry-run log reads
`[1/1391] output.weight ... type = q8_0, converting to q6_K` (642.81 -> 496.29 MiB). The recipe FILE was incomplete, which
is worth fixing and is now fixed; the box's behaviour was correct throughout. The warning was right about the artifact and
wrong about the consequence.

**And the real pack holds a trap the proxy structurally cannot show.** rig-glm read the served shards' GGUF headers: the
experts are iq3_xxs in 42 layers and **Q8_0 in `blk.45`, the NextN/MTP block** (42 + 1 for each of down/gate/up). A bare
`ffn_*_exps=iq3_xxs` asks for a Q8_0 -> IQ3_XXS requant there, which `llama-quantize` refuses without an imatrix — and
**with** an imatrix silently requantizes the draft head's experts instead, degrading speculative decoding while bytes and
perplexity still look right. All five recipes now pin `^blk\.45\.ffn_(down|gate|up)_exps\.weight=q8_0` as their first three
lines, ordering verified so `blk.3` and `blk.44` still take the recipe's expert type (on the proxy `blk.44` is its NextN
block, which `^blk\.45` does not match).

**The limit this exposes in the campaign's main instrument: the proxy's NextN block is not the pack's.** The proxy has 44
trunk layers and `nextn_predict_layers = 1`, so its NextN/MTP block is `blk.44`, with IQ3_XXS routed experts where the pack's
`blk.45` holds Q8_0 (`make-proxy.py` writes it so). No proxy file can hit the Q8_0 requant refusal, so this failure class is
invisible to every measurement in this file. MTP speculation itself does run on the proxy. The proxy is not a
subset-with-random-weights of the served pack; its tensor set differs in kind. **Reproduction checks belong on the proxy,
refusal and heterogeneity checks belong on the pack**, and no proxy result should be reported as validating a recipe
against the pack again.

### The byte lever on the cards: its quality, and +5 % rather than +16 % (L5, vast 53731187, Oct 1, 2026, 12:32-1:17 PM CT)

Two RTX PRO 6000 Blackwell Max-Q (300 W, driver 610.43.02) on a Xeon Platinum 8568Y+ in Czechia, P2P both ways
(`cudaDeviceCanAccessPeer` 1), rented from the template under a $9 cap: $4.046/h, created at 12:32:17, serving at
12:40:28, destroyed at 1:17:24 after 0.75 h, $8.13 of credit in all. The driver and its results are
`local/research/glm53-l5-2026-10-01/box5/` (the results `out5-53731187.tgz`; the driver, retired from use under the CLI
rule, archived as `driver-53731187.tgz`: `rent5.sh`, `box-run5.sh`, `analyze5.py`). The recipes
are `recipes/tier-{A,C,B}.txt` at `ecb52ac`, each built from the served pack by `llama-quantize` at engine `2cde3194b`
(`--allow-requantize --output-tensor-type q6_k`, base Q8_0) in 1.8-2.4 min on the box's CPUs: from the served 128,074.65
MiB, A 124,992.19 (-3,082), C 124,939.87, B 124,222.12.

KLD against the served pack: `llama-perplexity` at `2cde3194b` on the image's CUDA libraries, `-sm tensor`, wikitext-2's
test text, 4 sequences of 16,384 tokens scored from position 0 (`LLAMA_KLD_FIRST=0`). The served pack's PPL is 3.1272 ±
0.0233, the same to the last digit on the box before. A bin's cell is its mean dKLD / its same top; the bins hold 2,048,
14,336 and 49,148 tokens.

| | mean KLD | same top | PPL ratio | 0-512 | 512-4k | 4k+ |
|---|---|---|---|---|---|---|
| served, itself | 0.000000 | 99.998 % | 1.00015 | 0.000000 / 100.000 % | 0.000000 / 100.000 % | 0.000000 / 99.998 % |
| A | 0.0326 ± 0.0003 | 93.54 ± 0.10 % | 1.0122 ± 0.0012 | 0.0822 / 89.31 % | 0.0357 / 93.03 % | 0.0296 / 93.86 % |
| C (A, MLA latents Q6_K) | 0.0339 ± 0.0003 | 93.32 ± 0.10 % | 1.0156 ± 0.0012 | 0.0843 / 89.06 % | 0.0370 / 92.71 % | 0.0309 / 93.67 % |
| B (A, Tier B IQ4_XS) | 0.0434 ± 0.0004 | 92.52 ± 0.10 % | 1.0222 ± 0.0013 | 0.1125 / 87.40 % | 0.0471 / 91.92 % | 0.0395 / 92.90 % |

- No recipe compounds: each one's divergence falls with position and its same top rises, the largest error in the first
  512 tokens. B's KDA-state tensors at IQ4_XS add a constant third over A in every bin (1.37, 1.32, 1.33x), not a rising
  one.
- Decode, tg64 at 0 and 32,768 cached tokens, each recipe against the served pack in back-to-back pairs, the order
  alternating, three reps a run. The first rep of every run reads ~20 % slow (served: 131.9, then 163.2 / 162.8), so a
  run's rate is its second and third reps; ± is the standard error over pairs, and the 12 served runs read 163.0-163.7
  and 137.7-138.3.

| | no cached tokens | 32,768 cached | pairs |
|---|---|---|---|
| A | +6.33 ± 0.10 % (163.4 -> 173.7 tok/s) | +5.16 ± 0.10 % (138.1 -> 145.2) | 3 |
| C | +7.61 ± 0.08 % | +5.31 ± 0.04 % | 3 |
| B | +11.2 ± 4.7 % (its two runs settled at 173.6 and 189.8) | +11.11 ± 0.54 % | 2 (the third past TIME_MAX) |

- **The declared band is falsified.** It put A at +16 % to +21 % on the target and named a gain below +15.9 % as its
  falsifier. A measured +6.3 % and +5.2 %: f = 0.24 with no cached tokens and 0.20 at 32,768 (r = 0.748), against 0.545
  on the dev pair, on cards that have the P2P the band's mechanism counted on. The served token moves 11.9 GB in 6.1 ms,
  1.93 TB/s over two cards of 1.79 TB/s each (54 % of DRAM peak), and about a fifth of it moves with the bytes A removes.
  Whether the rest is non-byte work or IQ4_XS and Q6_K costing more per byte than Q8_0 on these 300 W cards, a per-kernel
  trace of the two packs on the cards would separate; the 500 W cards were not measured.
- On these cards A buys +5-6 % decode for a 6.5 % change in the top token and +1.2 % perplexity.
- The first L5 box (vast 53728146, 12:04-12:17 PM CT, $6.18) measured only the base and the floor, the same as above. Its
  tier step exited 127 on a `libgomp.so.1` the image does not carry (the tools link OpenMP; the step now loads the engine
  dir's), then was refused on `blk.45`'s experts (above). The box was destroyed when its driver, rewritten in place while
  it ran, read a syntax error and its exit trap ran `vast down`. Both drivers are now wrapped whole in `{ ... ; exit; }`.

### Prefill's ceiling is a different ceiling: the recurrent KDA kernel burns 25 % of it with ZERO tensor cores (local, Oct 1, 2026, 1:28 PM CT)

The decode census above measured against DRAM, which is the right roofline for M=1 and the wrong one for prefill. This is
prefill measured against BOTH rooflines plus SM issue, card 0, `-p 4096 -n 0`, `--replay-mode application
--cache-control all`, kernel-filtered:

| prefill kernel | DRAM | **tensor** | FMA | SM | share of kernel time |
|---|---|---|---|---|---|
| `mul_mat_q<8,128,0>` | 19.1 % | 36.0 % | 20.7 % | 49.7 % | 33.9 % |
| `mul_mat_q<18,128,0>` | 15.5 % | 34.7 % | 21.2 % | 45.2 % | 32.7 % |
| **`gated_delta_net_cuda<128,1,…>`** | 10.1 % | **0.0 %** | 22.7 % | **86.2 %** | **25.1 %** |
| `flash_attn_ext_f16<512,512,…>` | 26.6 % | 45.1 % | 6.0 % | 44.6 % | 3.8 % |
| `mul_mat_q_stream_k_fixup` | 57.3 % | 0.0 % | 1.3 % | 5.7 % | 4.5 % |

**A quarter of prefill runs on a kernel that does not touch a tensor core, while saturating the SM's scalar issue at
86.2 %.** It is not DRAM-bound (10.1 %) and not FMA-bound (22.7 %): the SM is simply full of the sequential recurrence's
loads, shuffles and scalar math. The metric is self-validating -- flash-attention reads 45.1 % tensor and the matmuls
34-36 % on the same run, so the recurrence's 0.0 % is a real property and not a missing counter.

**The cause is one condition, and the mechanism behind it is real.** `gated_delta_net.cu:403` gates the chunked
tensor-core path on `!kda`, where `kda = src_g->ne[0] == S_v` -- a per-channel gate rather than a scalar one. The chunked
formulation's math assumes a scalar gate; KDA's is a vector, which in the chunked form becomes a diagonal gate matrix. So
the exclusion is not an oversight to delete, it is a missing derivation. **GLM-5.3-flash is 33 KDA blocks of 45**, so
roughly three quarters of the model's gated-delta layers take the recurrent path through every prefill.

**This cannot be A/B'd with the existing switch.** `GGML_CUDA_GDN_CHUNKED=0/1` chooses recurrent vs chunked, but every
gated-delta block in this model is KDA and therefore excluded either way, so the flag is a no-op here. The lever has to be
sized by measurement, which is what the table does, and then built.

**Sizing it: 25.1 % of prefill kernel time, in a kernel with 0 % tensor utilization, next to matmuls getting 34-36 %.** A
chunked tensor-core KDA that reached even half the matmuls' tensor utilization would cut that 25 % substantially; at a
plausible 2-4x on the kernel, prefill improves by **12-19 %**. For scale, rig-glm's `-ub 1024` change measured **-22.9 %**
on a 27,642-token prefill, so this is the same order of win and it compounds with that one rather than competing.

**Why this matters more than the remaining decode levers.** Decode is at 82-86 % of DRAM peak and has nothing left but
bytes. Prefill has a quarter of its time in a kernel at zero percent of the hardware's main compute resource. **The
unexplored half of this model's serving cost is the half with the headroom**, and `next-levers.md` item 4 was right to name
it -- it just had no number on it until now.

**Honest limits of this measurement.** One card, no tensor split, random-weight L44-E8 proxy, `-p 4096`. The shares come
from ncu durations, which are inflated by replay and cache control, so they are shares and not milliseconds; the
`%`-of-peak figures are the trustworthy part. The served pack at `-c 524288` with TP across two cards will have different
shares, and a plain decode never runs the NextN block (`blk.44` on the proxy), so nothing here speaks to the MTP block.

### FALSIFIED: the +16-21 % band, and with it "bytes are the only lever" (Oct 1, 2026, 1:52 PM CT)

rig-glm measured recipe A on 2x RTX PRO 6000 Max-Q with P2P both ways: **+6.33 ± 0.10 % at d0 and +5.16 ± 0.10 % at
d32k**, against my declared **+16 % to +21 %**. The band is falsified, and it is falsified in the direction and by the
mechanism I pre-committed to naming:

> *"a target gain BELOW +15.9 % means f fell ... which would mean the target is NOT at its roofline and the whole 'bytes
> are the only lever' conclusion is local to this pair."*

**That is exactly what happened. The target token runs at 1.93 TB/s over the two cards, 54 % of DRAM peak** — against
82-86 % on the dev pair. **f = 0.24 at d0 and 0.20 at d32k, against 0.545 here.** So:

| | dev pair (960/896 GB/s, no P2P) | target (1.93 TB/s, P2P) |
|---|---|---|
| decode vs DRAM peak | **82-86 %** (at the roofline) | **54 %** (not at the roofline) |
| f | 0.545 | **0.24 / 0.20** |
| recipe A | **+15.94 %** | **+6.33 % / +5.16 %** |

**The reasoning error, stated plainly: I got the sign of f wrong, and it was elementary.** I argued f would RISE on the
target because P2P removes the PCIe all-reduce, which is 100 % exclusive here. P2P does remove it. But faster memory
shrinks the byte-proportional TIME while every fixed cost — launch, scalar work, attention compute, sync, dequantization —
stays where it is, so their share rises and f falls. I reasoned about one non-byte term and ignored what happens to the
rest of the denominator. The all-reduce was the term I had measured, so it was the term I thought with.

**What this retracts.** "Decode is at the roofline, so bytes is the only lever" was a property of **this pair of consumer
cards**, not of this model. On the serving hardware decode has **45 % of its time outside the byte-proportional part and
sits at 55 % of peak**, which means there IS non-byte headroom there — the opposite of what I concluded. Every null I
recorded for a scheduling lever was measured on the dev pair and says nothing about the target; those levers deserve
re-testing on the cards before anyone treats them as dead. The dev-pair numbers themselves stand — they were measured and
they remain true of that hardware — but the generalisation drawn from them does not.

**A confound the data cannot settle, and I will not pick a side of it.** rig-glm notes the shortfall may be non-byte work
OR IQ4_XS/Q6_K costing more compute per byte than Q8_0 on 300 W Max-Q cards, where compute is scarce relative to the
500 W parts. Those are different mechanisms with different fixes, and a per-kernel nsys of served-vs-A on the cards
separates them. **The 500 W cards were never measured**, and A could well do better there.

### The ship decision: NONE of A, B or C ships. (rig-lead, as the lever's owner, Oct 1, 2026)

| recipe | decode d32k | KLD | **same top** | PPL |
|---|---|---|---|---|
| A | +5.16 % | 0.0326 | **93.54 %** | x1.0122 |
| C | +5.31 % | 0.0339 | 93.32 % | x1.0156 |
| B | +11.11 % | 0.0434 | 92.52 % | x1.0222 |

**A changes the top token on 6.5 % of positions to buy 5 %.** One generated token in fifteen differs from the Q8_0 model.
For a head whose purpose is to be the best way to serve this model, that is the wrong side of the trade, and B's +11 %
costs 7.5 %. The comparison that settles it is on the same page: **`-ub 1024` bought -23 % of deep prefill at ZERO quality
cost.** A lever with no quality cost at four times the effect is what shipping looks like; this is not.

**Kept, not discarded**, because two things could change the answer: the 500 W cards are unmeasured, and the dequant
confound may be hiding a larger gain. The recipes stay at `heads/glm-5.3-flash/recipes/` with this result recorded beside
them, and the re-test is one paired `tiertime` leg on 500 W parts plus one per-kernel nsys.

**What the quality data retired, which is a real gain.** Every recipe's error **falls** with position (A: 0.082/89.3 % ->
0.036/93.0 % -> 0.030/93.9 %), and B's excess over A is a constant 1.37/1.32/1.33x by bin. **Nothing compounds through the
KDA recurrent state.** That was the risk the whole position-binned design existed to catch, it was the stated reason B and
D were ranked as higher-risk, and it is now measured absent. The binning was still the right instrument — it is how we know.

**And the instrument note worth keeping: rep 1 of every llama-bench run reads ~20 % slow on these cards** (131.9 -> 163.2),
so rates must come from reps 2-3. An `avg_ts` over all reps would have diluted every ratio in this table toward zero.

**Correction to the retraction above: it over-retracted, because 55 % and 85.7 % are not the same quantity.** The
retraction read the cards' 1.93 TB/s as "the target is not at its roofline." That does not follow:

- **1.93 TB/s is token-level** — bytes-per-token x tokens-per-second (11.860 GB x 163.0 tok/s = 1.93 TB/s, reconstructed
  from rig-glm's own figures). It averages over every microsecond the token spends *not* running a bandwidth-bound kernel.
- **85.7 % was per-kernel** — `dram__throughput.avg.pct_of_peak_sustained_elapsed` for `mul_mat_vec_q`, measured only while
  that kernel runs.

**Both can hold simultaneously**: kernels at 86 % of peak with the token at 55 %, the gap being attention, scalar work,
launch, sync and dequantization. That is f < 1 restated, not a contradiction.

So the retraction's surviving claim is the narrower one: **76 % of the token on the serving cards is
non-byte-proportional, so non-byte headroom exists there**, and every scheduling null measured on the dev pair is
unproven-there rather than dead. **Where the cards' matmuls sit against their own DRAM peak is unmeasured by anyone**, and
that is the 20-minute per-kernel ncu census that would settle whether decode on the serving hardware is finished or not.
The command and its two traps (`--cache-control all`, and shares-not-durations under application replay) are with rig-glm.

The pattern worth naming, since it is the third instance today: **a ratio is only comparable to another ratio with the same
denominator.** The campaign's 2816 GB/s-against-960 was an L2 denominator read as DRAM; the first prefill census labelled
compute-bound tensor-core matmuls "below roofline" against a DRAM denominator; and this retraction compared a token-level
average against a per-kernel peak. Same error class, three different costumes.

### The 25.1 % prefill share was a window artifact; the served shape is 12.2 % / 9.7 % (Oct 1, 2026, 2:31 PM CT)

rig-glm re-sized the recurrent KDA kernel on the served shape — both cards, `-sm tensor`, `-b 4096 -ub 1024`, nsys device
time with node tracing — and got **12.2 % of a 4k prefill and 9.7 % of a 28k one**, against the 25.1 % recorded in
`68ac2df`. **Their number supersedes mine, and the cause is a flaw in my census rather than in theirs.**

**`--launch-count 150` truncates; it does not sample.** My capture profiled **6** `gated_delta_net_cuda` launches out of
150 total, while a 4096-token prefill at `-ub 512` contains roughly 264 KDA launches alone. The window covered a fraction
of one ubatch, starting wherever ncu happened to begin, so those "shares of prefill kernel time" were shares of an
arbitrary 150-launch window. The six KDA launches it caught were long ones, and the share inflated accordingly. **A share
needs a complete denominator; this one had a truncated one** — the same error class as the three denominators above, in a
fourth costume.

**What survives from that census is the per-kernel part, which is what the lever actually rests on:** the recurrent KDA
kernel at **0.0 % tensor-core utilization and 86.2 % SM issue**. Those are per-launch properties, independent of how many
launches the window held, and they are why a 2-4x is plausible at all. **Re-sized, chunked KDA is worth about 5-9 % of
prefill, not 12-19 %.**

**rig-glm's trap, which is a real one and worth the campaign's attention: plain `nsys -t cuda` hides every kernel inside a
replayed CUDA graph.** Their first capture read 264 KDA launches at both 4,096 and 28,672 tokens — flat, because the graph
hid the scaling — and `--cuda-graph-trace=node` corrected it to 528 -> 3,696. **The decode captures in this campaign are
not affected**, checked rather than assumed: `nsys-orig.sqlite` carries a `CUDA_GRAPH_NODE_EVENTS` table with 756,042
kernel rows, while the twelve captures deliberately named `pl-nonode-*` hold 364 `GRAPH_TRACE` rows and 23,260 kernels. The
distinction was made on purpose at capture time, so the decode exclusive-occupancy table stands.

> **SUPERSEDED by rig-glm's own PDL finding, recorded immediately below.** The `657 µs each`, `5 % of DRAM peak` and
> `6.1 % / 4.9 % of prefill` in this paragraph are summed nsys durations, which under Programmatic Dependent Launch bill the
> kernel its wait on its predecessor. On the served shape the norm's 690 µs is **622.6 µs of overlap with the preceding
> `gated_delta_net_cuda` plus 67.8 µs of its own work**, so the real figure is **0.64 % of device time**, not 6.1 %, and the
> warp-per-row rewrite is worth **0.41 %**. The rewrite is still correct and still landing — it is 2.8x on the kernel's own
> time — but it is not the 5 % lever this paragraph describes. Read the next section before quoting any number here.

**And the census found a cheaper lever than the one it was aimed at**, which rig-glm is building first: a single
`rms_norm_f32<256, true>` instance, the KDA per-head output norm, 528 launches taking 657 µs each to normalize 33.5 MB —
**about 49 GB/s, 5 % of DRAM peak**, and 6.1 % / 4.9 % of prefill. A 256-thread block per 128-wide row, half of it idle,
two barriers, and 32,768 near-empty blocks. The fork already has a 128-thread path for `ncols <= 128` but gates it to
DGX Spark. The fix is one warp per row, 8 rows per block, with the butterfly run per virtual warp so the arithmetic order
and therefore every output bit is preserved.

### PDL makes summed nsys durations double-count, and two small corrections to this file's method (Oct 1, 2026, 3:14 PM CT)

**rig-glm found the fifth denominator error of the day, and it is the most general one: under Programmatic Dependent
Launch, nsys bills a kernel the time it spends waiting on its predecessor.** A kernel launches early, waits in
`ggml_cuda_pdl_sync`, and its reported duration contains that wait. On the served shape the KDA output norm's 690 µs is
**622.6 µs of overlap with the preceding `gated_delta_net_cuda` and 67.8 µs of its own work** — so the "rms_norm at 6.1 %
of prefill" was mostly KDA's tail wearing rms_norm's name. Summed durations over-count **5.8 % (p4096)** and **4.7 %
(p28672)** of the cards' time, and the error concentrates on whatever follows a long PDL predecessor. Their `shares.py`
now bills `end - max(start, previous end on stream)`.

**Checked rather than assumed: this file's decode rows are NOT affected, because they were already exclusive.** The decode
table reports EXCLUSIVE occupancy — `mul_mat_vec_q` 2.192 ms exclusive of a larger total, the elementwise pool 0.078 ms
exclusive against 1.274 ms total, the all-reduce 100 % exclusive — which is the same quantity rig-glm's corrected script
computes. **PDL in fact explains those gaps rather than undermining them**: the kernels I measured as "almost entirely
overlapped" were launching early and waiting, which is why `mul_mat_vec_f_vec` showed 1.256 ms total against 0.110 ms
exclusive. The conclusion drawn from it — that lifting that kernel was worth at most its 0.110 ms — stands, and stands for
a better-understood reason. **The retired prefill census (`68ac2df`) used ncu per-launch durations, so it carries the
launch-count truncation already recorded above, not this one.**

**Correction to this file's intervals: they used 1.96 where a small sample needs Student t.** At n=10 the right multiplier
is t(0.975, df=9) = **2.262**, not 1.96. The byte lever's published interval **+14.88 % to +17.00 %** should read
**+14.72 % to +17.16 %**. No conclusion changes — it excludes zero either way, and excludes the superseded +10.93 % either
way — but the method was wrong and the harnesses are fixed.

**Where that same error would have mattered, which is the point of fixing it.** rig-glm's warp-rows norm A/B reads
**+1.5 ± 0.5 % over 4 pairs** against **+0.4 %** predicted from kernel time. At n=4, t(0.975, df=3) = **3.182**, giving
**-0.09 % to +3.09 %** — which contains the prediction, so "unresolved" is correct. Computed with 1.96 the interval is
**+0.52 % to +2.48 %**, which EXCLUDES the prediction and would have invited a claim of beating it. **Same data, opposite
conclusion, decided entirely by the multiplier.** Small-n paired A/Bs are now the campaign's main instrument, so this is
not a pedantic fix.

**And the convert_unary hypothesis confirmed, with the count that tested it:** 374 `f32->bf16` activation-cast launches per
ubatch per card at p4096, one per BF16 matmul — 88 of them (the `hc_{attn,ffn}_fn` inputs, 1024x16384) at 155.8 µs each
and 61 % of the family. rig-glm declined F32 weights with a number: those 289 tensors are 448.5 MiB, re-read every decode
token. The better fix is theirs — a norm writing bf16 straight into the cuBLAS `src1`, bit-identical, cutting the hc
chain's 308 µs to ~114 µs — because **94 % of that chain is moving an activation for a 20.8 µs GEMM.**

### The FA lever: its mechanics can be built on the proxy, its SIZE cannot be measured there at all (rig-lead, Oct 1, 2026, 3:58 PM CT)

Taking the lever rig-glm handed off. At p28672 flash-attention is **22.0 %** of own device time, second only to `mmq`'s
27.5 %, and it is the one item that grows with depth — on a head served at `-c 524288` it eventually dominates. The sparse
kernel is **one query a tile, by construction**: `fattn.cu:244` says so in a comment — *"the sparse variant is one token a
tile, whatever the batch"* — and passes a literal `1` where the dense path switches over `ncols1` 1/2/4/8. So each query
separately gathers its own ~2080 cells: 264 launches at 9.1 / 11.9 ms, against the dense kernel's 286 at 6.3 / 10.4 ms.

The fix rig-glm names, as DeepSeek's sparse-MLA prefill does it, is a tile of queries sharing the union of their gathered
cells with a per-query mask. **Its whole value is how much adjacent queries' selections overlap:**

| tile | no overlap | 80 % overlap | 95 % overlap |
|---|---|---|---|
| 4 | 4.0x | 1.6x | 1.2x |
| 8 | 8.0x | 2.4x | **1.4x** |
| 16 | 16.0x | 4.0x | 1.8x |

(KV cells loaded per tile, as a multiple of one query's load.) At 95 % overlap a tile of 8 loads 1.4x what one query does
instead of 8x — about **6x fewer KV loads**. At no overlap it loads 8x and saves nothing: the same work in a bigger kernel.

**And here is the thing that decides who can do this work: the L44-E8 proxy has RANDOM WEIGHTS, so its indexer scores are
arbitrary and adjacent queries' selections are uncorrelated BY CONSTRUCTION.** The proxy will report the "no overlap"
column however good the trained model's structure is — 2048 of 28,672 cells chosen at chance is ~7 % overlap. **Selection
overlap is a property of the trained attention, so this lever's size is not measurable on the proxy, by any instrument,
ever.** That is a different class of limit from the ones recorded above: not a wrong denominator or an inflated duration,
but a question the available artifact cannot answer.

**What follows, and it is not "defer":**
1. **The mechanics are proxy-testable in full** — a query-tiled sparse kernel's correctness, its bit-identity against the
   one-query-a-tile path, and its cost *for a given union size* can all be built and gated here. Union size can even be
   forced synthetically to sweep the curve above.
2. **The economics need real weights**, which means one leg on a box: dump the indexer's top-k per query at a few depths
   and compute `|union| / (n x |individual|)` for n = 2, 4, 8, 16. That is a cheap measurement on a box rented for any
   other reason, and it is worth attaching to the next one rather than commissioning its own.
3. **The outside evidence is real and should be weighted**: DeepSeek ships exactly this for their sparse-MLA prefill, which
   is itself evidence that trained selections overlap enough to pay. That justifies building behind a switch before the
   overlap number exists — but it does not justify quoting a predicted gain, and no number for this lever belongs in this
   file until it is measured on real weights.

Built and parked (Oct 1): the kernel passed the eval cases that reach it and failed both mutants, then GLM-V1-PLAN.md
moved it to v2 because it would ship off. It is in the engine history as `fbd07f7b9` and its revert `cae423d3c`, so
v2 restores it with one `git revert`. The result is in the gate paragraph above ("a fourth way").

### Chunked KDA prefill: code landed, serving pin HELD until the trained model's KL exists (rig-lead, Oct 1, 2026, 4:46 PM CT)

rig-glm's chunked KDA is on `train/engine-10` (`7755e858a`, `4ceff2af6`, pushed). It halves KDA's own prefill kernel time —
**1232 -> 607 µs a layer on the 5080, 1292 -> 650 on the 5070 Ti** — for **-7.4 % device time** and **pp4096 +2.93 %, 95 %
CI +1.60 to +4.26 over 6 interleaved pairs**, resolved. Tests 79/79 and 80/80 on both cards with and without the switch.

**It is NOT bit for bit**, and that makes the serving decision different from the three commits before it.

**Decision: the code lands, the pin does not move.** The published image and template stay at engine `765692590`, which
does not contain it. `heads/glm-5.3-flash/head.toml` is unchanged.

**Why, in the one comparison that matters.** The proxy's KL against the recurrent base is **0.00646 at -c 2048 and 0.00655
at -c 16384** (positions 8192-16383, so no growth along the context — the KDA-accumulation risk was checked and is absent,
against a 0.000000 floor). That is the same ORDER as two changes already in serving: `-ub 512 -> 256` alone costs 0.0058 and
the reduction order alone costs 0.0026. **But every one of those is a proxy number, and the bar we serve elsewhere — 0.0015
mean, 98.3 % same top — is a TRAINED-model number.** This proxy has PPL 344k and is chaotic, which inflates any KL measured
on it. So the honest statement is not "4x the bar": it is that **the trained model's KL is unknown**.

**CORRECTED Oct 1, 2026, 10:05 PM CT, on the real pack (box 53787584, section below).** The trained model's KL for
chunked KDA is **0.011835 ± 0.000251, same top p 96.475 ± 0.102 %** at `-c 2048`, against the proxy's 0.00646. The
proxy **understated** it about 1.8×, so this paragraph's reasoning that a chaotic proxy "inflates any KL measured on
it" had the direction wrong. What survives is the comparison it reached for: on the trained model a `-ub 1024 -> 512`
change alone costs 0.011509 and a tensor-to-layer change 0.011627, so chunked KDA is still the size of a
reconfiguration already shipped. Chunked KDA **ships ON** (rig-orchestrator, below); `GGML_CUDA_KDA_CHUNKED_LEGACY=1`
is its off switch.

**And consistency decides it.** This afternoon recipe A was refused for a MEASURED 6.5 % top-token change buying +5-6 %
decode. Shipping an UNMEASURED quality cost for +2.93 % prefill would be a weaker decision, not a stronger one. Holding
costs nothing: there is no box up, the code is on the branch, and the next pin move picks it up for free.

**What unblocks it is one cheap leg, exactly as rig-glm specified**: on the first box with the real pack, run
`llama-perplexity --kl-divergence` with and without `GGML_CUDA_KDA_CHUNKED_LEGACY=1`. Two runs, no serving change. Under the
bar, the pin moves and the image rebuilds. Over it, the switch goes in the template env and the code stays for the next
refinement. **Either way the measurement is cheap and the decision stops being a judgement call.**

**The gate finding in that commit is the best methodological catch of the campaign, and it is a new failure class.** KDA's
test gate was drawn i.i.d. per token, so every channel was fast on some token of every chunk and **no state ever reached a
chunk boundary**. A mutant that decays each state row by its NEIGHBOUR channel's `G_last` therefore PASSED a 32-head
512-token case at NMSE 9e-8 — a float64 simulation puts its true effect at 1e-13 to 2e-11, because the data never stressed
it. The fix was the test's DATA, not its cases: `init_kda_gate` now gives each channel a rate plus per-token jitter, and all
7 chunked cases fail that mutant at NMSE 0.75-1.72 while the real path stays at <= 1.26e-7 under a 2e-7 bar.

**So the campaign has now produced three distinct ways a green gate can be empty**, and they are worth keeping together:
a ratio taken against the wrong denominator (five instances), a mutation that is null in this build (`__expf` under
`-use_fast_math`), and **a test whose data distribution cannot reach the boundary the test exists to check**. The third is
the subtlest: the cases were complete and the coverage was a lie. The scalar gate's activated test had the same i.i.d.
shape (a chunk boundary reached with P = 2e-27); `d5cff0d48` gives it per-head rates too, and rig-glm's neighbour-head
mutant (stage 3 decaying by `exp(G_last)` of head `bh^1`) fails **all 24 chunked scalar-gate cases** at NMSE 2.0e-4 to 3.4
against a 2e-7 bar, KDA's cases unaffected (`local/research/glm53-kda-chunk-2026-10-01/gdnmut-{1,2,fusion}.log`).
**A fourth way, found Oct 1: the gate's subject never executed.** Every batched sparse flash-attention case (`nb` 8-512)
sat in `make_test_cases_perf()`, which never compares against a backend, so eval reached the sparse path with at most
**3 queries** — the shipped one-query-a-tile path included. A probe in the dispatch found it; four eval cases now reach it
(`nb` 8, 9, 16 at kv 4096, and `n_kv_max` 2080 at kv 16640), and they stay in v1. The query-tiled kernel they were written
for (8 rows sharing one union list) passed them and failed two mutants at ERR 0.82-1.65 against 0.0005, and is parked as
a v2 candidate by GLM-V1-PLAN.md: its value is the trained indexer's selection overlap, which the proxy cannot show.

### The MTP catch-up fold engages and is slower on BONSAI; GLM is unmeasured for it (rig-lead, Oct 1, 2026, 8:18 PM CT, corrected 8:23 PM CT)

Lever 3(a). A round at n_max 3 decodes the MTP head four times, once in `process()` to carry the
verified rows into the head's cache and once per draft step. Those rows are only context for the
first draft, so they can ride in that decode. Built (engine `f911f3ed6`), measured, reverted
(`9e5976d7e`); one `git revert` restores it.

**What was measured, and on what.** `local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf`,
one RTX 5070 Ti (card 1) through gpu-lease, `--spec-type draft-mtp --spec-draft-n-max 3`,
`--spec-draft-mtp-vocab mtp-draft-vocab-98304.i32`, `-c 8192`, T 1.0 / top_p 0.95 / top_k 20, ONE
prompt, 2x320 tokens an arm, seed 1234, arms `LLAMA_MTP_FOLD_LEGACY` unset vs 1.

**This is not GLM.** glm-5.3-flash's served pack is 134 GB and does not load on a 16 GB card. Bonsai's
MTP head carries a 98,304-entry draft-vocab trim that GLM's NextN path does not have, and is a
different head. **GLM's NextN path is UNMEASURED for this fold.** The row sits in this file because
3(a) is a GLM lever and the two counts below are why it is parked.

| arm | decode calls | calls a round | tok/s | graph rebuilt |
|---|---|---|---|---|
| legacy | 1664 | 4.92 | 119.8, 123.3 | 1.7 % of decodes |
| fold | 1472 | 3.98 | 108.8, 107.4 | **8.8 % of decodes** |

The lever provably engages: 4.92 -> 3.98 calls a round is the predicted 5 -> 4, and a decode count is
weight- and timing-independent. It is still slower, for a reason absent from the estimate: the folded
first draft decode is n_accepted+2 rows wide, so its shape changes every round and
`llama_context::process_ubatch` cannot take the `res->can_reuse(gparams)` path. Losing reuse entirely
costs 16 % upstream (ggml-org/llama.cpp#20605, 91 -> 76 tok/s).

**Acceptance: NOT established, and worth a check before 3(b).** The arms recorded 151/501 = 30.1 %
against 135/547 = 24.7 %. That is z = 1.98 even under the false assumption that drafts are
independent; drafts within a sequence are correlated and this is one prompt at T 1.0. A fold that
only changes batching should leave acceptance near unchanged, so a real drop would point at a
token / h-row pairing fault rather than at batching, and 3(b) reuses that pairing. Not checked here
because the lever is parked; the check, if 3(a) is ever revived or 3(b) sees odd acceptance, is the
same build, 5+ prompts, T 0, both arms.

**The ceiling, as a hypothesis and not a refutation.** On GLM's bytes a head decode reads one NextN
block, 0.280 GB, against the trunk's 11.860 GB a token, so dropping one of four head decodes saves
0.280 of 12.98 GB a round = 2.2 %. That is a bytes-only figure, and it does NOT bound the lever: the
plan's +4-6 % came from the host round-trip tax the rental's round-time fit measured, 2.7 ms fixed a
round and 283-671 us idle between graphs, which a removed decode also removes. Those are different
mechanisms and the bytes number does not show the estimate wrong. A constant-width fold may still pay
on GLM.

**Parked because 3(b) subsumes it, not because it cannot pay.** Padding the batch to a constant
1+n_max+1 rows would restore graph reuse, but the unrolled draft chain builds one constant-width
graph for the whole round and drops the catch-up decode as a side effect, so the padding is work 3(b)
discards.

**The transferable constraint, which binds 3(b): an MTP draft batch whose width varies loses graph
reuse, and reuse is worth more than any single head decode.** 3(b)'s unrolled chain is naturally
constant-width and so is compatible; 3(a) never was.

Confound, stated: the two arms ran sequentially on card 1, and a card-0 lease spilled onto card 1 for
~5 s that evening (rig-orchestrator). The tok/s pair therefore carries a timing confound. The verdict
does not rest on it: the call count and the rebuild rate are counts.

### Draft sampling is re-seeded RANDOMLY every round: acceptance is not comparable between runs (rig-lead, Oct 1, 2026, 8:45 PM CT)

**CORRECTED Oct 1, 2026, 11:01 PM CT (rig-builder): drafts are argmax, so the reseed never reaches them.** Every
drafter takes `cur_p->data[0].id` after `common_sampler_get_candidates(smpl, true)` has sorted the candidates by
probability (`common/speculative.cpp:345, 823, 1369, 1388, 1816` and `common/sampling.cpp:911` at 7a4c89ca1). The dist
stage's draw only sets `selected`, which no drafter reads. Measured with Qwen3-0.6B Q4_0 as target and draft
(draft-simple, n_max 3, CPU build): the same `/completion` three times at seed 1234, T 1.0, top_p 0.95 gives identical
text with 228 drafted and 82 accepted, on the engine as it is (per-round random reseed) and with a per-position seed
alike. With no request seed every run differs, because the target samples randomly. So consequence 1 has the wrong
mechanism: run-to-run acceptance noise comes from the target's sampling when a request names no seed, and a seeded
request is reproducible end to end today. Consequence 4 is wrong: the determinism gate passes on an unmodified engine.
Consequence 3 holds for a stronger reason: drafts consume no randomness at all. The per-position draft seed is not
built, because it changes no output; its patch and the run are in `local/research/draft-seed-2026-10-01/`.

Found while checking whether 3(b) could compute draft steps past the host's early stop without
shifting later rounds' draws. It cannot shift them, and the reason invalidates run-to-run acceptance
comparison on this engine. Each link read, not inferred:

| step | where | what it does |
|---|---|---|
| the draft sampler is built | `common/speculative.cpp:1493-1499` | sets only `no_perf`, `top_k=10`, `samplers={TOP_K}`; **never sets `seed`** |
| its seed therefore defaults | `common/common.h:225` | `uint32_t seed = LLAMA_DEFAULT_SEED` |
| the chain still draws randomly | `common/sampling.cpp:521` | appends `llama_sampler_init_dist(params.seed)` by default, so a draft is sampled, not argmax |
| every round resets it | `draft()`'s setup loop -> `common/sampling.cpp:180` | `common_sampler_reset` -> `llama_sampler_reset(chain)` |
| reset re-draws the seed | `src/llama-sampler.cpp:1246-1252` | `seed_cur = get_rng_seed(ctx->seed); rng.seed(seed_cur)` |
| and that is entropy | `src/llama-sampler.cpp:341-349` | on `LLAMA_DEFAULT_SEED` (0xFFFFFFFF, `include/llama.h:37`) returns `std::random_device` or the system clock |

**Consequences.**
1. A request's `seed` does NOT make drafting reproducible. Two runs of the same prompt at the same
   seed draw different drafts, so **served acceptance differs run to run as noise**. Any acceptance
   figure from a single run, or any A/B of acceptance between two runs, measures that noise. This
   is the mechanism behind the 3(a) row's 30.1 % against 24.7 %, which is retracted there as
   unestablished on statistics alone; the mechanism is stronger than the statistics.
2. It is noise and not bias, so acceptance stays measurable with enough prompts and a stated
   interval. Any acceptance row in this file without that is a single draw.
3. For 3(b): extra draft steps cannot shift later rounds, because every round re-seeds. The
   truncation design is safe, but by accident of the reseed rather than by design.
4. A determinism gate on drafting cannot be written against this code at all: "same seed, identical
   draft stream" fails on an unmodified engine.

**The fix, approved and sequenced after rental 1's legs:** derive the round's draft seed from the
request's seed AND the round's position (a hash of request seed and `n_past`), keeping today's
random behaviour when the request names no seed. Not a single constant reset each round: that
replays one uniform stream per round and correlates the draws across rounds (rig-orchestrator).

### The real-pack KL verdict, and the cuBLAS artifact that nearly convicted three bit-for-bit commits (rig-builder, Oct 1, 2026, 9:56 PM CT)

The leg section 1440 above specified, run on the real pack at last. Box **53787584**, 2× RTX PRO 6000 Blackwell
Max-Q, 97,887 MiB each, `$2.450/h`, Texas. Pack
`local/packs/glm-5.3-flash/IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-0000{1,2,3}-of-00003.gguf`, three shards,
45,672,911,040 + 45,407,495,392 + 43,225,116,416 = **134,305,522,848 bytes**, hashes as `head.toml` pins them.
Corpus `local/kl/corpus.txt`, 3,150,612 bytes. Served geometry throughout: `-ngl 99 -sm tensor -ts 1,1 -fa on -b
4096 -ub 1024`, `GGML_CUDA_ALLREDUCE=internal`, `GGML_CUDA_GRAPH_MAX=0`, both cards.

**Read the instrument before the result.** The pin's library against its own base file reads **mean KLD 0 ± 0,
same top p 100 ± 0 %**. The engine is exactly reproducible at fixed geometry, so every figure below is a real
difference and not run-to-run noise. The same control also shows `PPL ratio` is **not** a discriminator at the
fourth decimal: it reads 1.000074 where KLD is exactly 0, so ~1e-4 of it is instrument.

**Five runs were wrong, and the cause was underneath the engine.** The first pass measured the candidate
`81aca7542` at 0.011990, and `GGML_CUDA_KDA_CHUNKED_LEGACY=1` only moved it to 0.011121. Adding
`GGML_CUDA_HC_POST_ROWS_LEGACY=1` and `GGML_CUDA_HC_POST_NORM_LEGACY=1` changed nothing, to every digit. Nor did
`GGML_CUDA_GDN_CHUNKED=0`. Nor did swapping the pin's own `libggml-cuda.so.0` into the candidate's directory, which
is the test that broke it open, since it put the difference somewhere other than the library under test. A
file-by-file comparison of the two directories — the denominator that should have been taken first, instead of a
list of six libraries — found it:

```
only in 7656925-sm120:  ./BUILD  ./libcublas.so.13.5.1.27  ./libcublasLt.so.13.5.1.27  ./libcudart.so.13.3.29
only in 81aca75-sm120:  ./llama-perplexity
```

The pin's directory was installed by rig, so it carries the NVIDIA runtime archives `build/build-publisher.ts`
unpacks beside the libraries **at install**. The candidate's was unpacked from a raw portable tarball by hand, which
carries none (`tar -tzf` on a published tarball matches zero `cublas`/`cudart` entries), so it fell back to the box's
system cuBLAS. **cuBLAS selects its GEMM algorithm by version**, so every candidate row was comparing two cuBLASes
and reporting it as an engine difference. An nsys capture taken before this was found confirms the switches were
never the problem: only `dsv4_hc_post_f32`, `dsv4_hc_pre_f32`, `dsv4_hc_weights_f32` and `gdn_kda_precompute_decay`
ran, with no `dsv4_hc_post_rows_f32` and no `cgdr_*` kernel anywhere.

**With the runtime equalised, every bit-for-bit claim in the range is TRUE**, measured rather than argued:

| library, all against the pin's base, runtime equalised | mean KLD | same top p |
|---|---|---|
| pin `765692590`, identical geometry (the control) | 0 ± 0 | 100 ± 0 % |
| candidate `81aca7542`, all three legacy switches | **0 ± 0** | **100 ± 0 %** |
| candidate `81aca7542` as it stands (chunked KDA on) | 0.011835 ± 0.000251 | 96.475 ± 0.102 % |

So `fd5176e75` (the hc post kernel and its fused flat norm) and `67275eacf` (KDA's fp16 stage 3) reproduce the pin
**to zero** on the trained model. The third row is the first uncontaminated measurement of **chunked KDA alone**.

**Section 1440 assumed the proxy INFLATED this. It understated it.** That section reasoned "this proxy has PPL 344k
and is chaotic, which inflates any KL measured on it". The proxy read 0.00646; the trained model reads 0.011835,
about **1.8× higher**. The direction of that assumption was wrong and the correction is marked in place above.

**The 0.0015 / 98.3 % bar is below this engine's own reduction-order floor, so it cannot separate a defect from a
summation order.** The pin's own library, against its own base, with one legitimate configuration change:

| the pin's library, one config change | mean KLD | same top p |
|---|---|---|
| `-ub 1024` → `512` | 0.011509 ± 0.000193 | 96.631 ± 0.1 % |
| `-sm tensor` → `layer` | 0.011627 ± 0.000238 | 96.395 ± 0.103 % |

Chunked KDA's 0.011835 sits **on** that floor: the cost of changing the ubatch, on shipped, accepted configuration
choices. For outside scale, upstream's own `tools/perplexity/README.md` puts LLaMA 3 BF16 against FP16 at same top p
**99.739 %**, and in upstream issue 25593 a same-top near **96.5 %** was the signature of fp32 math truncated to
fp16. So 96.5 % is a large number in absolute terms on this model, and this engine reaches it from a ubatch change
alone — upstream documents the mechanism as batch variance by design (PR 16016's deterministic mode exists to remove
split-K and fix reduction order; issue 7228: "the optimal order to do the operations in is simply different for
different batch sizes").

**The error does NOT grow with depth; it shrinks.** At `-c 16384`, 4 chunks, against a fresh pin base at 16k
(`base16k-7656925.kld`, 10,149,500,916 bytes, sha256 `beda08854b03`):

| at `-c 16384` | mean KLD | same top p |
|---|---|---|
| candidate as it stands | 0.007412 ± 0.000119 | 97.366 ± 0.088 % |
| pin at `-ub 512` (the floor) | 0.006932 ± 0.000128 | 97.491 ± 0.086 % |

Both fall from 2k to 16k, −37 % and −40 %, and the candidate's ratio to the floor is 1.028 at 2k against 1.069 at
16k, so it tracks the floor rather than accumulating against it. The recurrent-state risk this leg existed to check
is absent at a context eight times deeper. On same top p the candidate is **1.0 σ** from the floor, indistinguishable.
On mean KLD the difference is 0.000480, and how significant that is depends on which σ, so all three readings are
recorded rather than the one that passes:

1. **three times the floor's own σ** (0.000384): over by 1.25 σ, fails.
2. **σ of the difference assuming independence** (0.000175): 2.75 σ, passes a 3 σ bar. This is what I first
   recommended, and it is wrong in the lenient direction.
3. **paired** (rig-orchestrator): both sides are scored against the same base on the same tokens, so their per-token
   KLDs are positively correlated and the paired σ is **smaller** than 0.000175, putting the difference above 2.75 σ.
   A paired test would likely fail 3 σ. Assuming independence between two measurements that share their tokens is the
   same error this campaign already made once, on acceptance.

**So significance is the wrong instrument here, and the decision was taken on damage instead** (rig-orchestrator, as
plan owner, Oct 1, 2026, 10:02 PM CT): two non-identical numerics always differ given enough tokens, so a σ bar only
asks whether the corpus was long enough. The question the gate exists for is whether chunked KDA damages the model
more than a reordering already shipped. Same top p is at the floor (1.0 σ); KLD is 1.03× the floor at 2k and 1.07× at
16k; both fall with depth. **Decision: chunked KDA ships ON**, with `GGML_CUDA_KDA_CHUNKED_LEGACY=1` as the documented
off switch. The gate is re-specified **prospectively**, set after seeing these numbers and marked as such: a
non-bit-exact change passes when, on the real pack against the pin's base with the runtime equalised, its **KLD ≤
1.10× the floor at each depth measured**, its **same top p within 0.25 points of the floor**, and neither grows with
depth.

**Two fixes landed from this.** The prebuilt engine shipped `llama-server`, `llama-bench` and `llama-kv-mean-center`
but not `llama-perplexity`, so no rented box could measure the KL of the engine it was running — the one machine
where a 134 GB pack's KL can be measured at all, while the lab's own contract claimed installed builds worked
(`lab/engine-lab.service.ts:27`). Added to `TARGETS`, on main as `439e135`. And `ab()` now refuses a pair whose two
sides resolve a different `libcublas`, `libcublasLt` or `libcudart`, before the first bench runs rather than after
minutes of rented card, on `train/rig-0.1.8` as `898b703`; mutating its condition to a constant false fails the drift
test and only that test. **A fifth way a green gate can be empty, for the list section 1440 keeps: the measurement
compared two things that differed underneath the thing under test, and nothing in the instrument could see it.**

### Rental 1's speed legs, the 3(b) capture, and the pin move to 3d40ae99c (rig-builder, Oct 1, 2026, 10:43 PM CT)

Box 53787584, two RTX PRO 6000 Blackwell Max-Q (97,887 MiB, sm_120), the real IQ3_XXS pack, every side on the pin's
CUDA runtime (cuBLAS and cuBLASLt 13.5.1.27, cudart 13.3.29, checked byte for byte before each leg, the section above).
Served shape throughout: `-sm tensor -ts 1,1 -fa on -b 4096 -ub 1024`, `GGML_CUDA_ALLREDUCE=internal`,
`GGML_CUDA_GRAPH_MAX=0`.

**Lever 2 is bit for bit on real routing.** 3d40ae99c with `GGML_CUDA_KDA_CHUNKED_LEGACY=1` against the 765692590 base
(2k, 32 chunks): **KLD 0 ± 0, same top p 100 %**. The proxy's claim held where the experts are the trained model's.

**Lever 2's tile width on the cards** (llama-bench, tok/s, 3d40ae99c):

| config | pp4096 | pp32768 |
|---|---|---|
| `GGML_CUDA_MMQ_MOE_TILES_LEGACY=1` | 2718.21 ± 4.31 | 2239.39 ± 2.38 |
| `GGML_CUDA_MMQ_MOE_NCOLS=32` | 3241.65 | 2585.11 |
| `GGML_CUDA_MMQ_MOE_NCOLS=48` | 3293.53 | 2621.84 |
| `GGML_CUDA_MMQ_MOE_NCOLS=64` | 3222.69 | 2577.36 |
| **default** | **3300.33 ± 4.50** | **2625.48 ± 6.23** |

The default is +21.4 % at pp4096 and +17.2 % at pp32768 over LEGACY, and no fixed width beats it.

**The engine-10 line before lever 2 against the pin** (765692590 against 81aca7542, the same tree as 9e5976d7e;
`rig engine ab`, alternating pairs):

| shape | pairs | mean |
|---|---|---|
| pp32768 | +5.90, +6.07, +6.01 % | +6.0 % |
| tg128 at depth 0 | +0.33, −0.14, +0.13 % | flat |
| tg128 at depth 32k | +0.08, −0.03, −0.09 % | flat |

**Served on the tip** (3d40ae99c as llama-server, rig's own plan with only the build directory swapped; ready in
38 s; 12 prompts at T 1.0, top_p 0.95, n_max 2): decode **154.7 tok/s** (95 % CI 148.7–160.6), acceptance 0.6125
(0.5546–0.6703, random-seeded drafts, the section at 1553), a 32,721-token prompt prefilled at 2215.1 tok/s and decoded
at 146.9 tok/s.

**Where prefill time goes** (nsys, own kernel time, the 765692590 library; a 1024-token ubatch at depth 14336, 11.78 s;
and the tip at depth 131072, `-p 1024 -n 0 -d 131072`, 93.0 s):

| kernel family | depth 14336 (pin) | depth 131072 (tip) |
|---|---|---|
| mul_mat_q | 61.23 % | 43.93 % |
| flash_attn_ext_f16 | 8.83 % | 16.92 % |
| lightning_indexer | 1.79 % | 15.32 % |
| gated_delta_net / KDA | 6.54 % | 1.65 % + 0.88 % (state, fwdsub) |
| dsv4_hc_post | 3.83 % | 2.07 % |

At 128k the DSA side (sparse flash attention and the lightning indexer) is 32 % of prefill, against 10.6 % at 14k:
that is what the sparse gather's 32-head tile (258292bd8) and its switch-over are for, measured on the next rental.

**The 3(b) gate: removable host gaps between draft steps are 13.3 % of a round at n_max 2.** nsys in graph mode on
the 765692590 library, 640 decoded tokens a capture. Every round runs one verify graph (12.6–13.1 ms) and n_max + 1
draft-head graphs (475–563 µs: the MTP catch-up, then n_max steps). Gaps billed by the graphs on each side, from
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE` (`local/leg-3b/transitions.py` on the box):

| n_max | round | draft→draft a round (card 0 / 1) | share | verify→draft | draft→verify |
|---|---|---|---|---|---|
| 2 | 14.873 ms | 1971 / 1998 µs | 13.25 / 13.43 % | 2.0 / 2.1 % | 5.3 / 7.1 % |
| 3 | 18.788 ms | 1957 / 1996 µs | 10.42 / 10.63 % | 1.7 / 1.8 % | 5.5 / 7.2 % |

At n_max 2, 35 % of the draft→draft gaps exceed 1 ms (about 2.3 ms each). 3(b) launches the n_max steps back to back
with one readback, so draft→draft is what it removes; the other two transitions keep host work and are not counted.
Graph tracing ran at 141.8 tok/s against the untraced tip's 154.7 (decode is flat between the two libraries): charging that whole 8.3 % to the gaps leaves
draft→draft at no less than 0.73 ms of a 13.63 ms round, **5.3 %**, above the plan's 4 % rule. **3(b) is built.**
Two readings were void and are not evidence: `nsys stats cuda_gpu_trace` lists no graph executions, so its "1.6 %
busy" read only the kernels outside graphs; and `--cuda-graph-trace=node` costs 28 % (110.6 against 154.7 tok/s), so
its 27–37 % gaps are the instrument's.

**The pin moves to 3d40ae99c** (head.toml has its lineage). Chunked KDA ships on under the practical-equivalence gate
above; the hc post kernel and lever 2 are bit for bit on the real pack.

### Rental 2a: the v1 cache, the slot count, and a stall the served shape never reaches (rig-glm, Oct 2-3, 2026)

Box 53941474, two RTX PRO 6000 Blackwell Workstation (97,887 MiB, sm_120), the real IQ3_XXS pack (3 of 3 shards
sha256-checked on the box), engine 737ee4450 (libggml-cuda 31056d48), the served shape `-sm tensor -ts 1,1 -fa on
-b 4096 -ub 1024`, `GGML_CUDA_ALLREDUCE=internal`, `GGML_CUDA_GRAPH_MAX=0`. Receipts in rig-glm's research notes
(`local/research/glm53-mmq-deep-2026-10-02/notes.md`, legs a to d).

**The cache: q8_0 K, V and indexer.** Against the f16 cache on the same library, so the cache type is the only
difference; the floor is that f16 cache's own `-ub 512` reordering at the same depth; the gate is the
practical-equivalence one (KLD at most 1.10x the floor, same top within 0.25 points).

| depth | f16 floor (KLD / same top) | q8_0 K/V/idx | ratio | same top |
|---|---|---|---|---|
| 2k (32 chunks) | 0.012685 / 96.209 % | 0.011520 / 96.453 % | 0.908 | +0.244 |
| 16k (4 chunks) | 0.012742 / 96.749 % | 0.007469 / 97.354 % | 0.586 | +0.605 |

Decode, f16 against q8_0, `rig engine ab`, 3 pairs of 3 reps: **-0.68 % at d65536** (95 % CI -1.19 % to -0.16 %) and
**-0.85 % at d262144** (95 % CI -2.86 % to +1.16 %, inside noise).

**Three slots in one shared pool.** rig renders `-np 3 -c 524288 --kv-unified`: under `--kv-unified` every sequence's
context is the whole pool and the compute buffer scales with it, so a pool of `slots x context.model` does not load
(`-np 3 -c 1572864`: an 18,221 MiB compute-buffer cudaMalloc fails on card 0; `-np 2 -c 1048576`: 9.6 GiB fails),
and the one 524,288-cell pool shared by three does. Loaded with four sessions decoding at once, worst free is
14,547 MiB at three slots and 14,305 MiB at four. One session alone on the three-slot server, the same 12 short
prompts as the one-slot baseline: **152.25 tok/s** (min 132.48, acceptance 0.6256) against 142.18 (acceptance 0.626),
1.07x, unpaired and 3 hours apart on a box that swings; superseded by the paired 1.008 on the freeze, below. Three
concurrent sessions of ~20K-token prompts each decoding up to 2,048 tokens: 92 s wall, 32.1 tok/s a
session, acceptance 0.611. Decode-only MTP lost acceptance even on short prompts (0.905x short, 0.788x at a 99K-token
turn) and the windowed MTP draft is refused at load for this model's recurrent-plus-indexer memory, so neither ships.

**A host deadlock at 9-16 token ubatches past ~8k of depth, which the served shape does not reach.**
`llama-perplexity -b 9 -ub 9 -c 16384` never finishes on this engine: one `ggml_cuda_ar_kernel` spins on card 0 for
a peer that never arrives, card 1 has no kernel, and the host sits inside card 0's `cudaGraphLaunch` from the meta
backend's capture replay, so it never launches card 1's graph. It needs the internal all-reduce AND the capture
replay together (`GGML_META_CAPTURE_LEGACY=1`, `GGML_CUDA_ALLREDUCE=none`, `=nccl` and `-sm layer` each pass), at 9
or more tokens a ubatch, past 2k and by 8k of depth, on the 46-layer model (the 5-layer proxy passes); the cache
type, the LL kernel and the L2 issuer are each ruled out. Served on rig's own argv it does not occur: three
concurrent ~20K-token sessions (9-token MTP verify batches every step), and one prompt of 17 x 1024 + 9, 12 and 16
tokens sent three times each (the identical tail graph captured and replayed), all complete. The working reading is
host run-ahead: perplexity queues ~900 replays with no host sync, a server syncs every step. The fix is the first
engine lever after the v1 freeze; v1 ships without a mitigation.

**The freeze engine adds MMQ's L2 prefetch (0020175b4, Oct 3, 12:28-1:16 AM CT).** While a chunk of an IQ3_XXS or Q8_0
weight is computed, MMQ asks L2 for the next chunk's rows; only the issue time of loads moves. Same box and shape,
against 737ee4450's library:

- `test-backend-ops` with the prefetch on, CUDA0 against the CPU backend: 170 of 170 IQ3_XXS and Q8_0 cases pass
  (MUL_MAT 11 and 47, MUL_MAT_ID 37 and 75).
- Logits at 2k x 8 chunks, against 737ee4450's base: the prefetch on, off, and as committed with no env each read
  KLD 0, same top 99.988 %, PPL ratio 1.000439. That is exactly what 737ee4450 reads against its own base, and what a
  clean rebuild of 737ee4450 in the same tree reads, so the 99.988 % is the instrument at that shape, not a library
  difference.
- Prefill, `rig engine ab`, 3 pairs of 3 reps, off against on: **+6.15 % at pp512** (95 % CI +5.94 % to +6.36 %) and
  **+3.52 % at pp4096** (+2.90 % to +4.14 %). Decode tg128 +0.05 % (-0.60 % to +0.70 %), inside noise.
- As committed, no env against `GGML_CUDA_MMQ_L2_PF=0`, 3 pairs: pp512 **+5.81 %** (95 % CI +5.00 % to +6.62 %), so the
  prefetch is on when unset. The freeze is 009073391, which is 0020175b4 and its TORAD.md row.

**Served on the freeze library, on a box that swings (Oct 3, 1:18-1:45 AM CT).** rig's rendered argv, one session at a
time, the 12 short requests. Every run reads acceptance 0.6256, so the token streams are the same. The box's own speed
moved between two levels with nothing changed: 737ee4450 read 152.25 tok/s at 11:44 PM CT and 121.1 and 121.27 at
1:21-1:27 AM CT, and the freeze library read 138.67 (it changed level mid-run), about 120, then about 152. Within a run,
each request sits at its run's level from the first request. Only runs back to back compare, so these are paired.

| order | library, env | slots | tok/s |
|---|---|---|---|
| 1 | 737ee4450 | 3 | 121.10 |
| 2 | freeze, `GGML_CUDA_MMQ_L2_PF=0` | 3 | 121.05 |
| 3 | freeze, default | 3 | 119.82 |
| 4 | 737ee4450 | 3 | 121.27 |
| 5 | freeze, default | 3 | 119.88 |
| 6 | freeze, `GGML_CUDA_MMQ_L2_PF=0` | 3 | 121.48 |
| 7 | freeze, `GGML_CUDA_MMQ_L2_PF=0` | 3 | 122.56 |
| 8 | freeze, default | 3 | 152.63 |
| 9 | freeze, default | 1 | 150.95 |
| 10 | freeze, default | 3 | 152.82 |
| 11 | freeze, default | 3 | 152.77 |
| 12 | freeze, default | 1 | 152.08 |

- **The prefetch, default against PF=0 on one library, 3 pairs in ABBA order:** -1.02 % and -1.32 % at the low level,
  then +24.5 % where the box moved up between the two runs. The first two pairs sit at one level, with 737ee4450 between
  them reading like PF=0, and put the default about 1 % under PF=0 in served decode. MMQ is not on the 3-token verify path
  (`ggml-cuda.cu:2772` dense, `:2881` MUL_MAT_ID: mmvq takes up to 8 tokens), so the mechanism is unknown. Under
  rig-orchestrator's rule (only a paired 95 % CI wholly below zero turns it off) the prefetch stays on. The first
  post-freeze measurement is a paired served A/B on a box that holds one level.
- **Slots, np3 one session at a time against np1, ABBA, all at the high level:** 1.012 and 1.005, so 1.008
  (np3 152.82 and 152.77 against np1 150.95 and 152.08). The 1.07 above compared
  runs 3 hours apart across this swing and is superseded. The bar is 0.95, and slots = 3 holds.
