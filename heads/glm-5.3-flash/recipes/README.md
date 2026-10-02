# Quantization recipes for GLM-5.3 Flash

**These are version-controlled because the last one was not, and a flag went missing.**

`tier-A.txt` describes the model measured at **+15.94 %** (95 % interval +14.88 to +17.00, n=10 paired) against the
Q8_0/IQ3_XXS baseline, at **-25.2 %** of per-token reads. Its earlier copy lived only under `local/`, which is
`.gitignore`d, and omitted `output.weight` — the logits head — because that tensor's type had been set by a
`--output-tensor-type q6_k` flag on an unrecorded command line. The recipe therefore described a 1.57 % larger, *more
conservative* model than the one whose speed is shipped, and a quality run against it would have come back cleaner than
reality while never testing the tensor that decides token selection.

## Use

    llama-quantize --allow-requantize --tensor-type-file recipes/tier-A.txt <base.gguf> <out.gguf> q8_0

Engine `22d5415f0` or later is required: without it, requantizing an IQ-family model fails while `--dry-run` passes on
the same arguments.

**The acceptance check for any change here is byte-exact reproduction**, not a plausible-looking file. `tier-A.txt`
against the L44-E8 proxy must produce exactly **9762864448 bytes**. A recipe that cannot rebuild its own artifact
byte-for-byte is a note, not a recipe.

## Rules

| recipe | per-token read | vs baseline | changes from A | quality risk |
|---|---|---|---|---|
| `tier-A.txt` | 9.079 GB | **-25.2 %, +15.94 % measured** | — | the Q6_K logits head is the exposure; unvalidated on real weights |
| `tier-C.txt` | ~8.94 GB | -26 % | MLA `attn_k_b`/`attn_v_b`/`attn_kv_a_mqa` -> Q6_K | LOW: small absorbed latents, no recurrence |
| `tier-B.txt` | ~8.23 GB | -32 % | `attn_k`, `attn_v`, `ffn_down*` Q6_K -> IQ4_XS | HIGHER: these write the KDA recurrent state |
| `tier-D.txt` | ~8.37 GB | -31 % | experts IQ3_XXS -> IQ2_S | HIGHEST: 28 % of the read at 2.5 bpw |
| `tier-E.txt` | ~7.56 GB | -38 % | B + C + D | only if each passes alone |

Sizes other than A's are modelled from bits-per-weight and run about **1 % optimistic**; A's is measured.

## Two traps, both verified

1. **`^output\.weight` must stay anchored.** Matching is `std::regex_search`, so unanchored it also matches
   `attn_output.weight` as a substring, and first-match-wins would hand every attention output projection the logits
   head's type based on line order alone.
2. **`token_embd.weight` stays Q8_0.** It is a row gather, not a per-token read: quantizing it costs quality and buys no
   bandwidth. Every byte figure for this model excludes it for the same reason.

## The MTP block's experts are NOT iq3_xxs in the served pack (found by rig-glm on box 53728146)

The first three lines of every recipe pin **`blk.45.ffn_{down,gate,up}_exps`** to `q8_0`, and they must stay first.

The served pack's experts are **iq3_xxs in 42 layers and Q8_0 in `blk.45`, the NextN/MTP block** — read from the shards'
GGUF headers, 42 + 1 for each of down/gate/up. A bare `ffn_*_exps=iq3_xxs` therefore asks for a Q8_0 -> IQ3_XXS requant
there, which `llama-quantize` refuses outright without an imatrix (`offending tensor: blk.45.ffn_down_exps.weight`).
**With an imatrix it does not refuse — it silently requantizes the draft head's experts**, degrading speculative decoding
invisibly while every byte and perplexity number still looks right.

**The L44-E8 proxy cannot show this.** Its highest block index is 44; it has no `blk.45` and no MTP expert block, so the
rule is a no-op there and matches nothing. That is a real limit on what the proxy can validate: **a recipe exercised only
against the proxy is untested against the served pack's heterogeneous expert types**, because the proxy's tensor set is
not a superset of the pack's. Recipes get their reproduction check on the proxy and their refusal check on the pack.

Ordering verified: with these three lines first, `blk.45.ffn_down_exps.weight` -> `q8_0` while `blk.3` and `blk.44` take
the recipe's expert type (`iq3_xxs` in A, `iq2_s` in D).

## Gate

Quality is measured with position-binned dKLD (`LLAMA_KLD_FIRST`, engine `a61c58f5c`), binned **0-512 / 512-4k / 4k+**.
An unbinned run cannot see KDA error accumulate, because 512-token chunks restart from cleared recurrent state. The
expected signature of a bad tier here is a clean 0-512 bin with a degraded 4k+ bin, which a single mean would pass.
