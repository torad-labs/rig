# heads/bonsai-2-27b/assets/

The files this head's runtime args reference; `assets/…` in `head.toml` resolves against the
head's directory, and `rig verify` refuses to start a server while any of them is missing.

- `chat-template.jinja` — the model's template at the pinned revision with one line changed
  (evidence.md, "Runtime args"), the reasoning block routed to `reasoning_content`.
- `kv-mean-center-PQ2_0.gguf` — the calibrated K-cache mean-centering bias (16 vectors, no
  text), calibrated on the engine corpus rig builds itself; the recipe and its KL are in
  evidence.md.
- `lora/bonsai-abliterate-lora.gguf` — the rank-1 refusal adapter the `[derive]` step bakes
  into the served pack; pinned by `lora_sha256`. A private asset, not in git: `torad model pull
  bonsai-2-27b-derive` fetches it on Torad's machines. A machine with neither it nor the derived
  pack serves the source pack (`rig describe` names why, in `undrived`).

**Belongs here:** a small, load-bearing file the server or the derive step reads, pinned by
its sha256 in `head.toml` where the loader checks one, and explained in `evidence.md`.

**Does not belong here:** weights and packs (`*.gguf` is gitignored except the allowlisted
pattern in `/.gitignore`), a converted or intermediate copy of an asset, a corpus (the gates
build theirs from the engine tree), anything nothing in `head.toml` or `gates.toml` names.
