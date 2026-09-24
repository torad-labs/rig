# rig

Builds [torad-labs/llama.cpp](https://github.com/torad-labs/llama.cpp) per GPU and brings model
heads up, gated and measured — the local runtime behind an OpenAI-compatible provider row. One
Bun program, one binary; the model is data under `heads/<name>/`, the engine is a pinned commit
under `engine/`, everything generated lives under `local/`.

On a Linux machine (glibc 2.35 or newer: Ubuntu 22.04, Debian 12, Fedora 36 and later) with an
NVIDIA card and its driver:

```
curl -fsSL https://github.com/torad-labs/rig/releases/latest/download/install.sh | sh
rig up bonsai-2-27b
```

The installer puts the release in `~/.local/share/rig` and links `~/.local/bin/rig`. On a card
`engine/engine.toml` publishes a build for (sm_120: RTX 5080, 5070 Ti, 5090, RTX PRO 6000), `rig
build` installs that build and NVIDIA's CUDA runtime, each checked by sha256, so no toolkit or
compiler is needed; any other measured card compiles the engine (git, cmake, ninja, nvcc). `rig
up bonsai-2-27b` then serves ProCreations' published pack with our retrained MTP draft head spliced
in (`heads/bonsai-2-27b/evidence.md`, "Draft head retrained"), fetched from this repo's releases
and checked by sha256 like everything else.

From a checkout:

```
bun install && bun run build          # → dist/rig
rig up bonsai-2-27b                   # prepare → fetch → build → derive → unit → start
rig gate bonsai-2-27b                 # the head's probes on the gate card; evidence in local/gate-runs/
rig describe bonsai-2-27b             # what the head is, as JSON, for whatever sits in front of it
rig vast up bonsai-2-27b --gpu H100_SXM   # the same head on a rented card, through an ssh tunnel
rig help
```

`docs/architecture.md` is the map; `heads/bonsai-2-27b/evidence.md` is where the numbers come
from. `bun test` runs the suite over in-memory ports (no card, no model); `bun run lint` holds
the format (Biome, 100 columns), the dependency rule and the types; `bun run format` rewrites.
