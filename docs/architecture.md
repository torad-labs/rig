# rig — architecture

rig builds [torad-labs/llama.cpp](https://github.com/torad-labs/llama.cpp) per GPU and brings
model heads up, gated and measured. It is a Bun program compiled to one binary; the model is
data (`heads/<name>/`), the engine is a pinned commit (`engine/`), and everything generated
lives under `local/`. Kernels and server features live in the fork; how a build is produced, how
a pack is fetched, derived, gated and served lives here. Nothing in `src/` is model-specific:
adding a head is adding a directory, and a head that needs a code path is a bug in rig.

## Layers

```
src/main.ts                 the entry point: builds the platform and wires each feature's service into its command
src/features/<feature>/     one folder per thing rig does: index.ts (public API), <verb>.command.ts, <feature>.service.ts, tests, own models
src/shared/                 ports/ (one capability per file), platform/ (the adapters), cli/, head/, engine/, artifact, layout, result
test/fakes/            in-memory ports — every feature is tested without a card, a model or a network
heads/<name>/          head.toml, gates.toml, assets/ (template, K-cache bias, adapter), evidence.md, evidence/ (the results it cites)
engine/                engine.toml (the pin, data) and llama.cpp (the submodule at the pin; a test holds them equal)
local/                 what rig fetched or built here: packs, engine builds, logs, gate runs, the rented box — gitignored, never /tmp
dist/rig               the compiled binary the unit and the rented boxes run (bun run build)
scripts/               the layer lint behind `bun run lint` and the calibration-corpus recipe
docs/                  this map
```

The dependency rule, enforced by `scripts/lint-orthogonality.ts` (`bun run lint`):

- a feature imports `shared/` and itself; another feature only through that feature's `index.ts`; the feature graph is acyclic;
- `shared/` never imports a feature; `platform/` imports `ports/` only; only `main.ts` constructs the platform;
- **main.ts** sees everything and is the one place features are plugged together.

When a feature needs another's use case (`head-bringup` needs the steps, `systemd-unit` needs a
serve plan, `head-description` needs the unit's status, `gpu-rental` needs the live gates), it declares the interface it needs in its own
file and `main.ts` satisfies it with the other feature's service. Every effect on the machine
goes through a port (`src/shared/ports/`); ports name capabilities, not tools.

## Features

| command | feature | what it decides |
|---|---|---|
| `prepare` | machine-check | tools, card, driver — before any download; exit 1 / 3 / 4. A card with a published prebuilt needs only the driver, curl, tar and xz, on a glibc at least the build's floor (`glibc` in `[[prebuilt]]`; below it the card compiles, and `noPrebuilt` says why) |
| `build` | engine-build | the engine at its pin for this card → `local/engine-builds/<sha7>-sm<cap>/`, published in one rename with a marker written last: the pin's published build plus NVIDIA's CUDA runtime where `engine.toml` pins one for the card (`[[prebuilt]]`, `[cuda]`, each by sha256), else compiled; `--compile`; `--portable` tarball; `--from-tarball` |
| `fetch` | pack-download | the source pack by sha256 (adopted by hard link, or aria2c/curl to a `.part` sibling), and every public `[[derive]]` asset (a step with a `url`) into `local/packs/<head>/` the same way |
| `derive` | pack-derivation | the served pack from the source pack — the head's `[[derive]]` steps as data; a machine without a private asset derives the `[public]` pack (the public steps alone) instead, or serves the source pack when there is none; published only at the pinned sha, a different edit is removed |
| `verify` / `serve` | head-serving | complete build, pinned pack, assets present; `-np`/`-c` from the head's measured tiers, `--cache-ram` from RAM/4; the argv in one order (golden test = the live head's cmdline) |
| `unit` | systemd-unit | the systemd user unit: `ExecStartPre=rig verify`, `ExecStart=<llama-server argv>`, the host's `--cache-ram` kept across re-renders, dated backup of a changed unit; linger turned on for the user (`loginctl enable-linger`) so the head outlives logout, named with the command where refused |
| `describe` | head-description | one JSON object for whatever sits in front of the head (a proxy's wizard) |
| `up` | head-bringup | prepare → fetch → build → derive → unit → start; a serving head is left running unless `--restart`, and a restart is refused (exit 2) while a slot is processing |
| `gate` | head-gating | the head's probes (`gates.toml`) on the gate card, one fresh server per leg; evidence under `local/gate-runs/<head>/<run>/`; `--live` adds the probes against the running head |
| `vast` | gpu-rental | a rented card as a head: rent, ship rig + head + pin, bring up with rig's own steps on the box, tunnel here, idle timer; `down` re-reads the listing; `bench` runs the gates there |

## Invariants the code holds

- **A path is trusted only after its hash matched.** Every pack is an artifact with a sha256;
  "the file exists" is never enough (`shared/artifact.ts`).
- **Nothing writes a served path in place.** Writes go to a sibling and are renamed over: a
  running llama-server has the file mmapped, and an in-place write is a SIGBUS in the live head.
- **A build directory either is complete or does not exist.** The marker is written last and the
  directory is published by one rename.
- **The derive step is reproducible to the byte.** The PQ2_0 lattice ablation is a transcription
  of the numpy reference down to arithmetic order; `RIG_REAL_PACK=<source> bun test` re-proves it
  on the real pack (sha `e7b99670…`, 74 s).
- **Gates never touch a serving card.** `gates.toml` declares the gate card; the use case also
  refuses the head's own card while `:<port>` answers, so a rented single-card box can gate with
  its server stopped.
- **A prebuilt is the pin's own build.** Its file is named for the pinned commit and card
  (`engine-sm<cap>-<sha7>.tar.gz`) and `loadEngine` refuses an entry that names another commit,
  so a pin move without a republished build compiles instead of serving the old kernels.
- **The engine has one pin, two readers.** `engine.toml` and the submodule gitlink must agree
  (`shared/engine/engine.test.ts`); the build reads the toml, a developer edits the submodule.
- **No credential leaves this machine.** A rented box binds loopback and is reached over ssh;
  the vast API key stays local.

## Exit codes

`0` ok · `1` failure (named) · `2` busy (a slot is processing) · `3` unsupported card ·
`4` driver older than the toolkit (or than the prebuilt's CUDA runtime) · `64` usage.

## Adding a head

1. `heads/<name>/head.toml`: source (HF repo, rev, file, sha256), served (file, sha256), optional
   `[[derive]]` steps (public ones, with a `url`, first; `[public]` pins what they produce alone),
   context, geometry (constants + measured tiers), runtime args, client facts.
   `loadHead` checks the invariants (tiers fit their VRAM by the constants, advertise ≤ model…).
2. `heads/<name>/assets/`: whatever the runtime args reference (`assets/…` resolves to the head).
3. `heads/<name>/gates.toml`: the probes and their criteria.
4. `rig up <name>`; then `rig gate <name>` on the gate card and `evidence.md` citing the run.
