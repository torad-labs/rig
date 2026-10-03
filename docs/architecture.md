# rig — architecture

rig builds [torad-labs/llama.cpp](https://github.com/torad-labs/llama.cpp) per GPU and brings
model heads up, gated and measured. It is a Bun program compiled to one binary; the model is
data (`heads/<name>/`), the engine is a pinned commit (`engine/`), and everything generated
lives under `local/`. Kernels and server features live in the fork; how a build is produced, how
a pack is fetched, derived, gated and served lives here. Nothing in `apps/` or `packages/` is model-specific:
adding a head is adding a directory, and a head that needs a code path is a bug in rig.

## Workspaces

rig is a platform of Bun workspaces: two apps over nine domain packages, two support packages and
the tools. Each package is one bounded context with one public API (`src/index.ts`, the only
entry its `package.json` exports) and one line in its `package.json` saying what it owns.

```
apps/
  cli/            @rig/cli       the rig binary: src/main.ts, the composition root (the one place that
                                 constructs the machine and wires every package), src/commands/<verb>.command.ts
                                 (flags in, a service called, an exit code out), src/cli/ (the argument grammar
                                 and the command contract). Its package.json carries the product version.
  registry/       @rig/registry  the pull-only OCI registry Worker over R2 (wrangler.toml beside it)
packages/
  core/           @rig/core      outcomes and exit codes, the layout, sha-pinned artifacts, card sets,
                                 downloads, and ports/ — every capability through which rig touches a machine
  engine/         @rig/engine    the engine as data (the pin, prebuilts, the CUDA runtime, the corpus) and
                                 build/ (building it for a card)
  head/           @rig/head      a head as data: head.toml's schema and invariants, profiles and their
                                 placement on cards, cache formats, the draft head, a serving head over http
  machine/        @rig/machine   whether this machine runs the engine and a head (prepare, room)
  pack/           @rig/pack      the weights: fetch/ (by sha256) and derive/ (the served pack)
  serve/          @rig/serve     a head on this machine: serving/ (plan, verify, serve), unit/ (systemd),
                                 describe/ (the JSON a proxy reads), up/ (bring-up)
  gate/           @rig/gate      the head's numbers re-proven: gate runs, the gate server, probes/
  image/          @rig/image     the head as a container image: build, the public rules, the OCI push
  rental/         @rig/rental    a rented card as a head: rent, template, guard
  adapters/       @rig/adapters  the real ports on Linux under Bun, one file per tool
  testing/        @rig/testing   the in-memory machine: a fake for every port, shared fixtures, repoRoot
tools/            @rig/tools     what a developer runs from the checkout: the architecture lint, the public
                                 export, the prebuilt build and its gate, benches and generators
heads/<name>/     head.toml, gates.toml, assets/, evidence.md, evidence/ — a head is data
engine/           engine.toml (the pin, data) and llama.cpp (the submodule at the pin; a test holds them equal)
local/            what rig fetched or built here: packs, engine builds, logs, gate runs, images, the rented
                  box — gitignored, never /tmp
dist/rig          the compiled binary the unit, the image and the rented boxes run (bun run build)
docs/             this map
```

A test lives beside the code it tests (`*.test.ts` in the package's `src/`); `bun test` at the
root runs every workspace's (`bunfig.toml` keeps `engine/`, `local/` and `dist/` out).

## The dependency rule

Structure holds it first. `bunfig.toml` installs with the isolated linker, so a workspace resolves
only the packages its `package.json` declares, and each package exports `src/index.ts` alone, so
nothing reaches another's internals. `tools/lint-architecture.ts` (`bun run lint`, and its own
test) holds the rest:

| workspace | may depend on, at runtime |
|---|---|
| core | nothing |
| engine | core |
| head | core, engine |
| machine | core, engine, head |
| pack | core, head |
| serve · gate · image | core, engine, head |
| rental | core, engine, head, image |
| adapters | core |
| testing | core, engine, head, image, registry |
| cli | every package but testing |
| tools | adapters, core, engine, head, image |

- a workspace with no row fails by name: a new package is a decision someone makes;
- a test leans on `@rig/testing` and `@rig/adapters` as devDependencies; code that ships never
  imports a devDependency;
- no relative import leaves its workspace;
- in the cli, only `main.ts` constructs the machine (`@rig/adapters`).

When a package needs another's use case at runtime (`up` needs the steps, `unit` needs a serve
plan, `describe` needs the unit's status, `rental` needs the live gates), it declares the interface
it needs in its own file and `apps/cli/src/main.ts` satisfies it with the other package's service.
Every effect on the machine goes through a port (`packages/core/src/ports/`); ports name
capabilities, not tools, and the adapters behind them are chosen in one place.

## Commands

| command | package | what it decides |
|---|---|---|
| `prepare` | machine | tools, card, driver — before any download; exit 1 / 3 / 4. A card with a published prebuilt needs only the driver, curl, tar and xz, on a glibc at least the build's floor (`glibc` in `[[prebuilt]]`; below it the card compiles, and `noPrebuilt` says why); so does a card whose build is already installed with the pin's CUDA runtime beside it (`built`: a tarball's install, as an image bakes it). A root box runs apt only for packages dpkg does not already hold |
| `build` | engine (build/) | the engine at its pin for this card → `local/engine-builds/<sha7>-sm<cap>/`, published in one rename with a marker written last: the pin's published build plus NVIDIA's CUDA runtime where `engine.toml` pins one for the card (`[[prebuilt]]`, `[cuda]`, each by sha256), else compiled; `--compile`; `--portable` tarball; `--from-tarball`, and with `--off-pin NAME` a tarball of another commit → `local/engine-lab/NAME/` with the pin's CUDA runtime beside it, for the lab to A/B against the pin; `--prebuilt [--sha FORK_SHA]` builds the release's prebuilt of the pin (or of `--sha`'s commit, through a copy of `engine.toml` that pins none) with `--portable` in `tools/prebuilt`'s image, with no card (`--gpu` names the card whose compute capability it targets; a compile never holds one), capped at 14 GiB and 6 CPUs → `local/prebuilt/engine-builds/` |
| `e2e` | engine (release/) | the driver-only gate: this checkout packed as `release.yml` packs it and installed by `install.sh` in a fresh container with one card and only the driver, then `prepare` must find a prebuilt and no toolkit, `build` must install it, and `llama-bench` must decode the pack on the card; `--prebuilt TARBALL` gates a local build `engine.toml` pins by name before it is published, `--head NAME` also adopts and derives the head as a machine without private assets would; `RIG_GATE_LOCK` serializes the decode on a shared card; a pass without `--prebuilt`, in a checkout with no change git sees, writes the commit's receipt under `local/release/` with its log, and such a run that fails removes it |
| `tag` | engine (release/) | the release tag `v<version>` (`apps/cli/package.json`, as `release.yml` checks) on this checkout's HEAD, pushed: refused unless HEAD is `origin/main` with no change git sees and the commit has a receipt from `e2e` (a machine with only the driver installed the published pin), so no release ships a CLI that cannot install its own pin |
| `fetch` | pack (fetch/) | the source pack by sha256 (adopted by hard link, or aria2c/curl to a `.part` sibling), and every public `[[derive]]` asset (a step with a `url`) into `local/packs/<head>/` the same way |
| `derive` | pack (derive/) | the served pack from the source pack — the head's `[[derive]]` steps as data; a machine without a private asset derives the `[public]` pack (the public steps alone) instead, or serves the source pack when there is none; published only at the pinned sha, a different edit is removed |
| `verify` / `serve` | serve (serving/) | complete build, pinned pack, assets present; the profile the cards hold, its `-np`/`-c` and split, `--cache-ram` from RAM/4; the argv in one order (golden test = the live head's cmdline) |
| `unit` | serve (unit/) | the systemd user unit: `ExecStartPre=rig verify`, `ExecStart=<llama-server argv>`, the host's `--cache-ram` kept across re-renders, dated backup of a changed unit; linger turned on for the user (`loginctl enable-linger`) so the head outlives logout, named with the command where refused |
| `describe` | serve (describe/) | one JSON object for whatever sits in front of the head (a proxy's wizard) |
| `up` | serve (up/) | prepare → fetch → build → derive → unit → start; a serving head is left running unless `--restart`, and a restart is refused (exit 2) while a slot is processing; `--foreground` is every step but the unit, then the server in this process (a container) |
| `gate` | gate | the head's probes (`gates.toml`) on the gate card, one fresh server per leg; evidence under `local/gate-runs/<head>/<run>/`; `--live` adds the probes against the running head |
| `image` | image | the head as a container image with no source in it: the CLI compiled here from the committed tree (`git archive HEAD`) and its bytes held to `public-export.toml`'s deny patterns, the head and the pin through the same rules (`packages/image/src/public-rules.ts`, the export's own) and gitleaks, the head's private `[derive]` assets never, the engine for the card's sm installed by `rig build` from the pin's tarball on Ubuntu 22.04; proven on that card (prepare finds it `built`, build finds it built, it decodes on CUDA) before `--push` writes it into `registry.toml`'s R2 bucket (`oci-push.ts`: gzipped layers, a schema-2 manifest, the tag last, read back with the pull key), where `apps/registry/src/worker.ts` serves pulls behind that key; `CMD` is `rig up <head> --foreground`; recorded in `local/images/<head>-sm<cap>/image.json` |
| `engine` | engine (lab/) | an engine change measured on a developer's cmake tree (`--tree`, a name under `local/engine-build-trees/`), its evidence under `local/engine-lab/<run>/`: `test` (test-backend-ops per card, by default and under `--legacy`'s switches), `ident` (the KLD base file byte for byte against `--ref`: a change that should move no value proves it there), `kld` (the KL divergence from `--base`), `ab` (llama-bench in back-to-back pairs of two libraries or switch sets, the order alternating, the change's 95 % interval on t), `profile` (llama-bench under nsys, each kernel billed its own time: under PDL nsys's duration includes the wait for the kernel before), `relink` (the tree's libggml-cuda with `--head`'s paths as HEAD has them, in a checkout where a peer's uncommitted edit would otherwise land in the build; each mirrored object's dependency file proves it). A run holds its cards once, through the `CardLease` port (gpu-lease where installed): rig runs itself again inside the lease, so an A/B's pairs are never split by another job |
| `vast` | rental | a rented card as a head: rent, ship rig + head + pin, bring up with rig's own steps on the box, tunnel here, idle timer; `lab` rents a card with no head, ships rig + pin only, and arms the idle timer, whose check then reads the card alone; `down` re-reads the listing; `bench` runs the gates there; `template` makes the head's pushed image a private vast template (the registry login in the request body, never an argument; disk from the head's pins; offers from its first profile and the runtime's CUDA; on-start `rig up --foreground` beside `rig vast guard`), created once and edited in place; `guard`, on such a box, stops it through vast's API with the box's own key after an hour with nothing served, computed or fetched |

## Invariants the code holds

- **A path is trusted only after its hash matched.** Every pack is an artifact with a sha256;
  "the file exists" is never enough (`packages/core/src/artifact.ts`).
- **Nothing writes a served path in place.** Writes go to a sibling and are renamed over: a
  running llama-server has the file mmapped, and an in-place write is a SIGBUS in the live head.
- **A build directory either is complete or does not exist.** The marker is written last and the
  directory is published by one rename.
- **The derive step is reproducible to the byte.** The PQ2_0 lattice ablation is a transcription
  of the numpy reference down to arithmetic order; `RIG_REAL_PACK=<source> bun test` re-proves it
  on the real pack (sha `e7b99670…`, 74 s).
- **Gates never touch a serving card.** `gates.toml` declares the gate card; the use case also
  refuses any card the process holding `:<port>` has memory on, so a rented single-card box can
  gate with its server stopped. The legs run on one card: a head whose every profile spans
  several is refused.
- **A head names no card.** `--gpu` takes an index, distinct indices (`0,1`) or `auto`; the head
  gets the first of its profiles the named cards hold (that many of one compute capability, each
  with its VRAM free), and a profile over several renders its split (`-sm`, an even `-ts`).
  Without `--gpu`, serve and a first install take `auto`; the unit records the cards it resolved
  and a re-render (and `up`, and `describe`) keeps them.
- **A prebuilt is the pin's own build.** Its file is named for the pinned commit and card
  (`engine-sm<cap>-<sha7>.tar.gz`) and `loadEngine` refuses an entry that names another commit,
  so a pin move without a republished build compiles instead of serving the old kernels.
- **The engine has one pin, two readers.** `engine.toml` and the submodule gitlink must agree
  (`packages/engine/src/engine.test.ts`); the build reads the toml, a developer edits the submodule.
  A head the pin does not run yet names its own fork commit in `[engine]` (and the builds
  published for it); every command naming that head runs on it (`apps/cli/src/main.ts`, `headEngine`),
  its builds under their own `<sha7>-sm<cap>`, the cards, caches, compilers and CUDA runtime
  still `engine.toml`'s.
- **No credential leaves this machine.** A rented box binds loopback and is reached over ssh;
  the vast API key stays local.

## Exit codes

`0` ok · `1` failure (named) · `2` busy (a slot is processing) · `3` unsupported card ·
`4` driver older than the toolkit (or than the prebuilt's CUDA runtime) · `64` usage.

## Adding a head

1. `heads/<name>/head.toml`: source (HF repo, rev, and `[[source.files]]` each with its sha256 and
   bytes, the shards of a split GGUF in order), optional `[[derive]]` steps (public ones, with a
   `url`, first) with `[served]` pinning what they produce and `[public]` what the public steps
   produce alone, context, cache formats, `[[profiles]]` (the cards, the VRAM each, the split
   over several, slots, context: measured), optional `[sizing]` (the memory model the single-card
   profiles are checked against), runtime args, client facts. `loadHead` checks the invariants
   (every profile reachable, a split named over several cards, [sizing] fits, advertise ≤ model…).
2. `heads/<name>/assets/`: whatever the runtime args reference (`assets/…` resolves to the head).
3. `heads/<name>/gates.toml`: the probes and their criteria.
4. `rig up <name>`; then `rig gate <name>` on the gate card and `evidence.md` citing the run.

## Adding a command or a package

- A command over an existing package: its use case in that package (a service that decides and
  returns a Result, never prints), exported from the package's `src/index.ts`; its
  `apps/cli/src/commands/<verb>.command.ts` (a `const USAGE`, flags in, the service called, an exit
  code out); one line in `apps/cli/src/main.ts`; a row in the table above.
- A new package: `packages/<name>/` with `package.json` (`@rig/<name>`, a one-line description, the
  `@rig/*` packages it uses as `workspace:*`, `valibot` as `catalog:`), `src/index.ts` as its
  public API, tests beside the code; a row in `tools/lint-architecture.ts`'s `LAYERS` and in the
  dependency table above. `bun install` links it.
- A new machine capability: a port in `packages/core/src/ports/`, its adapter in
  `packages/adapters/src/` (named for the tool), its fake in `packages/testing/src/fakes.ts`, wired
  in `real-ports.ts`.
