# scripts/

What a developer of rig runs and a user of rig never sees. Nothing here ships in `dist/rig`.

**Belongs here:** `tools/` — scripts run from the checkout under `bun`: the layer lint behind
`bun run lint` (`tools/lint-orthogonality.ts`) and the calibration-corpus recipe
(`tools/engine-corpus.ts`, which writes to the layout's `local/calibration/`). A script imports rig's own modules through `../src/`, resolves the
repo root as `resolve(import.meta.dir, "..")`, and is covered by `bunx tsc --noEmit` through
tsconfig's `scripts` include. `build-prebuilt.sh` makes the prebuilt engine a release publishes: `rig
build --portable` inside `prebuilt/Dockerfile`'s image (CUDA 13.3 on Ubuntu 22.04, pinned by
digest), so the tarball's floor is glibc 2.35 whatever machine builds it. `e2e-driver-only.sh` is
the gate it passes before it is published: this checkout packed and installed in a fresh
ubuntu:22.04 container (`--base` another image) that has one card and only the driver, then `rig
prepare`, `rig build` and a decode. Both need docker with the NVIDIA CDI devices.

**Does not belong here:** anything a slice or the binary imports (that is `src/`); a probe or a
measurement the gates run (that is `src/features/head-gating/probes/`); an experiment and its output
(the project wrapper's `research/`); a bash script that replaces a rig command; documents
(`docs/`).

A tool that a user would need is a `<verb>.command.ts` in its feature, not a tool here.
