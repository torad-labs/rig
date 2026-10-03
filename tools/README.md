# tools/

What a developer of rig runs from the checkout and a user of rig never sees: the `@rig/tools`
workspace. Nothing here ships in `dist/rig`.

**Belongs here:** scripts run under `bun` from the checkout: the architecture lint behind `bun run
lint` (`lint-architecture.ts`, with its test), the public export (`public-export.ts`), the
calibration-corpus recipe (`engine-corpus.ts`, which writes to the layout's `local/calibration/`),
the flag-alias generator a pin move re-runs (`rendered-flag-aliases.ts`), and the bench and probe a
head's evidence cites (`bench-head.ts`, `glm53-verify.ts`). A tool imports rig's packages by name
(`@rig/core`, `@rig/head`, …, declared in `package.json`) and resolves the repo root as
`resolve(import.meta.dir, "..")`. `prebuilt/Dockerfile` is the image `rig build --prebuilt` makes
the prebuilt engine a release publishes in: `rig build --portable` inside it (CUDA 13.3 on Ubuntu
22.04, pinned by digest, with no card: the build never uses one), so the tarball's floor is glibc 2.35 whatever machine builds it. `rig e2e`
is the gate it passes before it is published: this checkout packed and installed in a fresh
ubuntu:22.04 container (`--base` another image) that has one card and only the driver, then `rig
prepare`, `rig build` and a decode, and with `--head` the head's `rig fetch` and `rig derive` as a
machine without Torad's private assets runs them. Both need docker with the NVIDIA CDI devices.

**Does not belong here:** anything a package or the binary imports (that is `packages/`); a probe
or a measurement the gates run (that is `packages/gate/src/probes/`); an experiment and its output
(the project wrapper's `research/`); a bash script that replaces a rig command; documents (`docs/`).

A tool that a user would need is a `<verb>.command.ts` in `apps/cli/src/commands/`, not a tool here.
