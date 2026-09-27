# engine/

The engine as data: one pin, two readers.

- `engine.toml` — the fork, the commit every build is made from, the base it sits on, and the
  compute capabilities with a measured run behind them (`[[archs]]`). `prepare` and `build`
  refuse any other card unless `--allow-arch`.
- `llama.cpp/` — the fork as a submodule at the same commit; what a developer edits.
  `src/shared/engine/engine.test.ts` holds the gitlink and the toml equal.

**Belongs here:** the pin and the evidence for each supported card. Moving the pin: change the
submodule, change `engine.toml`, say in its comment which commit changed what a head serves.

**Does not belong here:** a build (`local/engine-builds/<sha7>-sm<cap>/`, published by `rig build`), a
build tree (`local/engine-build-trees/`), a patch to the fork (that is a commit in torad-labs/llama.cpp and a
new pin), a per-card argument (that is a head's tier in `heads/<head>/head.toml`), a second
engine until the head contract names one.
