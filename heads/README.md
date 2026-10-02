# heads/

One directory per head. Everything rig knows about a model is data here; nothing in `packages/` is
model-specific, and a head that needs a code path is a bug in rig.

```
<name>/head.toml     the identity: source pack (repo, rev and each file with its sha256 and bytes: a split
                     GGUF lists its shards, the first the one loaded), the optional [derive] steps with the
                     [served] (and [public]) pack they produce, [speculative] draft head, context, cache
                     formats, the measured [[profiles]] (cards, VRAM each, split, slots, context), the
                     optional [sizing] memory model, runtime args, client facts
<name>/gates.toml    the probes that gate this head and what counts, run by `rig gate <name>`
<name>/evidence.md   where every number in the two toml files came from, citing runs
<name>/assets/       what the runtime args reference (see assets/README.md)
<name>/evidence/     the result files evidence.md cites (see evidence/README.md)
```

**Belongs here:** a head's data and its provenance. `loadHead` checks the invariants (a
multi-card profile names its split, every profile is reachable, the single-card profiles fit
their VRAM by [sizing] when the head declares it, advertise ≤ model, the draft head charged to
every profile that loads it) and refuses a head that fails them.

**Does not belong here:** weights and packs (they live under `local/packs/<name>/`, fetched by
sha256), logs, a build, a script, an argument the server is loaded with that is not in
`[runtime]`, a number without a line in `evidence.md` saying where it was measured.

Adding a head is adding a directory: `docs/architecture.md`, "Adding a head".
