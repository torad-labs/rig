# heads/bonsai-2-27b/evidence/

The result files `evidence.md` cites, so a clone resolves every citation instead of pointing at
a gitignored `local/` or another machine.

```
gates/<run>/         a `rig gate` run's per-probe results and summary.json (local/gate-runs/<head>/<run>/):
                     20260920T082930Z the six gate-card probes, 20260920T085039Z the two --live probes,
                     20260921T014813Z the MTP pack (speculative FAILED byte-identity), 20260921T021949Z the near-tie probe
spec-bench-local/    the draft-head decode measurements on the 5070 Ti (results.jsonl)
spec-bench-4x5090/   the same on a rented 4×5090 box (results*.jsonl)
spec-bench-mtp-5070ti/ the in-pack MTP head against plain on the 5070 Ti (rows from scripts/bench-head.ts)
lens-legs-5070ti/    the lens's decode cost: off / graph / full / full + MTP on the 5070 Ti (lens-legs.sh, scripts/bench-head.ts)
```

**Belongs here:** the machine-readable result behind a number in `evidence.md`: a
`summary.json`, a probe's JSON, a `results.jsonl`. Small, and cited by path from
`evidence.md` the moment it lands.

**Does not belong here:** logs and transcripts (they stay in `local/`), completions and generated
code (a gate run's `humaneval-*/` directories), the third-party HumanEval set, a result nothing
in `evidence.md` cites, anything from an experiment that did not become a number in
`head.toml` or `gates.toml` (that is the project wrapper's `research/`).
