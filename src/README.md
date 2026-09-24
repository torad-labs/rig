# src/

Feature folders, the way the reference material lays a TypeScript program out: what changes
together lives together, each feature publishes one `index.ts`, and `shared/` holds only what
every feature needs. Three levels at most; kebab-case; tests beside the code they test.

```
main.ts              the entry point: builds the platform, wires each feature's service into its command
features/<feature>/  one folder per thing rig does, named for it (machine-check, engine-build,
                     pack-download, pack-derivation, head-serving, systemd-unit, head-description,
                     head-bringup, head-gating, gpu-rental)
  index.ts             the public API: what main.ts and other features may import
  <verb>.command.ts    the CLI entry point for `rig <verb>`: flags in, the service called, an exit code out
  <feature>.service.ts the use case: decides, returns a Result, does not print
  <feature>.test.ts    the feature over in-memory ports
  <noun>.ts            the feature's own models and helpers: gates-config, box-state, server-argv, gguf …
shared/              cross-feature primitives only, never importing a feature
  ports/               the seams to the machine, one capability per file; index.ts the set a service is handed
  platform/            the real adapters, one per port, named for the tool; index.ts constructs them
  cli/                 the argument grammar and the command contract
  head/                a head as data: head, head-config, draft-head; head-endpoint, a serving head over http
  engine/              the engine as data and the corpus built from its tree
  artifact.ts          a sha-pinned file  ·  layout.ts  where things live  ·  result.ts  outcomes and exit codes
  stamp.ts             a moment as a path-safe token, for runs and backups
```

The rule, enforced by `scripts/lint-orthogonality.ts` (`bun run lint`): a feature imports
`shared/` and itself; another feature only through that feature's `index.ts`; `shared/` never
imports a feature; `platform/` imports `ports/` only; only `main.ts` constructs the platform;
the feature graph is acyclic. When a feature needs another's use case at runtime, it declares
the interface it needs and `main.ts` hands it the other feature's service.

Adding a command: a folder here with the four files above, one line in `main.ts`, a row in
`docs/architecture.md`.
