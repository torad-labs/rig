# .github/

`workflows/rig.yml` — rig's own checks on every pull request and push to main: `bun install
--frozen-lockfile`, `bun run lint` (the layer rule and the types), `bun test` (every slice over
in-memory ports), `bun run build`, and the binary as an artifact. Nothing here touches a card;
the gates that do run through `rig gate` on a machine with one.

**Belongs here:** a workflow that runs without a GPU, a model or a credential. Actions are
pinned by commit sha. The org's secret scan, actionlint, zizmor and title checks come from
torad-labs/.github and are not duplicated here.

**Does not belong here:** a gate that needs a card, a deploy, a secret, a workflow that
installs a service.
