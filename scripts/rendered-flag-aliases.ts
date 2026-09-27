// Regenerates src/shared/head/rendered-flag-aliases.json from the pinned engine's common/arg.cpp:
// every spelling the engine accepts for each flag serve renders, so a head's free-form lists are
// refused under all of them. Run after moving the engine pin: `bun scripts/rendered-flag-aliases.ts`.

import { join } from "node:path";
import { argAliases } from "../src/shared/engine/arg-aliases.ts";
import { engineSource, loadEngine } from "../src/shared/engine/engine.ts";
import { RENDERED } from "../src/shared/head/head-config.ts";
import { layoutAt } from "../src/shared/layout.ts";
import { realPorts } from "../src/shared/platform/index.ts";

const root = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const p = realPorts();
const layout = layoutAt(root);
const engine = await loadEngine(p.fs, layout);
if (!engine.ok) {
  console.error(engine.message);
  process.exit(1);
}
const src = await engineSource(p.fs, p.git, layout, engine.value);
if (!src.ok) {
  console.error(src.message);
  process.exit(1);
}
const table = await p.fs.readText(join(src.value, "common/arg.cpp"));
const out = join(root, "src/shared/head/rendered-flag-aliases.json");
await p.fs.writeText(
  out,
  `${JSON.stringify({ engine: engine.value.fork.sha, aliases: argAliases(table, RENDERED) }, null, 2)}\n`,
);
// the writer indents every array; the repo's formatter keeps short ones on one line, so the
// remedy this script IS must leave `bun run lint` green rather than red in a different place
Bun.spawnSync(["bunx", "biome", "format", "--write", out], { cwd: root });
console.log(`wrote ${out} from ${src.value} at ${engine.value.sha7}`);
