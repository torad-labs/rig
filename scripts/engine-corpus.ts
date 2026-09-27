// Writes the gates' corpus for the head's pinned engine to local/calibration/engine-corpus.txt (the
// layout's calibrationDir), for the K-cache bias calibration recipe in evidence.md:
// `bun scripts/engine-corpus.ts [chars]`.

import { join } from "node:path";
import { engineSource, loadEngine } from "../src/shared/engine/engine.ts";
import { engineCorpus } from "../src/shared/engine/engine-corpus.ts";
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
const out = join(layout.calibrationDir, "engine-corpus.txt");
await p.fs.mkdirp(layout.calibrationDir);
await p.fs.writeText(
  out,
  await engineCorpus(p.fs, src.value, Number(process.argv[2] ?? 3_145_728)),
);
console.error(`wrote ${out}`);
