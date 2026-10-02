// Writes the gates' corpus for the head's pinned engine to local/calibration/engine-corpus.txt (the
// layout's calibrationDir), for the K-cache bias calibration recipe in evidence.md:
// `bun tools/engine-corpus.ts [chars]`.

import { join } from "node:path";
import { realPorts } from "@rig/adapters";
import { layoutAt } from "@rig/core";
import { engineCorpus, engineSource, loadEngine } from "@rig/engine";

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
