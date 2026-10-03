// Regenerates packages/head/src/rendered-flag-aliases.json: every spelling an engine accepts for each flag serve
// renders, so a head's free-form lists are refused under all of them. One block per engine rig serves on — the
// default pin, and every head's own [engine] sha — because a head serves on the engine IT pins, and a flag rig
// renders for it has to be declared there. `aliases` stays the default pin's block, the flat table head-config reads.
// Run after moving the pin or a head's pin: `bun tools/rendered-flag-aliases.ts`.

import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { realPorts } from "@rig/adapters";
import { layoutAt } from "@rig/core";
import { argAliases, engineSource, loadEngine } from "@rig/engine";
import { RENDERED, RENDERED_BY_HEAD_ENGINE } from "@rig/head";

const root = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const p = realPorts();
const layout = layoutAt(root);
const engine = await loadEngine(p.fs, layout);
if (!engine.ok) {
  console.error(engine.message);
  process.exit(1);
}
const pin = engine.value; // narrowed once: tableAt below is a closure, and TS does not carry the guard into it
const src = await engineSource(p.fs, p.git, layout, pin);
if (!src.ok) {
  console.error(src.message);
  process.exit(1);
}
const table = await p.fs.readText(join(src.value, "common/arg.cpp"));
const pinned = argAliases(table, RENDERED);

/** the arg table at `sha`, read out of the fork's git rather than a checkout: a head's pin is rarely the commit the
 *  submodule sits on, and nothing here needs a worktree at it */
function tableAt(sha: string): string | null {
  const show = Bun.spawnSync(["git", "-C", pin.submoduleDir, "show", `${sha}:common/arg.cpp`]);
  return show.exitCode === 0 ? show.stdout.toString() : null;
}

const engines: Record<string, Record<string, string[]>> = { [pin.fork.sha]: pinned };
const missing: string[] = [];
for (const name of (await readdir(join(root, "heads"))).sort()) {
  const toml = join(root, "heads", name, "head.toml");
  if (!(await p.fs.exists(toml))) continue;
  const own = /\[engine\][^[]*?sha = "([0-9a-f]{40})"/s.exec(await p.fs.readText(toml))?.[1];
  if (!own || engines[own]) continue;
  const head = tableAt(own);
  if (head === null) {
    missing.push(`${name} pins ${own.slice(0, 7)}, which ${pin.submoduleDir} does not have`);
    continue;
  }
  // every flag of RENDERED and RENDERED_BY_HEAD_ENGINE the head's engine declares: one it does not is a flag no head on it may render, and the
  // test says which head renders what, so a partial block is the honest record rather than a throw here
  const declared: Record<string, string[]> = {};
  for (const flag of [...RENDERED, ...RENDERED_BY_HEAD_ENGINE]) {
    try {
      Object.assign(declared, argAliases(head, [flag]));
    } catch {
      /* this engine does not declare it */
    }
  }
  // a block with none of the flags every head renders is not an engine's arg table, it is a parse that found nothing:
  // writing it would make the per-head check green against an empty record
  if (!("-m" in declared) || !("-ngl" in declared))
    missing.push(
      `${name} pins ${own.slice(0, 7)} and its arg table parsed to ${Object.keys(declared).length} of the ${RENDERED.length} flags rig renders: not an arg table`,
    );
  engines[own] = declared;
}
if (missing.length > 0) {
  console.error(
    `${missing.join("\n")}\nfetch the fork's commits first: git -C ${pin.submoduleDir} fetch origin`,
  );
  process.exit(1);
}

const out = join(root, "packages/head/src/rendered-flag-aliases.json");
await p.fs.writeText(
  out,
  `${JSON.stringify({ engine: pin.fork.sha, aliases: pinned, engines }, null, 2)}\n`,
);
// the writer indents every array; the repo's formatter keeps short ones on one line, so the
// remedy this script IS must leave `bun run lint` green rather than red in a different place
Bun.spawnSync(["bunx", "biome", "format", "--write", out], { cwd: root });
const blocks = Object.keys(engines)
  .map((sha) => sha.slice(0, 7))
  .join(", ");
console.log(`wrote ${out} from ${src.value} at ${pin.sha7}; blocks: ${blocks}`);
