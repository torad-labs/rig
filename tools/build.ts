#!/usr/bin/env bun
// `bun run build`: dist/rig, stamped with the tree it was built from (@rig/core's BUILT_FROM). `rig vast up` and `lab`
// ship dist/rig to a box and refuse one whose stamp is not HEAD's tree, before anything is rented. Takes the output
// path, dist/rig by default.
import { resolve } from "node:path";
import { BUILT_FROM } from "@rig/core";

const root = resolve(import.meta.dir, "..");

/** what the compile reads from the checkout, its tests aside, and the lock node_modules was installed by: a change to
 *  one is in the binary and not in HEAD's tree */
const SOURCES = [
  "apps",
  "packages",
  "package.json",
  "bun.lock",
  "bunfig.toml",
  "tsconfig.json",
  ":(exclude,glob)**/*.test.ts",
];

/** the stamp: the tree `git rev-parse` named, -dirty after it when `git status --porcelain` listed a source */
export function stampOf(tree: string, changed: string): string {
  const id = tree.trim();
  if (!/^([0-9a-f]{40}|[0-9a-f]{64})$/.test(id))
    throw new Error(`git rev-parse ${BUILT_FROM.ref} said ${JSON.stringify(id)}, not a tree id`);
  return changed.trim() === "" ? id : `${id}${BUILT_FROM.dirty}`;
}

function git(...args: string[]): string {
  const run = Bun.spawnSync(["git", "-C", root, ...args], { stderr: "inherit" });
  if (run.exitCode !== 0) throw new Error(`git ${args.join(" ")} exited ${run.exitCode}`);
  return run.stdout.toString();
}

if (import.meta.main) {
  const outfile = process.argv[2] ?? "dist/rig";
  const changed = git("status", "--porcelain", "--untracked-files=all", "--", ...SOURCES);
  const stamp = stampOf(git("rev-parse", BUILT_FROM.ref), changed);
  if (changed.trim())
    console.error(
      `rig: these are not HEAD's, so rig vast up and lab refuse this build:\n${changed}`,
    );
  const compile = Bun.spawnSync(
    [
      "bun",
      "build",
      "--compile",
      "--target=bun-linux-x64",
      "--define",
      `RIG_BUILT_FROM=${JSON.stringify(stamp)}`,
      "--outfile",
      outfile,
      "apps/cli/src/main.ts",
    ],
    { cwd: root, stdout: "inherit", stderr: "inherit" },
  );
  if (compile.exitCode === 0) console.log(`${outfile} built from tree ${stamp}`);
  process.exit(compile.exitCode);
}
