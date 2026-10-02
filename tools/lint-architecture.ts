#!/usr/bin/env bun
// The architecture, checked where structure alone does not hold it. Structure holds most of it: the
// isolated linker (bunfig.toml) resolves only what a workspace declares, and every package exports
// its src/index.ts alone, so an undeclared package or a reach into another's internals does not
// resolve. What is left, and checked here:
//   - the layers: a workspace depends at runtime only on what LAYERS allows it, and every workspace
//     has a row (a new one without a decision fails by name);
//   - a test's support (@rig/testing, @rig/adapters) is a devDependency, and code that ships never
//     imports a devDependency;
//   - no relative import leaves its workspace;
//   - within the cli, only the composition root (main.ts) constructs the adapters.
import { dirname, join, relative, resolve } from "node:path";
import { Glob } from "bun";

/** each workspace and the workspaces it may depend on at runtime */
export const LAYERS: Record<string, readonly string[]> = {
  "@rig/core": [],
  "@rig/engine": ["@rig/core"],
  "@rig/head": ["@rig/core", "@rig/engine"],
  "@rig/machine": ["@rig/core", "@rig/engine", "@rig/head"],
  "@rig/pack": ["@rig/core", "@rig/head"],
  "@rig/serve": ["@rig/core", "@rig/engine", "@rig/head"],
  "@rig/gate": ["@rig/core", "@rig/engine", "@rig/head"],
  "@rig/image": ["@rig/core", "@rig/engine", "@rig/head"],
  "@rig/rental": ["@rig/core", "@rig/engine", "@rig/head", "@rig/image"],
  "@rig/adapters": ["@rig/core"],
  "@rig/testing": ["@rig/core", "@rig/engine", "@rig/head", "@rig/image", "@rig/registry"],
  "@rig/registry": [],
  "@rig/cli": [
    "@rig/adapters",
    "@rig/core",
    "@rig/engine",
    "@rig/gate",
    "@rig/head",
    "@rig/image",
    "@rig/machine",
    "@rig/pack",
    "@rig/rental",
    "@rig/serve",
  ],
  "@rig/tools": ["@rig/adapters", "@rig/core", "@rig/engine", "@rig/head", "@rig/image"],
};
/** what a test may lean on beyond its own workspace's dependencies */
const TEST_SUPPORT = ["@rig/testing", "@rig/adapters"];

export interface Workspace {
  /** repo-relative directory */
  dir: string;
  name: string;
  dependencies: Record<string, string>;
  devDependencies: Record<string, string>;
  /** repo-relative path → source text, every .ts under the workspace */
  files: Map<string, string>;
}

const IMPORT =
  /^\s*(?:import|export)\b[^'"]*?\bfrom\s+["']([^"']+)["']|\bimport\s*\(\s*["']([^"']+)["']\s*\)/gm;
const isTest = (path: string) => /\.test\.ts$/.test(path);
const ours = (spec: string) => spec.startsWith("@rig/");

export function check(workspaces: Workspace[]): string[] {
  const violations: string[] = [];
  const names = new Set(workspaces.map((w) => w.name));
  for (const w of workspaces) {
    const allowed = LAYERS[w.name];
    if (!allowed) {
      violations.push(`${w.name} (${w.dir}) has no row in LAYERS: decide what it may depend on`);
      continue;
    }
    for (const dep of Object.keys(w.dependencies).filter(ours)) {
      if (!names.has(dep)) violations.push(`${w.name} depends on ${dep}, which is no workspace`);
      else if (!allowed.includes(dep))
        violations.push(
          `${w.name} depends on ${dep}: its layer allows ${allowed.join(", ") || "nothing"}`,
        );
    }
    for (const dep of Object.keys(w.devDependencies).filter(ours))
      if (!TEST_SUPPORT.includes(dep) && !allowed.includes(dep))
        violations.push(
          `${w.name} devDepends on ${dep}: a test leans only on ${TEST_SUPPORT.join(", ")}`,
        );

    for (const [path, text] of w.files) {
      for (const m of text.matchAll(IMPORT)) {
        const spec = (m[1] ?? m[2])!;
        if (spec.startsWith(".")) {
          const target = join(dirname(path), spec);
          if (relative(w.dir, target).startsWith(".."))
            violations.push(`${path} → ${spec}: a relative import stays inside ${w.dir}`);
          continue;
        }
        if (!ours(spec)) continue;
        const pkg = spec.split("/").slice(0, 2).join("/");
        if (pkg === w.name) {
          violations.push(`${path} → ${spec}: a workspace imports itself relatively, not by name`);
          continue;
        }
        const runtime = pkg in w.dependencies;
        if (!isTest(path) && !runtime)
          violations.push(
            `${path} → ${pkg}: code that ships imports only ${w.name}'s dependencies${pkg in w.devDependencies ? ` (${pkg} is a devDependency)` : ""}`,
          );
        if (
          w.name === "@rig/cli" &&
          pkg === "@rig/adapters" &&
          !/\/src\/main(\.test)?\.ts$/.test(path)
        )
          violations.push(
            `${path} → @rig/adapters: only the composition root constructs the machine`,
          );
      }
    }
  }
  return violations;
}

async function readWorkspaces(root: string): Promise<Workspace[]> {
  const manifest = await Bun.file(join(root, "package.json")).json();
  const globs: string[] = manifest.workspaces.packages;
  const out: Workspace[] = [];
  for (const pattern of globs)
    for await (const pj of new Glob(`${pattern}/package.json`).scan({ cwd: root })) {
      const dir = dirname(pj);
      const m = await Bun.file(join(root, pj)).json();
      const files = new Map<string, string>();
      for await (const f of new Glob("**/*.ts").scan({ cwd: join(root, dir) }))
        if (!f.includes("node_modules/"))
          files.set(join(dir, f), await Bun.file(join(root, dir, f)).text());
      out.push({
        dir,
        name: m.name,
        dependencies: m.dependencies ?? {},
        devDependencies: m.devDependencies ?? {},
        files,
      });
    }
  return out;
}

if (import.meta.main) {
  const root = resolve(import.meta.dir, "..");
  const workspaces = await readWorkspaces(root);
  const violations = check(workspaces);
  if (violations.length) {
    console.error(`architecture: ${violations.length} violation(s)\n  ${violations.join("\n  ")}`);
    process.exit(1);
  }
  const files = workspaces.reduce((n, w) => n + w.files.size, 0);
  console.log(
    `architecture: ${workspaces.length} workspaces, ${files} files: every dependency on its layer, no relative import leaves its workspace, only main.ts constructs the machine`,
  );
}
