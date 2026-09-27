#!/usr/bin/env bun
import { dirname, relative, resolve } from "node:path";
// The dependency rule of the tree, enforced — the modular-monolith rules: a feature imports shared/
// and itself; another feature only through that feature's index.ts; shared/ never imports a
// feature; platform/ imports ports/ only; nothing imports main.ts; the feature graph is acyclic;
// a test may also reach test/fakes/ and shared/platform/; main.ts and its test read package.json
// (the version `rig --version` prints).
import { Glob } from "bun";

const root = resolve(import.meta.dir, "..");
const src = `${root}/src`;
type Place =
  | { kind: "main" }
  | { kind: "shared"; module: string }
  | { kind: "feature"; feature: string };
const place = (file: string): Place => {
  const rel = relative(src, file);
  const [top, second] = rel.split("/");
  if (rel === "main.ts" || rel === "main.test.ts") return { kind: "main" };
  if (top === "shared")
    return {
      kind: "shared",
      module: rel.includes("/", 7) ? second! : second!.replace(/\.ts$/, ""),
    };
  if (top === "features" && second) return { kind: "feature", feature: second };
  throw new Error(`unplaced file ${rel}: every file is main.ts, shared/… or features/<feature>/…`);
};
const violations: string[] = [];
const edges = new Map<string, Set<string>>();
for await (const path of new Glob("**/*.ts").scan({ cwd: src, absolute: true })) {
  const from = place(path);
  const isTest = /\.test\.ts$/.test(path);
  const text = await Bun.file(path).text();
  for (const m of text.matchAll(/^\s*(?:import|export)[^'"]*from\s+["'](\.[^"']+)["']/gm)) {
    const target = resolve(dirname(path), m[1]!);
    const edge = `${relative(src, path)} → ${relative(src, target)}`;
    if (!target.startsWith(src)) {
      if (from.kind === "main" && target === `${root}/package.json`) continue;
      if (!isTest || !target.includes("/test/fakes/"))
        violations.push(`${edge}: outside src/ (only tests may reach test/fakes)`);
      continue;
    }
    const to = place(target);
    if (from.kind === "main") {
      if (to.kind === "feature" && !target.endsWith(`/features/${to.feature}/index.ts`))
        violations.push(`${edge}: main.ts imports a feature only through its index.ts`);
      continue;
    }
    if (to.kind === "main") {
      violations.push(`${edge}: nothing imports main.ts`);
      continue;
    }
    if (
      to.kind === "shared" &&
      to.module === "platform" &&
      !(from.kind === "shared" && from.module === "platform") &&
      !isTest
    ) {
      violations.push(`${edge}: only main.ts constructs the platform`);
      continue;
    }
    if (from.kind === "shared") {
      if (to.kind === "feature") {
        violations.push(`${edge}: shared/ never imports a feature`);
        continue;
      }
      if (from.module === "platform" && !["platform", "ports"].includes(to.module)) {
        violations.push(`${edge}: platform/ imports ports/ only`);
        continue;
      }
      continue;
    }
    if (to.kind === "feature" && to.feature !== from.feature) {
      if (!target.endsWith(`/features/${to.feature}/index.ts`)) {
        violations.push(`${edge}: another feature is imported only through its index.ts`);
        continue;
      }
      if (!edges.has(from.feature)) edges.set(from.feature, new Set());
      edges.get(from.feature)!.add(to.feature);
    }
  }
}
const state = new Map<string, "visiting" | "done">();
const walk = (n: string, trail: string[]) => {
  if (state.get(n) === "done") return;
  if (state.get(n) === "visiting") {
    violations.push(`cycle: ${[...trail, n].join(" → ")}`);
    return;
  }
  state.set(n, "visiting");
  for (const t of edges.get(n) ?? []) walk(t, [...trail, n]);
  state.set(n, "done");
};
for (const n of edges.keys()) walk(n, []);
if (violations.length) {
  console.error(`orthogonality: ${violations.length} violation(s)\n  ${violations.join("\n  ")}`);
  process.exit(1);
}
console.log(
  `orthogonality: features import shared and themselves; cross-feature only via index.ts (${edges.size ? [...edges].map(([f, t]) => `${f} → ${[...t].join(", ")}`).join("; ") : "none today"}); platform ← ports`,
);
