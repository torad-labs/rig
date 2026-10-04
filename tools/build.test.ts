import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { BUILT_FROM } from "@rig/core";
import { stampOf } from "./build.ts";

const root = resolve(import.meta.dir, "..");
const TREE = "47c0853456eff6e35f31bf988a784c15f8690f19";
const { version } = await Bun.file(join(root, "apps/cli/package.json")).json();

describe("the build's stamp", () => {
  test("HEAD's tree, and -dirty after it when a compiled source is not the tree's", () => {
    expect(stampOf(`${TREE}\n`, "")).toBe(TREE);
    expect(stampOf(TREE, " M packages/rental/src/gpu-rental.service.ts\n")).toBe(`${TREE}-dirty`);
    expect(stampOf(TREE, "?? apps/cli/src/new.ts\n")).toBe(`${TREE}${BUILT_FROM.dirty}`);
  });
  test("what is not a tree id is refused, never stamped", () => {
    for (const said of ["", "fatal: not a git repository", TREE.slice(0, 12)])
      expect(() => stampOf(said, "")).toThrow(/not a tree/);
  });
});

describe("bun run build", () => {
  test("compiles a rig that prints the tree it was built from, and keeps --version as the release reads it", () => {
    const dir = mkdtempSync(join(tmpdir(), "rig-build-"));
    try {
      const out = join(dir, "rig");
      const build = Bun.spawnSync(["bun", join(import.meta.dir, "build.ts"), out], { cwd: root });
      expect(build.exitCode).toBe(0);
      const tree = Bun.spawnSync([
        "git",
        "-C",
        root,
        "rev-parse",
        BUILT_FROM.ref,
      ]).stdout.toString();
      const said = Bun.spawnSync([out, BUILT_FROM.flag]);
      expect(said.exitCode).toBe(0);
      // -dirty where this checkout holds an uncommitted source; CI's checkout holds none
      expect(said.stdout.toString()).toMatch(
        new RegExp(`^${tree.trim()}(${BUILT_FROM.dirty})?\\n$`),
      );
      expect(Bun.spawnSync([out, "--version"]).stdout.toString()).toBe(`rig ${version}\n`);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
