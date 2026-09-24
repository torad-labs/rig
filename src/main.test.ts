import { describe, expect, test } from "bun:test";
import { dirname, join } from "node:path";
import { version } from "../package.json" with { type: "json" };

const root = dirname(import.meta.dir);
const rig = (...args: string[]) =>
  Bun.spawnSync(["bun", join(root, "src/main.ts"), ...args], {
    env: { ...process.env, RIG_ROOT: root },
  });

describe("rig", () => {
  test("--version prints package.json's version, for an installer or a proxy to read", () => {
    const run = rig("--version");
    expect(run.exitCode).toBe(0);
    expect(run.stdout.toString()).toBe(`rig ${version}\n`);
  });
  test("<command> --help prints the command's usage and runs nothing", () => {
    const run = rig("build", "--help");
    expect(run.exitCode).toBe(0);
    expect(run.stderr.toString()).toStartWith("usage: rig build");
  });
});
