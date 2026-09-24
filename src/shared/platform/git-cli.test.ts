import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BunShell } from "./bun-shell.ts";
import { GitCli } from "./git-cli.ts";

const dir = mkdtempSync(join(tmpdir(), "rig-git-cli-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const git = (...args: string[]) =>
  Bun.spawnSync(["git", "-C", dir, "-c", "user.email=t@t", "-c", "user.name=t", ...args]);

describe("GitCli.isClean", () => {
  test("an untracked source is a dirty tree (the engine's CMake globs *.cu); an ignored file is not", async () => {
    git("init", "-q");
    writeFileSync(join(dir, "kernel.cu"), "tracked");
    writeFileSync(join(dir, ".gitignore"), "build/\n");
    git("add", "-A");
    git("commit", "-qm", "base");
    const cli = new GitCli(new BunShell());
    expect(await cli.isClean(dir)).toBe(true);
    Bun.spawnSync(["mkdir", "-p", join(dir, "build")]);
    writeFileSync(join(dir, "build", "out.o"), "ignored");
    expect(await cli.isClean(dir)).toBe(true);
    writeFileSync(join(dir, "mmq-instance-new.cu"), "untracked, and compiled");
    expect(await cli.isClean(dir)).toBe(false);
  });
});
