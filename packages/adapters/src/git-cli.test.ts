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

describe("GitCli.exportTree", () => {
  test("the named paths as committed: an uncommitted edit, an untracked file and an unnamed path stay out", async () => {
    const repo = mkdtempSync(join(tmpdir(), "rig-git-export-"));
    const into = mkdtempSync(join(tmpdir(), "rig-git-into-"));
    try {
      const at = (...args: string[]) =>
        Bun.spawnSync(["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t", ...args]);
      at("init", "-q");
      Bun.spawnSync(["mkdir", "-p", join(repo, "src"), join(repo, "heads", "h")]);
      writeFileSync(join(repo, "src", "main.ts"), "committed");
      writeFileSync(join(repo, "heads", "h", "head.toml"), "name = 'h'");
      writeFileSync(join(repo, "PRODUCT.md"), "not named");
      at("add", "-A");
      at("commit", "-qm", "base");
      writeFileSync(join(repo, "src", "main.ts"), "edited, not committed");
      writeFileSync(join(repo, "src", "new.ts"), "untracked");
      await new GitCli(new BunShell()).exportTree(repo, "HEAD", ["src", "heads/h"], into);
      const files = new Bun.Glob("**/*").scanSync({ cwd: into, dot: true });
      expect([...files].sort()).toEqual(["heads/h/head.toml", "src/main.ts"]);
      expect(await Bun.file(join(into, "src", "main.ts")).text()).toBe("committed");
    } finally {
      rmSync(repo, { recursive: true, force: true });
      rmSync(into, { recursive: true, force: true });
    }
  });
});
