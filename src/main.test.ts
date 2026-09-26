import { describe, expect, test } from "bun:test";
import {
  chmodSync,
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { version } from "../package.json" with { type: "json" };

const root = dirname(import.meta.dir);
const rigAt = (at: string, ...args: string[]) =>
  Bun.spawnSync(["bun", join(root, "src/main.ts"), ...args], {
    env: { ...process.env, RIG_ROOT: at },
  });
const rig = (...args: string[]) => rigAt(root, ...args);

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
  test("a numeric flag given a value it cannot use stops the command with exit 64, never runs it on the default", () => {
    const run = rig("describe", "bonsai-2-27b", "--gpu", "1x");
    expect(run.exitCode).toBe(64);
    expect(run.stdout.toString()).toBe("");
    expect(run.stderr.toString()).toStartWith(
      'rig: --gpu takes a whole number, not "1x"\nusage: rig describe',
    );
  });
  test("an engine.toml this binary cannot read stops every command but vast, whose idle check still runs", () => {
    const at = mkdtempSync(join(tmpdir(), "rig-main-"));
    try {
      mkdirSync(join(at, "engine"));
      const pin = readFileSync(join(root, "engine/engine.toml"), "utf8");
      writeFileSync(join(at, "engine/engine.toml"), `${pin}\n[future]\nkey = 1\n`);
      copyFileSync(join(root, "vast.toml"), join(at, "vast.toml"));
      const build = rigAt(at, "build");
      expect(build.exitCode).toBe(1);
      expect(build.stderr.toString()).toContain("future");
      const idle = rigAt(at, "vast", "idle-check", "--json");
      expect(idle.exitCode).toBe(0);
      expect(JSON.parse(idle.stdout.toString())).toEqual({ action: "no-box" });
    } finally {
      rmSync(at, { recursive: true, force: true });
    }
  });
});

describe("install.sh", () => {
  test("a release whose binary does not run here fails and keeps the install it would replace", () => {
    const at = mkdtempSync(join(tmpdir(), "rig-install-"));
    /** a release as release.yml publishes it, its dist/rig a script with this body */
    const release = (name: string, body: string) => {
      const dir = join(at, name);
      mkdirSync(join(dir, "pack/rig/dist"), { recursive: true });
      writeFileSync(join(dir, "pack/rig/dist/rig"), `#!/bin/sh\n${body}\n`);
      chmodSync(join(dir, "pack/rig/dist/rig"), 0o755);
      Bun.spawnSync(["tar", "-C", join(dir, "pack"), "-czf", `${dir}/rig-linux-x64.tar.gz`, "rig"]);
      const sum = Bun.spawnSync(["sha256sum", "rig-linux-x64.tar.gz"], { cwd: dir }).stdout;
      writeFileSync(join(dir, "rig-linux-x64.tar.gz.sha256"), sum);
      return `file://${dir}`;
    };
    const install = (url: string) =>
      Bun.spawnSync(["sh", join(root, "install.sh")], {
        env: {
          ...process.env,
          RIG_HOME: join(at, "home"),
          RIG_BIN_DIR: join(at, "bin"),
          RIG_RELEASE_URL: url,
        },
      });
    try {
      const good = install(release("good", 'echo "rig 9.9.9"'));
      expect(good.exitCode).toBe(0);
      expect(good.stdout.toString()).toContain("rig install: rig 9.9.9 in");
      const broken = install(release("broken", "exit 1"));
      expect(broken.exitCode).toBe(1);
      expect(broken.stderr.toString()).toContain("the release's dist/rig does not run here");
      const kept = Bun.spawnSync([join(at, "bin/rig"), "--version"]).stdout.toString();
      expect(kept).toBe("rig 9.9.9\n");
    } finally {
      rmSync(at, { recursive: true, force: true });
    }
  });
});
