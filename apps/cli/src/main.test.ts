import { describe, expect, test } from "bun:test";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { version } from "../package.json" with { type: "json" };

const root = resolve(import.meta.dir, "../../..");
const rigAt = (at: string, ...args: string[]) =>
  Bun.spawnSync(["bun", join(import.meta.dir, "main.ts"), ...args], {
    env: { ...process.env, RIG_ROOT: at },
  });
const rig = (...args: string[]) => rigAt(root, ...args);

describe("rig", () => {
  test("--version prints package.json's version, for an installer or a proxy to read", () => {
    const run = rig("--version");
    expect(run.exitCode).toBe(0);
    expect(run.stdout.toString()).toBe(`rig ${version}\n`);
  });
  test("--built-from: a rig run from the source was not stamped by bun run build, and says so", () => {
    const run = rig("--built-from");
    expect(run.exitCode).toBe(1);
    expect(run.stdout.toString()).toBe("");
    // the whole of stderr: a crash prints the source around it, this message's line included
    expect(run.stderr.toString()).toBe(
      "rig: this rig was not compiled by bun run build, so it does not know its tree\n",
    );
  });
  test("<command> --help prints the command's usage and runs nothing", () => {
    const run = rig("build", "--help");
    expect(run.exitCode).toBe(0);
    expect(run.stderr.toString()).toStartWith("usage: rig build");
  });
  test("a flag given a value it cannot use stops the command with exit 64, never runs it on the default", () => {
    const run = rig("describe", "bonsai-2-27b", "--gpu", "1x");
    expect(run.exitCode).toBe(64);
    expect(run.stdout.toString()).toBe("");
    expect(run.stderr.toString()).toStartWith(
      'rig: --gpu takes a card index, distinct indices (0,1) or auto, not "1x"\nusage: rig describe',
    );
  });
  test("a subcommand named after an Object.prototype key is an unknown one: usage and exit 64, never a crash", () => {
    const run = rig("engine", "constructor", "--tree", "x");
    expect(run.exitCode).toBe(64);
    expect(run.stderr.toString()).not.toContain("TypeError");
    expect(run.stderr.toString()).toContain("usage: rig engine test");
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

describe("a head's own engine pin", () => {
  test("every command naming the head runs on the head's commit; the others on engine.toml's", () => {
    const at = mkdtempSync(join(tmpdir(), "rig-main-"));
    try {
      mkdirSync(join(at, "engine"));
      mkdirSync(join(at, "heads/bonsai-2-27b"), { recursive: true });
      copyFileSync(join(root, "engine/engine.toml"), join(at, "engine/engine.toml"));
      const toml = readFileSync(join(root, "heads/bonsai-2-27b/head.toml"), "utf8");
      const own = '[engine]\nsha = "a786bcdc42f79dbb3a513def3eb97cfe5a155e5c"\n\n[context]';
      writeFileSync(join(at, "heads/bonsai-2-27b/head.toml"), toml.replace("[context]", own));
      const describe = rigAt(at, "describe", "bonsai-2-27b", "--gpu", "0");
      expect(describe.exitCode).toBe(0);
      expect(JSON.parse(describe.stdout.toString()).engine_commit).toBe("a786bcd");
      const pinned = readFileSync(join(root, "engine/engine.toml"), "utf8");
      const sha7 = /^sha = "([0-9a-f]{7})/m.exec(pinned)?.[1];
      writeFileSync(join(at, "heads/bonsai-2-27b/head.toml"), toml);
      const plain = rigAt(at, "describe", "bonsai-2-27b", "--gpu", "0");
      expect(JSON.parse(plain.stdout.toString()).engine_commit).toBe(sha7);
    } finally {
      rmSync(at, { recursive: true, force: true });
    }
  });
});

describe("install.sh", () => {
  /** a scratch directory with a release builder and the installer pointed at it, never ~/.local/share/rig */
  const scratch = () => {
    const at = mkdtempSync(join(tmpdir(), "rig-install-"));
    /** a release as release.yml publishes it, its dist/rig a script with this body, and `files` beside it */
    const release = (name: string, body: string, files: Record<string, string> = {}) => {
      const dir = join(at, name);
      mkdirSync(join(dir, "pack/rig/dist"), { recursive: true });
      writeFileSync(join(dir, "pack/rig/dist/rig"), `#!/bin/sh\n${body}\n`);
      chmodSync(join(dir, "pack/rig/dist/rig"), 0o755);
      for (const [path, text] of Object.entries(files)) {
        mkdirSync(join(dir, "pack/rig", path, ".."), { recursive: true });
        writeFileSync(join(dir, "pack/rig", path), text);
      }
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
    return { at, release, install };
  };
  test("an upgrade replaces the heads the release ships, head by head, and keeps one the user added", () => {
    const { at, release, install } = scratch();
    const home = join(at, "home");
    try {
      const first = release("first", 'echo "rig 1.0.0"', {
        "heads/README.md": "v1",
        "heads/bonsai-2-27b/head.toml": "v1",
        "heads/bonsai-2-27b/stale.md": "only in v1",
        "engine/engine.toml": "v1",
      });
      expect(install(first).exitCode).toBe(0);
      mkdirSync(join(home, "heads/mine"));
      writeFileSync(join(home, "heads/mine/head.toml"), "the user's own head");
      const second = release("second", 'echo "rig 1.1.0"', {
        "heads/README.md": "v2",
        "heads/bonsai-2-27b/head.toml": "v2",
        "heads/glm-5.3-flash/head.toml": "v2",
        "engine/engine.toml": "v2",
      });
      const upgrade = install(second);
      expect(upgrade.exitCode).toBe(0);
      expect(readFileSync(join(home, "heads/mine/head.toml"), "utf8")).toBe("the user's own head");
      expect(readFileSync(join(home, "heads/README.md"), "utf8")).toBe("v2");
      expect(readFileSync(join(home, "heads/bonsai-2-27b/head.toml"), "utf8")).toBe("v2");
      expect(existsSync(join(home, "heads/bonsai-2-27b/stale.md"))).toBe(false); // a shipped head is the release's whole
      expect(readFileSync(join(home, "heads/glm-5.3-flash/head.toml"), "utf8")).toBe("v2");
      expect(readFileSync(join(home, "engine/engine.toml"), "utf8")).toBe("v2");
    } finally {
      rmSync(at, { recursive: true, force: true });
    }
  });
  test("a release whose binary does not run here fails and keeps the install it would replace", () => {
    const { at, release, install } = scratch();
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
