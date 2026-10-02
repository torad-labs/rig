import { describe, expect, test } from "bun:test";
import { layoutAt } from "@rig/core";
import { fakePorts, repoRoot } from "@rig/testing";
import { loadEngine } from "../engine.ts";
import { DriverOnlyGate } from "./driver-only-gate.service.ts";

const engineToml = await Bun.file(`${repoRoot}/engine/engine.toml`).text();
const sha256 = (text: string) => new Bun.CryptoHasher("sha256").update(text).digest("hex");
const STAGE = `/r/local/e2e-driver-only-${process.pid}`;
const TRACKED = ["heads/h/head.toml", "LICENSE", "README.md", "engine/engine.toml"];

/** the machine as the gate sees it: git lists the tracked files, tar copies them (dist/rig among them) into the
 *  staged checkout and packs it, and the container answers `run` (by default the checks pass) */
async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  p.fs.put("/r/engine/engine.toml", engineToml);
  p.fs.put("/r/install.sh", "#!/bin/sh\n");
  p.fs.put("/m/pack.gguf", "gguf");
  const engine = await loadEngine(p.fs, layout);
  if (!engine.ok) throw new Error(engine.message);
  p.shell.on(/^bun run build$/, () => {
    p.fs.put("/r/dist/rig", "elf");
    return { code: 0, stdout: "", stderr: "" };
  });
  p.shell.on(/^git -C \/r ls-files -z -- heads LICENSE README\.md engine\/engine\.toml$/, {
    code: 0,
    stdout: `${TRACKED.join("\0")}\0`,
    stderr: "",
  });
  const listed: string[] = [];
  p.shell.on(/^tar -C \/r --null -T (\S+) -cf (\S+)$/, (cmd) => {
    listed.push(...p.fs.text(cmd[5]!)!.split("\0").filter(Boolean));
    p.fs.put(cmd[7]!, "files");
    return { code: 0, stdout: "", stderr: "" };
  });
  p.shell.on(/^tar -C (\S+) -xf (\S+)$/, (cmd) => {
    for (const path of listed) p.fs.put(`${cmd[2]}/${path}`, p.fs.text(`/r/${path}`) ?? path);
    return { code: 0, stdout: "", stderr: "" };
  });
  p.shell.on(/-czf (\S+) rig$/, (cmd) => {
    p.fs.put(cmd.at(-2)!, "the release");
    return { code: 0, stdout: "", stderr: "" };
  });
  const staged: { toml?: string | undefined; release?: string[] } = {};
  p.containers.on(/^bash -c/, () => {
    staged.toml = p.fs.text(`${STAGE}/pack/rig/engine/engine.toml`);
    staged.release = ["rig-linux-x64.tar.gz", "rig-linux-x64.tar.gz.sha256", "install.sh"].map(
      (name) => p.fs.text(`${STAGE}/release/${name}`) ?? `(no ${name})`,
    );
    return { code: 0, stdout: "== the machine\n== PASS\n", stderr: "" };
  });
  return { p, gate: new DriverOnlyGate(p, layout), listed, staged, engine: engine.value };
}

describe("e2e: the driver-only gate", () => {
  test("stages the release as release.yml packs it and runs the checks in ubuntu:22.04 on the card", async () => {
    const { p, gate, listed, staged } = await setup();
    const r = await gate.run({ pack: "/m/pack.gguf", gpu: 1, base: "ubuntu:22.04" });
    expect(r.ok).toBe(true);
    // the binary built from this checkout, and only the files git tracks, as release.yml's checkout holds them
    expect(p.shell.calls[0]).toEqual(["bun", "run", "build"]);
    expect(listed).toEqual([...TRACKED, "dist/rig"]);
    expect(staged.toml).toBe(engineToml);
    expect(staged.release).toEqual([
      "the release",
      `${sha256("the release")}  rig-linux-x64.tar.gz\n`,
      "#!/bin/sh\n",
    ]);
    const pack = p.shell.calls.find((c) => c.includes("-czf"))!;
    expect(pack).toEqual([
      "tar",
      "-C",
      `${STAGE}/pack`,
      "--owner=0",
      "--group=0",
      "--numeric-owner",
      "-czf",
      `${STAGE}/release/rig-linux-x64.tar.gz`,
      "rig",
    ]);
    const run = p.containers.runs[0]!;
    expect([run.image, run.cmd[0], run.cmd[1]]).toEqual(["ubuntu:22.04", "bash", "-c"]);
    for (const check of [
      "RIG_RELEASE_URL=file:///rel sh /rel/install.sh",
      String.raw`grep -q "\"toolkitCuda\": null" /tmp/prepare.json`,
      'grep -q "^  CUDA0: " /tmp/devices.txt',
      'flock /gate.lock "$dir/llama-bench" -m /pack.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 2',
      "== PASS",
    ])
      expect(run.cmd[2]).toContain(check);
    expect(run.options).toEqual({
      gpu: 1,
      env: { HEAD_NAME: "" },
      mounts: { [`${STAGE}/release`]: "/rel", "/m/pack.gguf": "/pack.gguf" },
      writable: { "/dev/null": "/gate.lock" },
      timeoutMs: expect.any(Number),
    });
    expect(await p.fs.exists(STAGE)).toBe(false);
  });
  test("--prebuilt points engine.toml's entry for that file at the mounted copy, with its sha256", async () => {
    const { p, gate, staged, engine } = await setup();
    const name = `engine-sm120-${engine.sha7}.tar.gz`;
    p.fs.put(`/p/${name}`, "a local build");
    const r = await gate.run({
      pack: "/m/pack.gguf",
      prebuilt: `/p/${name}`,
      gpu: 0,
      base: "ubuntu:22.04",
    });
    expect(r.ok).toBe(true);
    const lines = staged.toml!.split("\n");
    const at = lines.indexOf(`url = "file:///prebuilt/${name}"`);
    expect(at).toBeGreaterThan(0);
    expect(lines.slice(at + 1).find((line) => line.startsWith("sha256 = "))).toBe(
      `sha256 = "${sha256("a local build")}"`,
    );
    // nothing else in engine.toml moved
    expect(lines.length).toBe(engineToml.split("\n").length);
    expect(p.containers.runs[0]?.options?.mounts).toEqual({
      [`${STAGE}/release`]: "/rel",
      "/m/pack.gguf": "/pack.gguf",
      [`/p/${name}`]: `/prebuilt/${name}`,
    });
  });
  test("--prebuilt naming a tarball engine.toml pins no prebuilt for is refused before any container runs", async () => {
    const { p, gate } = await setup();
    p.fs.put("/p/engine-sm120-0123abc.tar.gz", "another commit");
    const r = await gate.run({
      pack: "/m/pack.gguf",
      prebuilt: "/p/engine-sm120-0123abc.tar.gz",
      gpu: 0,
      base: "ubuntu:22.04",
    });
    expect(!r.ok && r.message).toContain(
      "engine.toml pins no prebuilt named engine-sm120-0123abc.tar.gz",
    );
    expect(p.containers.runs).toEqual([]);
    expect(await p.fs.exists(STAGE)).toBe(false);
  });
  test("the gate passes on exit 0 with the PASS line wherever it falls, and exit 0 without it is a failure", async () => {
    const { p, gate } = await setup();
    p.containers.on(/^bash -c/, { code: 0, stdout: "== PASS\n", stderr: "" });
    const first = await gate.run({ pack: "/m/pack.gguf", gpu: 0, base: "ubuntu:22.04" });
    expect(first.ok).toBe(true);
    p.containers.on(/^bash -c/, { code: 0, stdout: "== decode\n", stderr: "" });
    const none = await gate.run({ pack: "/m/pack.gguf", gpu: 0, base: "ubuntu:22.04" });
    expect(!none.ok && none.message).toContain("the driver-only gate failed (exit 0");
  });
  test("--head and the gate lock reach the container; a check that fails fails the gate, and the stage goes either way", async () => {
    const { p, gate } = await setup();
    p.containers.on(/^bash -c/, {
      code: 1,
      stdout: "== rig prepare\nprepare: no prebuilt for this card\n",
      stderr: "",
    });
    const r = await gate.run({
      pack: "/m/pack.gguf",
      gpu: 0,
      base: "ubuntu:24.04",
      head: "bonsai-2-27b",
      lock: "/run/gate.lock",
    });
    expect(!r.ok && r.message).toContain("prepare: no prebuilt for this card");
    const run = p.containers.runs[0]!;
    expect([run.image, run.options?.env, run.options?.writable]).toEqual([
      "ubuntu:24.04",
      { HEAD_NAME: "bonsai-2-27b" },
      { "/run/gate.lock": "/gate.lock" },
    ]);
    expect(await p.fs.exists(STAGE)).toBe(false);
  });
});
