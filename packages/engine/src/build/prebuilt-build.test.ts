import { describe, expect, test } from "bun:test";
import { ExitCode, layoutAt } from "@rig/core";
import { fakePorts, repoRoot } from "@rig/testing";
import { loadEngine } from "../engine.ts";
import { BuildPrebuilt } from "./prebuilt-build.service.ts";

const engineToml = await Bun.file(`${repoRoot}/engine/engine.toml`).text();
const sha256 = (text: string) => new Bun.CryptoHasher("sha256").update(text).digest("hex");
const OTHER = "3d40ae99ca0b9becce1b91134ad2bdd5f8aadf26";
const DOCKERFILE = "FROM ubuntu:22.04\n";

async function setup() {
  const p = fakePorts();
  p.gpu.card(1, { name: "NVIDIA GeForce RTX 5070 Ti" });
  const layout = layoutAt("/r");
  p.fs.put("/r/engine/engine.toml", engineToml);
  p.fs.put("/r/tools/prebuilt/Dockerfile", DOCKERFILE);
  const engine = await loadEngine(p.fs, layout);
  if (!engine.ok) throw new Error(engine.message);
  p.shell.on(/^bun run build$/, { code: 0, stdout: "", stderr: "" });
  // the build in the container leaves the tarball its engine.toml names under the writable mount
  p.containers.on(/^\/rig\/dist\/rig build/, () => {
    // the engine.toml this run mounted over the container's own
    const mounts = Object.entries(p.containers.runs.at(-1)?.options?.mounts ?? {});
    const mounted = mounts.find(([, at]) => at === "/rig/engine/engine.toml")?.[0];
    const pinned = (mounted && p.fs.text(mounted)) || engineToml;
    const sha7 = /^sha = "([0-9a-f]{7})/m.exec(pinned)?.[1];
    p.fs.put(`/r/local/prebuilt/engine-builds/engine-sm120-${sha7}.tar.gz`, "tgz");
    return { code: 0, stdout: "built", stderr: "" };
  });
  return { p, uc: new BuildPrebuilt(p, layout, engine.value), engine: engine.value };
}

describe("build --prebuilt", () => {
  test("builds the pin with --portable in tools/prebuilt's image, as the caller, capped at 14 GiB and 6 CPUs", async () => {
    const { p, uc, engine } = await setup();
    const r = await uc.run({ gpu: 1, jobs: 6 });
    const image = `rig-prebuilt:${sha256(DOCKERFILE).slice(0, 12)}`;
    expect(p.containers.builds).toEqual([{ context: "/r/tools/prebuilt", tag: image, labels: {} }]);
    expect(p.shell.calls).toContainEqual(["bun", "run", "build"]); // the binary the container runs
    expect(p.containers.runs).toEqual([
      {
        image,
        cmd: ["/rig/dist/rig", "build", "--gpu", "0", "--portable", "--jobs", "6"],
        options: {
          gpu: 1,
          asCaller: true,
          limits: { memory: "14g", cpus: 6 },
          env: { HOME: "/tmp", RIG_ROOT: "/rig" },
          mounts: { "/r": "/rig" },
          writable: { "/r/local/prebuilt": "/rig/local" },
          timeoutMs: expect.any(Number),
        },
      },
    ]);
    expect(r.ok && r.value.tarball).toBe(
      `/r/local/prebuilt/engine-builds/engine-sm120-${engine.sha7}.tar.gz`,
    );
    expect(await p.fs.exists("/r/local/prebuilt/engine.toml")).toBe(false);
  });
  test("--sha builds that commit through a copy of engine.toml naming it, without the pin's [[prebuilt]]", async () => {
    const { p, uc } = await setup();
    const r = await uc.run({ gpu: 0, jobs: 4, sha: OTHER });
    const copy = p.fs.text("/r/local/prebuilt/engine-3d40ae9.toml") ?? "";
    expect(copy).toContain(`\nsha = "${OTHER}"`);
    expect(copy.split("\n")).not.toContain("[[prebuilt]]");
    expect(copy.endsWith("\n")).toBe(engineToml.endsWith("\n")); // [[prebuilt]] is the last section
    // everything else is engine.toml's, line for line
    const kept = engineToml.split("\n").filter((line) => !/^sha = "[0-9a-f]{40}"/.test(line));
    for (const line of copy.split("\n").filter((l) => !l.startsWith("sha = ")))
      expect(kept).toContain(line);
    expect(p.containers.runs[0]?.options?.mounts).toEqual({
      "/r": "/rig",
      "/r/local/prebuilt/engine-3d40ae9.toml": "/rig/engine/engine.toml",
    });
    expect(p.containers.runs[0]?.cmd).toContain("4");
    expect(r.ok && r.value.tarball).toBe(
      "/r/local/prebuilt/engine-builds/engine-sm120-3d40ae9.tar.gz",
    );
  });
  test("builds of two commits at once each keep their own copy of engine.toml, so neither reads the other's pin", async () => {
    const { p, uc } = await setup();
    const THIRD = "9900ebcaf0de42c1b29f39ae7fe3ca87113c914d";
    const [a, b] = await Promise.all([
      uc.run({ gpu: 0, jobs: 6, sha: OTHER }),
      uc.run({ gpu: 1, jobs: 6, sha: THIRD }),
    ]);
    expect(a.ok && b.ok).toBe(true);
    expect(p.fs.text("/r/local/prebuilt/engine-3d40ae9.toml")).toContain(`\nsha = "${OTHER}"`);
    expect(p.fs.text("/r/local/prebuilt/engine-9900ebc.toml")).toContain(`\nsha = "${THIRD}"`);
    expect(await p.fs.exists("/r/local/prebuilt/engine.toml")).toBe(false);
    expect(p.containers.runs.map((run) => Object.keys(run.options?.mounts ?? {}).sort())).toEqual([
      ["/r", "/r/local/prebuilt/engine-3d40ae9.toml"],
      ["/r", "/r/local/prebuilt/engine-9900ebc.toml"],
    ]);
  });
  test("a --sha that is not a full 40-hex commit is a usage error before any image is built", async () => {
    const { p, uc } = await setup();
    const r = await uc.run({ gpu: 0, jobs: 6, sha: "3d40ae9" });
    expect(!r.ok && r.code).toBe(ExitCode.Usage);
    expect(p.containers.builds).toEqual([]);
  });
  test("a build that fails in the container, or leaves no tarball, is a failure", async () => {
    const failing = await setup();
    failing.p.containers.on(/^\/rig\/dist\/rig build/, {
      code: 1,
      stdout: "",
      stderr: "nvcc died",
    });
    const r = await failing.uc.run({ gpu: 0, jobs: 6 });
    expect(!r.ok && r.message).toContain("nvcc died");
    const empty = await setup();
    empty.p.containers.on(/^\/rig\/dist\/rig build/, { code: 0, stdout: "", stderr: "" });
    const none = await empty.uc.run({ gpu: 0, jobs: 6 });
    expect(!none.ok && none.message).toContain("engine-sm120-");
  });
});
