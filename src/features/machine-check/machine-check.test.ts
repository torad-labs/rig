import { describe, expect, test } from "bun:test";
import { fakePorts } from "../../../test/fakes/index.ts";
import type { Engine } from "../../shared/engine/engine.ts";
import { ExitCode } from "../../shared/result.ts";
import { CheckMachine, PREBUILT_TOOLS, REQUIRED_TOOLS } from "./machine-check.service.ts";

const engine = {
  archs: [
    { cap: "120", cards: "GB20x", evidence: "" },
    { cap: "90", cards: "H100", evidence: "" },
  ],
  supports: (c: string) => c === "120" || c === "90",
  prebuiltFor: () => undefined,
} as unknown as Engine;

/** the same engine with a prebuilt published for sm_120 on the CUDA 13.3 runtime */
const withPrebuilt = {
  ...engine,
  cuda: { version: "13.3", runtime: [] },
  prebuiltFor: (c: string) => (c === "120" ? { cap: "120", url: "u", sha256: "s" } : undefined),
} as unknown as Engine;

/** a driver-only machine: nvidia-smi, curl, tar, xz; no git, cmake, ninja or toolkit */
function driverOnly() {
  const p = fakePorts();
  for (const t of PREBUILT_TOOLS) p.shell.tools.add(t);
  p.gpu.toolkit = null;
  return p;
}

function ready() {
  const p = fakePorts();
  for (const t of REQUIRED_TOOLS) p.shell.tools.add(t);
  return p;
}

describe("prepare", () => {
  test("exit 0 on a supported card with every tool and a driver at least as new as the toolkit", async () => {
    const p = ready();
    const r = await new CheckMachine(p, engine).run({ gpu: 0, uid: 1000 });
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.value.supported).toBe(true);
      expect(r.value.installed).toBe(false);
    }
  });
  test("exit 1 names every missing tool", async () => {
    const p = ready();
    p.shell.tools.delete("cmake");
    p.shell.tools.delete("ninja");
    const r = await new CheckMachine(p, engine).run({ gpu: 0, uid: 1000 });
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.code).toBe(ExitCode.Failure);
      expect(r.message).toContain("cmake, ninja");
    }
  });
  test("exit 3 on a card the engine is not measured on, before any download", async () => {
    const p = ready();
    p.gpu.card(0, { computeCap: "89", name: "NVIDIA GeForce RTX 4090" });
    const r = await new CheckMachine(p, engine).run({ gpu: 0, uid: 1000 });
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.code).toBe(ExitCode.Unsupported);
      expect(r.message).toContain("sm_89");
      expect(r.message).toContain("--allow-arch 89");
    }
  });
  test("--allow-arch lets an unmeasured card through for a benchmark", async () => {
    const p = ready();
    p.gpu.card(0, { computeCap: "89" });
    const r = await new CheckMachine(p, engine).run({ gpu: 0, uid: 1000, allowArch: "89" });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.supported).toBe(false);
  });
  test("exit 4 when the driver's CUDA major is below the toolkit's", async () => {
    const p = ready();
    p.gpu.driver = "12.8";
    p.gpu.toolkit = "13.0";
    const r = await new CheckMachine(p, engine).run({ gpu: 0, uid: 1000 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.code).toBe(ExitCode.Driver);
  });
  test("a root box with apt installs the toolchain first", async () => {
    const p = ready();
    p.shell.tools.add("apt-get");
    p.shell.on(/^apt-get/, { code: 0, stdout: "", stderr: "" });
    const r = await new CheckMachine(p, engine).run({ gpu: 0, uid: 0 });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.installed).toBe(true);
    expect(p.shell.calls.filter((c) => c[0] === "apt-get").length).toBe(2);
  });
});

describe("prepare, a card with a published prebuilt", () => {
  test("a driver-only machine is ready: no compiler and no toolkit", async () => {
    const p = driverOnly();
    const r = await new CheckMachine(p, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(r.ok && r.value).toMatchObject({ supported: true, prebuilt: true, toolkitCuda: null });
  });
  test("it still needs what fetches and unpacks the build: xz missing is named", async () => {
    const p = driverOnly();
    p.shell.tools.delete("xz");
    const r = await new CheckMachine(p, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(!r.ok && r.message).toBe("missing on this machine: xz");
  });
  test("exit 4 when the driver cannot run the prebuilt's CUDA runtime, toolkit or not", async () => {
    const p = driverOnly();
    p.gpu.driver = "12.8";
    const r = await new CheckMachine(p, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(!r.ok && r.code).toBe(ExitCode.Driver);
    expect(!r.ok && r.message).toContain("CUDA 13.3 runtime");
  });
  test("a root box with apt installs what fetches and unpacks the prebuilt, not the toolchain", async () => {
    const p = driverOnly();
    p.shell.tools.delete("xz");
    p.shell.tools.add("apt-get");
    p.shell.on(/^apt-get install/, (cmd) => {
      p.shell.tools.add("xz");
      return { code: 0, stdout: cmd.join(" "), stderr: "" };
    });
    p.shell.on(/^apt-get update/, { code: 0, stdout: "", stderr: "" });
    const r = await new CheckMachine(p, withPrebuilt).run({ gpu: 0, uid: 0 });
    expect(r.ok && r.value).toMatchObject({ installed: true, prebuilt: true });
    const install = p.shell.calls.find((c) => c[1] === "install")!;
    expect(install).toContain("xz-utils");
    expect(install).not.toContain("cmake");
  });
  test("a card with no prebuilt still needs the whole toolchain", async () => {
    const p = driverOnly();
    p.gpu.card(0, { computeCap: "90", name: "NVIDIA H100" });
    const r = await new CheckMachine(p, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(!r.ok && r.message).toContain("git, cmake, ninja");
    expect(!r.ok && r.message).toContain("cuda toolkit (nvcc)");
  });
});
