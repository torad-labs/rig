import { describe, expect, test } from "bun:test";
import { putHead } from "../../../test/fakes/head-fixtures.ts";
import { type FakePorts, fakePorts } from "../../../test/fakes/index.ts";
import type { Engine } from "../../shared/engine/engine.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { ExitCode } from "../../shared/result.ts";
import {
  CheckMachine,
  ENGINE_PEAK_BYTES,
  PREBUILT_TOOLS,
  REQUIRED_TOOLS,
} from "./machine-check.service.ts";

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
  prebuiltFor: (c: string) =>
    c === "120" ? { cap: "120", url: "u", sha256: "s", glibc: "2.35" } : undefined,
} as unknown as Engine;

/** the engine with CUDA 13.2.1's compiler known to build its kernels wrong for sm_120 */
const miscompiled = {
  ...engine,
  miscompilers: [{ nvcc: "13.2.78", caps: ["120"], why: "the IQ matmuls compute wrong" }],
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
  test("a compiler engine.toml knows to build the card's kernels wrong is refused before any download, by name", async () => {
    const p = ready();
    p.gpu.toolkit = "13.2.78";
    const r = await new CheckMachine(p, miscompiled).run({ gpu: 0, uid: 1000 });
    expect(!r.ok && r.code).toBe(ExitCode.Failure);
    expect(!r.ok && r.message).toBe(
      "nvcc 13.2.78 compiles this engine wrong for sm_120: the IQ matmuls compute wrong",
    );
  });
  test("the same compiler passes for a card it is not known to miscompile, and a fixed build passes for the card", async () => {
    const p = ready();
    p.gpu.toolkit = "13.2.78";
    p.gpu.card(0, { computeCap: "90" });
    expect((await new CheckMachine(p, miscompiled).run({ gpu: 0, uid: 1000 })).ok).toBe(true);
    const fixed = ready();
    fixed.gpu.toolkit = "13.2.86";
    expect((await new CheckMachine(fixed, miscompiled).run({ gpu: 0, uid: 1000 })).ok).toBe(true);
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
  test("a prebuilt is not compiled here, so the machine's miscompiling toolkit does not matter", async () => {
    const p = driverOnly();
    p.gpu.toolkit = "13.2.78";
    const withBoth = {
      ...withPrebuilt,
      miscompilers: miscompiled.miscompilers,
    } as unknown as Engine;
    const r = await new CheckMachine(p, withBoth).run({ gpu: 0, uid: 1000 });
    expect(r.ok && r.value).toMatchObject({ prebuilt: true, toolkitCuda: "13.2.78" });
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
  test("below the prebuilt's glibc the card compiles: the toolchain is required and the reason named, in the report or the exit-1 message", async () => {
    const p = driverOnly();
    p.host.libc = "2.31";
    const r = await new CheckMachine(p, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(!r.ok && r.code).toBe(ExitCode.Failure);
    expect(!r.ok && r.message).toContain("git, cmake, ninja");
    expect(!r.ok && r.message).toContain(
      "glibc 2.31 on this machine; the published sm_120 build needs glibc 2.35 or newer",
    );
    const full = ready();
    full.host.libc = "2.31";
    const compiles = await new CheckMachine(full, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(compiles.ok && compiles.value).toMatchObject({
      prebuilt: false,
      noPrebuilt:
        "glibc 2.31 on this machine; the published sm_120 build needs glibc 2.35 or newer",
    });
    const at = driverOnly();
    at.host.libc = "2.35"; // the floor itself is enough
    const atFloor = await new CheckMachine(at, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(atFloor.ok && atFloor.value.prebuilt).toBe(true);
    expect(atFloor.ok && "noPrebuilt" in atFloor.value).toBe(false);
  });
  test("a card with no prebuilt still needs the whole toolchain", async () => {
    const p = driverOnly();
    p.gpu.card(0, { computeCap: "90", name: "NVIDIA H100" });
    const r = await new CheckMachine(p, withPrebuilt).run({ gpu: 0, uid: 1000 });
    expect(!r.ok && r.message).toContain("git, cmake, ninja");
    expect(!r.ok && r.message).toContain("cuda toolkit (nvcc)");
  });
});

const headToml = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();
const PACK = 7_657_489_728;
const DRAFT_HEAD = 451_320_896;

/** the engine above, publishing its builds under local/engine-builds */
const building = {
  ...engine,
  binDir: (cap: string) => `/r/local/engine-builds/c008fe8-sm${cap}`,
} as unknown as Engine;

/** the real head as a stranger's clone sees it (no private adapter, nothing fetched), or as a
 *  Torad machine does, every [derive] asset in place */
async function bonsai(p: FakePorts, assets = false) {
  if (assets) putHead(p.fs, "/r", headToml);
  else p.fs.put("/r/heads/bonsai-2-27b/head.toml", headToml);
  const head = await loadHead(p.fs, layoutAt("/r"), "bonsai-2-27b");
  if (!head.ok) throw new Error(head.message);
  return head.value;
}

describe("room, before the first byte is fetched", () => {
  test("a card below the head's smallest tier is refused with exit 3, naming the card and the tier", async () => {
    const p = ready();
    p.gpu.card(0, { name: "NVIDIA GeForce RTX 5070", memoryMiB: 12227 });
    const r = await new CheckMachine(p, building).room(await bonsai(p), { gpu: 0 });
    expect(!r.ok && r.code).toBe(ExitCode.Unsupported);
    expect(!r.ok && r.message).toBe(
      "NVIDIA GeForce RTX 5070 at index 0 cannot serve bonsai-2-27b: 12227 MiB of VRAM for it is below the smallest tier this head declares (13500 MiB)",
    );
  });
  test("what other processes hold is not the head's: a 16 GB card whose desktop keeps 4 GB is refused, naming the share", async () => {
    const p = ready();
    p.gpu.card(0, { name: "NVIDIA GeForce RTX 5070 Ti", usedMiB: 4000 });
    const r = await new CheckMachine(p, building).room(await bonsai(p), { gpu: 0 });
    expect(!r.ok && r.code).toBe(ExitCode.Unsupported);
    expect(!r.ok && r.message).toBe(
      "NVIDIA GeForce RTX 5070 Ti at index 0 cannot serve bonsai-2-27b (4000 of its 16303 MiB are held by other processes: a desktop, another model): 12303 MiB of VRAM for it is below the smallest tier this head declares (13500 MiB)",
    );
  });
  test("a card driving a desktop gets the tier that fits beside it; the head already serving keeps its own share", async () => {
    const p = ready();
    p.fs.free = Number.MAX_SAFE_INTEGER;
    p.gpu.card(0, { usedMiB: 2318 }); // the 5070 Ti here: compositor, browsers, editors
    const desktop = await new CheckMachine(p, building).room(await bonsai(p), { gpu: 0 });
    expect(desktop.ok && desktop.value.tier.min_vram_mib).toBe(13700);
    p.gpu.card(0, { usedMiB: 13796 }); // the head itself serving on :8099 holds all but 12 of it
    p.host.listeners.set(8099, 3919564);
    p.gpu.held.set(3919564, 13784);
    const serving = await new CheckMachine(p, building).room(await bonsai(p), { gpu: 0 });
    expect(serving.ok && serving.value.tier.min_vram_mib).toBe(16000);
  });
  test("a fresh clone needs the source pack, our draft head, the public pack it derives and the engine: one byte short is refused", async () => {
    const p = ready();
    const head = await bonsai(p);
    const total = PACK + DRAFT_HEAD + PACK + ENGINE_PEAK_BYTES;
    p.fs.free = total;
    const r = await new CheckMachine(p, building).room(head, { gpu: 0 });
    expect(r.ok && r.value.tier.min_vram_mib).toBe(16000);
    expect(r.ok && r.value.needs).toEqual([
      { what: "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf", bytes: PACK },
      { what: "bonsai-2-27b-mtp-r2.gguf", bytes: DRAFT_HEAD },
      { what: "Ternary-Bonsai-2-27B-PQ2_0-MTP-r2.gguf", bytes: PACK },
      { what: "the engine", bytes: ENGINE_PEAK_BYTES },
    ]);
    p.fs.free = total - 1;
    const short = await new CheckMachine(p, building).room(head, { gpu: 0 });
    expect(!short.ok && short.code).toBe(ExitCode.Failure);
    p.fs.free = 5e9;
    const full = await new CheckMachine(p, building).room(head, { gpu: 0 });
    expect(!full.ok && full.message).toBe(
      "not enough disk under /r/local/packs/bonsai-2-27b: bonsai-2-27b still needs 17.4 GB (Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf 7.7 GB, bonsai-2-27b-mtp-r2.gguf 0.5 GB, Ternary-Bonsai-2-27B-PQ2_0-MTP-r2.gguf 7.7 GB, the engine 1.6 GB) and 5.0 GB is free",
    );
  });
  test("what is there is not counted, and a resumed download or a restarted bake counts only what it still lacks", async () => {
    const p = ready();
    const head = await bonsai(p, true);
    p.fs.put(`${head.sourcePath}.part`, "partial");
    p.fs.put(`${head.servedPath}.deriving`, "a stale bake");
    p.fs.put("/r/local/engine-builds/c008fe8-sm120/BUILD", "c008fe8");
    p.fs.put("/r/local/engine-builds/c008fe8-sm120/llama-server", "elf");
    p.fs.free = 0;
    const r = await new CheckMachine(p, building).room(head, { gpu: 0 });
    expect(!r.ok && r.message).toContain(
      "still needs 15.3 GB (Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf 7.7 GB, Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf 7.7 GB)",
    );
    p.fs.free = PACK - "partial".length + PACK - "a stale bake".length;
    const r2 = await new CheckMachine(p, building).room(head, { gpu: 0 });
    expect(r2.ok && r2.value.needs.map((need) => need.bytes)).toEqual([
      PACK - "partial".length,
      PACK - "a stale bake".length,
    ]);
    p.fs.put(head.sourcePath, "the pack");
    p.fs.put(head.servedPath, "the served pack");
    const present = await new CheckMachine(p, building).room(head, { gpu: 0 });
    expect(present.ok && present.value.needs).toEqual([]);
  });
});
