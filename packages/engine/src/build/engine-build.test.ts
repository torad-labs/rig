import { describe, expect, test } from "bun:test";
import { ExitCode, layoutAt, type RunOptions } from "@rig/core";
import { fakePorts, repoRoot } from "@rig/testing";
import { loadEngine, TARGETS } from "../engine.ts";
import { BuildEngine } from "./engine-build.service.ts";

const engineToml = await Bun.file(`${repoRoot}/engine/engine.toml`).text();
// the pin every expected path below is built from, read once from engine.toml like the use case does
const engine = await (async () => {
  const p = fakePorts();
  p.fs.put("/r/engine/engine.toml", engineToml);
  const e = await loadEngine(p.fs, layoutAt("/r"));
  if (!e.ok) throw new Error(e.message);
  return e.value;
})();

/** engine.toml without its closing [cuda] / [[prebuilt]] section: the engine with nothing
 *  published for the card, which compiles */
const compiledToml = engineToml.replace(/^\[cuda\][\s\S]*$/m, "");

const GOMP = "/usr/lib/gcc/x86_64-linux-gnu/15/libgomp.so.1";

async function setup(toml = compiledToml) {
  const p = fakePorts();
  const layout = layoutAt("/r");
  p.fs.put("/r/engine/engine.toml", toml);
  const engine = await loadEngine(p.fs, layout);
  if (!engine.ok) throw new Error(engine.message);
  p.git.heads.set("/r/engine/llama.cpp", engine.value.fork.sha);
  await p.fs.mkdirp("/r/engine/llama.cpp"); // the submodule is checked out at the pin
  p.shell.on(/^cmake -S/, { code: 0, stdout: "configured", stderr: "" });
  p.shell.on(/cmake --build (\S+)/, (cmd) => {
    // "compile": drop the targets into <bld>/bin
    const bld = cmd[cmd.indexOf("--build") + 1]!;
    for (const t of TARGETS) p.fs.put(`${bld}/bin/${t}`, `elf ${t}`);
    p.fs.put(`${bld}/bin/libllama-server-impl.so`, "so");
    return { code: 0, stdout: "built", stderr: "" };
  });
  p.shell.on(/^find /, { code: 0, stdout: "", stderr: "" });
  p.shell.on(/^ldd -r /, { code: 0, stdout: "\tlinux-vdso.so.1 (0x00007ffd)\n", stderr: "" });
  // the compiler's OpenMP runtime, which a portable build ships beside the targets
  p.fs.put(GOMP, "gomp");
  p.shell.on(/^c\+\+ -print-file-name=libgomp\.so\.1$/, {
    code: 0,
    stdout: `${GOMP}\n`,
    stderr: "",
  });
  const uc = new BuildEngine(p, layout, engine.value);
  return { p, layout, engine: engine.value, uc };
}

describe("build", () => {
  test("publishes local/engine-builds/<sha7>-sm<cap>/ by one rename with the marker written last", async () => {
    const { p, uc } = await setup();
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value.dir).toBe(`/r/local/engine-builds/${engine.sha7}-sm120`);
    expect(p.fs.text(`/r/local/engine-builds/${engine.sha7}-sm120/BUILD`)).toContain(
      `fork=${engine.fork.sha} cap=sm_120 native=on`,
    );
    expect(p.fs.renames.at(-1)).toEqual([
      `/r/local/engine-builds/${engine.sha7}-sm120.tmp-${process.pid}`,
      `/r/local/engine-builds/${engine.sha7}-sm120`,
    ]);
    expect(
      p.shell.calls.some(
        (c) => c.includes("-DGGML_CCACHE=OFF") && c.includes("-DCMAKE_CUDA_ARCHITECTURES=120a"),
      ),
    ).toBe(true);
  });
  test("is a no-op when the directory is complete, and not when only the launcher stub exists", async () => {
    const { p, uc } = await setup();
    p.fs.put(`/r/local/engine-builds/${engine.sha7}-sm120/llama-server`, "stub"); // half-copied: no marker
    const first = await uc.run({ gpu: 0 });
    expect(first.ok && !first.value.alreadyBuilt).toBe(true);
    const second = await uc.run({ gpu: 0 });
    expect(second.ok && second.value.alreadyBuilt).toBe(true);
  });
  test("refuses an unmeasured card with exit 3 and never configures", async () => {
    const { p, uc } = await setup();
    p.gpu.card(0, { computeCap: "89" });
    const r = await uc.run({ gpu: 0 });
    expect(!r.ok && r.code === ExitCode.Unsupported).toBe(true);
    expect(p.shell.calls.some((c) => c[0] === "cmake")).toBe(false);
  });
  test("builds through buildgate when it is on PATH, with an explicit -j", async () => {
    const { p, uc } = await setup();
    p.shell.tools.add("buildgate");
    await uc.run({ gpu: 0, jobs: 6 });
    const build = p.shell.calls.find((c) => c.includes("--build"))!;
    expect(build[0]).toBe("buildgate");
    expect(build).toContain("-j6");
  });
  test("a missing target is a failure and leaves no published directory", async () => {
    const { p, uc } = await setup();
    p.shell.on(/cmake --build (\S+)/, (cmd) => {
      p.fs.put(`${cmd[cmd.indexOf("--build") + 1]}/bin/llama-server`, "x");
      return { code: 0, stdout: "", stderr: "" };
    });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("llama-bench is missing");
    expect(await p.fs.exists(`/r/local/engine-builds/${engine.sha7}-sm120`)).toBe(false);
  });
  test("a target whose symbols do not resolve is refused by name and leaves no published directory", async () => {
    const { p, uc } = await setup();
    const missing =
      "undefined symbol: _Z14mul_mat_q_caseIL9ggml_type142EEvR25ggml_backend_cuda_contextRK8mmq_argsP11CUstream_st\t(/r/libggml-cuda.so.0)";
    p.shell.on(/^ldd -r \S+\/llama-server$/, { code: 0, stdout: "", stderr: missing });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.message).toContain("llama-server does not resolve");
      expect(r.message).toContain("mul_mat_q_case");
    }
    expect(await p.fs.exists(`/r/local/engine-builds/${engine.sha7}-sm120`)).toBe(false);
    expect(
      await p.fs.exists(`/r/local/engine-builds/${engine.sha7}-sm120.tmp-${process.pid}`),
    ).toBe(false);
  });
  test("empty objects a cut-short build left are pruned from the tree before it builds", async () => {
    const { p, uc } = await setup();
    await uc.run({ gpu: 0 });
    const tree = `/r/local/engine-build-trees/${engine.sha7}-sm120`;
    const prune = p.shell.calls.findIndex((c) => c[0] === "find");
    const build = p.shell.calls.findIndex((c) => c.includes("--build"));
    expect(p.shell.calls[prune]).toEqual([
      "find",
      tree,
      "-type",
      "f",
      "(",
      "-name",
      "*.o",
      "-o",
      "-name",
      "*.a",
      "-o",
      "-name",
      "*.so*",
      ")",
      "-size",
      "0",
      "-print",
      "-delete",
    ]);
    expect(prune).toBeGreaterThan(-1);
    expect(prune).toBeLessThan(build);
  });
  test("a prune that fails refuses the build instead of compiling over the empty outputs", async () => {
    const { p, uc } = await setup();
    p.shell.on(/^find /, { code: 1, stdout: "", stderr: "find: 'x.o': Permission denied" });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("Permission denied");
    expect(p.shell.calls.some((c) => c.includes("--build"))).toBe(false);
  });
  test("symbols are resolved against the staged libraries, not an inherited LD_LIBRARY_PATH", async () => {
    const { p, uc } = await setup();
    // the real loader: LD_LIBRARY_PATH outranks the $ORIGIN runpath, so a check that inherits a
    // path naming another build resolves that build's libraries instead of these
    p.shell.on(/^ldd -r /, (cmd, opts) => {
      const staged = cmd[2]!.slice(0, cmd[2]!.lastIndexOf("/"));
      return opts?.env?.LD_LIBRARY_PATH === staged
        ? {
            code: 0,
            stdout: "",
            stderr: "undefined symbol: ggml_backend_cuda_init\t(libggml-cuda.so)",
          }
        : { code: 0, stdout: "\tlibggml-cuda.so => /elsewhere/libggml-cuda.so\n", stderr: "" };
    });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("ggml_backend_cuda_init");
  });
  test("--from-tarball refuses the wrong arch by name and publishes the right one with a marker", async () => {
    const { p, uc } = await setup();
    p.fs.put(`/r/local/engine-builds/engine-sm90-${engine.sha7}.tar.gz`, "tgz");
    const wrong = await uc.run({
      gpu: 0,
      fromTarball: `/r/local/engine-builds/engine-sm90-${engine.sha7}.tar.gz`,
    });
    expect(!wrong.ok && wrong.message.includes("not the tarball for sm_120")).toBe(true);
    p.fs.put(`/r/local/engine-builds/engine-sm120-${engine.sha7}.tar.gz`, "tgz");
    p.shell.on(/^tar -C (\S+) -xzf/, (cmd) => {
      for (const t of TARGETS) p.fs.put(`${cmd[2]}/${t}`, t);
      return { code: 0, stdout: "", stderr: "" };
    });
    const right = await uc.run({
      gpu: 0,
      fromTarball: `/r/local/engine-builds/engine-sm120-${engine.sha7}.tar.gz`,
    });
    expect(right.ok).toBe(true);
    if (right.ok)
      expect(right.value.marker).toContain(`source=tarball:engine-sm120-${engine.sha7}.tar.gz`);
  });
  test("fetches the pinned commit under local/engine-sources when the submodule is not at the pin", async () => {
    const { p, uc, engine } = await setup();
    p.git.heads.delete("/r/engine/llama.cpp");
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(true);
    expect(p.git.fetched).toEqual([
      {
        dir: `/r/local/engine-sources/llama.cpp-${engine.sha7}`,
        repo: engine.fork.repo,
        sha: engine.fork.sha,
      },
    ]);
  });
  test("a build tree whose CMakeCache was generated for another source or directory is discarded first", async () => {
    const { p, uc, layout } = await setup();
    const bld = `${layout.engineBuildTreesDir}/${engine.sha7}-sm120`;
    p.fs.put(
      `${bld}/CMakeCache.txt`,
      `CMAKE_HOME_DIRECTORY:INTERNAL=/elsewhere/llama.cpp\nCMAKE_CACHEFILE_DIR:INTERNAL=/old/build\n`,
    );
    p.fs.put(`${bld}/stale.o`, "x");
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(true);
    expect(await p.fs.exists(`${bld}/stale.o`)).toBe(false);
    expect(p.log.lines.join("\n")).toContain("discarding");
    // a cache made for this source in this directory, with the web UI off, is kept (an incremental build)
    p.fs.put(
      `${bld}/CMakeCache.txt`,
      `CMAKE_HOME_DIRECTORY:INTERNAL=/r/engine/llama.cpp\nCMAKE_CACHEFILE_DIR:INTERNAL=${bld}\nLLAMA_USE_PREBUILT_UI:BOOL=OFF\n`,
    );
    p.fs.put(`${bld}/keep.o`, "x");
    await p.fs.remove(`${layout.engineBuildsDir}/${engine.sha7}-sm120`);
    await uc.run({ gpu: 0 });
    expect(await p.fs.exists(`${bld}/keep.o`)).toBe(true);
  });
  test("a published build is self-contained: its libraries resolve from the directory it sits in", async () => {
    const { p, uc } = await setup();
    await uc.run({ gpu: 0 });
    const configure = p.shell.calls.find((c) => c[0] === "cmake" && c[1] === "-S")!;
    // a RUNPATH into the build tree makes the published directory — and the portable tarball —
    // depend on a path rig itself calls disposable; $ORIGIN makes it depend on nothing
    expect(configure).toContain("-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON");
    expect(configure).toContain("-DCMAKE_INSTALL_RPATH=$ORIGIN");
  });
  test("no compiler records the source's path: ggml's asserts print __FILE__", async () => {
    const { p, uc } = await setup();
    await uc.run({ gpu: 0 });
    const configure = p.shell.calls.find((c) => c[0] === "cmake" && c[1] === "-S")!;
    const map = "-ffile-prefix-map=/r/engine/llama.cpp=llama.cpp";
    expect(configure).toContain(`-DCMAKE_C_FLAGS=${map}`);
    expect(configure).toContain(`-DCMAKE_CXX_FLAGS=${map}`);
    expect(configure).toContain(`-DCMAKE_CUDA_FLAGS=-Xcompiler=${map}`);
  });
  test("the engine embeds no web UI, so no build reaches the network for one: the HF download is off, not only the npm build", async () => {
    const { p, uc } = await setup();
    await uc.run({ gpu: 0 });
    const configure = p.shell.calls.find((c) => c[0] === "cmake" && c[1] === "-S")!;
    // LLAMA_BUILD_UI=OFF alone leaves ui-assets.cmake's step 3 on: LLAMA_USE_PREBUILT_UI (default ON) downloads
    // whatever the bucket's "latest" is and embeds it, so no two builds of one commit were the same
    expect(configure).toContain("-DLLAMA_BUILD_UI=OFF");
    expect(configure).toContain("-DLLAMA_USE_PREBUILT_UI=OFF");
  });
  test("a build tree configured with the prebuilt UI on is discarded, with the reason: its downloaded assets would be embedded as stale", async () => {
    const { p, uc, layout } = await setup();
    const bld = `${layout.engineBuildTreesDir}/${engine.sha7}-sm120`;
    for (const entry of ["LLAMA_USE_PREBUILT_UI:BOOL=ON\n", ""]) {
      p.fs.put(
        `${bld}/CMakeCache.txt`,
        `CMAKE_HOME_DIRECTORY:INTERNAL=/r/engine/llama.cpp\nCMAKE_CACHEFILE_DIR:INTERNAL=${bld}\n${entry}`,
      );
      p.fs.put(`${bld}/tools/ui/dist/index.html`, "downloaded");
      await p.fs.remove(`${layout.engineBuildsDir}/${engine.sha7}-sm120`);
      p.log.lines.length = 0;
      expect((await uc.run({ gpu: 0 })).ok).toBe(true);
      expect(await p.fs.exists(`${bld}/tools/ui/dist/index.html`)).toBe(false);
      expect(p.log.lines.join("\n")).toContain("web UI");
    }
  });
  test("--portable links no OpenSSL (a newer host's would not load on an older one); native does, both say so", async () => {
    const configureOf = async (portable: boolean) => {
      const { p, uc } = await setup();
      p.shell.on(/^tar -C \S+ -czf/, { code: 0, stdout: "", stderr: "" });
      await uc.run({ gpu: 0, portable });
      return p.shell.calls.find((c) => c[0] === "cmake" && c[1] === "-S")!;
    };
    expect(await configureOf(true)).toContain("-DLLAMA_OPENSSL=OFF");
    expect(await configureOf(false)).toContain("-DLLAMA_OPENSSL=ON");
  });
  test("--portable packs the build without its marker (it names this host) or this machine's user", async () => {
    const { p, uc } = await setup();
    p.shell.on(/^tar -C \S+ -czf/, { code: 0, stdout: "", stderr: "" });
    const r = await uc.run({ gpu: 0, portable: true });
    expect(r.ok && r.value.tarball).toBe(
      `/r/local/engine-builds/engine-sm120-${engine.sha7}.tar.gz`,
    );
    const pack = p.shell.calls.find((c) => c.includes("-czf"))!;
    expect(pack).toContain("--exclude=./BUILD");
    expect(pack).toContain("--owner=0");
    expect(pack).toContain("--numeric-owner");
  });
  describe("--portable over a build already published here", () => {
    const dir = `/r/local/engine-builds/${engine.sha7}-sm120`;
    const tarball = `/r/local/engine-builds/engine-sm120-${engine.sha7}.tar.gz`;
    /** a build published here before this run: its marker, and `targets` */
    async function over(native: "on" | "off", targets: readonly string[], source = "compiled") {
      const s = await setup();
      s.p.shell.on(/^tar -C \S+ -czf/, { code: 0, stdout: "", stderr: "" });
      s.p.fs.put(
        `${dir}/BUILD`,
        `fork=${engine.fork.sha} cap=sm_120 native=${native} built=2026-10-01T19:43:36.779Z host=h source=${source}`,
      );
      for (const t of targets) s.p.fs.put(`${dir}/${t}`, `old ${t}`);
      return s;
    }
    test("one made before a target existed is compiled again, and the tarball holds every target", async () => {
      const { p, uc } = await over(
        "off",
        TARGETS.filter((t) => t !== "llama-perplexity"),
      );
      const r = await uc.run({ gpu: 0, portable: true });
      expect(r.ok ? { built: r.value.alreadyBuilt, tarball: r.value.tarball } : r.message).toEqual({
        built: false,
        tarball,
      });
      expect(p.shell.calls.some((c) => c[0] === "cmake")).toBe(true);
      for (const t of TARGETS) expect(p.fs.text(`${dir}/${t}`)).toBe(`elf ${t}`);
    });
    test("a native one is compiled again as the portable baseline", async () => {
      const { p, uc } = await over("on", TARGETS);
      const r = await uc.run({ gpu: 0, portable: true });
      expect(r.ok && r.value.tarball).toBe(tarball);
      expect(p.fs.text(`${dir}/BUILD`)).toContain("native=off");
    });
    test("an installed one is compiled again: it holds NVIDIA's runtime beside its libraries, which a prebuilt never carries", async () => {
      const { p, uc } = await over("off", TARGETS, `tarball:engine-sm120-${engine.sha7}.tar.gz`);
      p.fs.put(`${dir}/libcudart.so.13`, "so libcudart.so.13");
      const r = await uc.run({ gpu: 0, portable: true });
      expect(r.ok && r.value.tarball).toBe(tarball);
      expect(p.shell.calls.some((c) => c[0] === "cmake")).toBe(true);
      expect(await p.fs.exists(`${dir}/libcudart.so.13`)).toBe(false);
    });
    test("a complete portable one is packed again, compiling nothing: --portable always answers with its tarball", async () => {
      const { p, uc } = await over("off", TARGETS);
      const r = await uc.run({ gpu: 0, portable: true });
      expect(r.ok ? { built: r.value.alreadyBuilt, tarball: r.value.tarball } : r.message).toEqual({
        built: true,
        tarball,
      });
      expect(p.shell.calls.some((c) => c[0] === "cmake")).toBe(false);
      expect(p.shell.calls.some((c) => c.includes("-czf"))).toBe(true);
    });
  });
  test("a portable build ships the compiler's libgomp beside its targets; a native one does not", async () => {
    const portable = await setup();
    portable.p.shell.on(/^tar -C \S+ -czf/, { code: 0, stdout: "", stderr: "" });
    expect((await portable.uc.run({ gpu: 0, portable: true })).ok).toBe(true);
    expect(portable.p.fs.text(`/r/local/engine-builds/${engine.sha7}-sm120/libgomp.so.1`)).toBe(
      "gomp",
    );
    const native = await setup();
    expect((await native.uc.run({ gpu: 0 })).ok).toBe(true);
    expect(
      await native.p.fs.exists(`/r/local/engine-builds/${engine.sha7}-sm120/libgomp.so.1`),
    ).toBe(false);
  });
  test("the compiler named in CMakeCache is the one asked, and one that has no libgomp refuses the portable build", async () => {
    const { p, uc } = await setup();
    const tree = `/r/local/engine-build-trees/${engine.sha7}-sm120`;
    p.fs.put(
      `${tree}/CMakeCache.txt`,
      `CMAKE_HOME_DIRECTORY:INTERNAL=/r/engine/llama.cpp\nCMAKE_CACHEFILE_DIR:INTERNAL=${tree}\nCMAKE_CXX_COMPILER:FILEPATH=/opt/gcc/bin/g++\nLLAMA_USE_PREBUILT_UI:BOOL=OFF\n`,
    );
    p.shell.on(/^\/opt\/gcc\/bin\/g\+\+ -print-file-name=/, {
      code: 0,
      stdout: "libgomp.so.1\n",
      stderr: "",
    });
    const r = await uc.run({ gpu: 0, portable: true });
    expect(!r.ok && r.message).toContain("/opt/gcc/bin/g++ does not know where it is");
    expect(await p.fs.exists(`/r/local/engine-builds/${engine.sha7}-sm120`)).toBe(false);
  });
});

describe("build, a compiler engine.toml lists as miscompiling", () => {
  /** configure writes the cache naming the CUDA compiler it found, as cmake does */
  function configuresWith(p: Awaited<ReturnType<typeof setup>>["p"], nvcc: string) {
    const tree = `/r/local/engine-build-trees/${engine.sha7}-sm120`;
    p.shell.on(/^cmake -S/, () => {
      p.fs.put(
        `${tree}/CMakeCache.txt`,
        `CMAKE_HOME_DIRECTORY:INTERNAL=/r/engine/llama.cpp\nCMAKE_CACHEFILE_DIR:INTERNAL=${tree}\nCMAKE_CUDA_COMPILER:FILEPATH=${nvcc}\n`,
      );
      return { code: 0, stdout: "configured", stderr: "" };
    });
  }

  test("CUDA 13.2.1's nvcc is refused for sm_120 after configure, before anything compiles; the compiler asked is cmake's", async () => {
    const { p, uc } = await setup();
    configuresWith(p, "/usr/local/cuda-13.2/bin/nvcc");
    p.gpu.toolkit = "13.2.78";
    const r = await uc.run({ gpu: 0 });
    expect(!r.ok && r.message).toContain("nvcc 13.2.78 compiles this engine wrong for sm_120");
    expect(!r.ok && r.message).toContain("(the compiler cmake configured)");
    expect(p.gpu.toolkitAsked).toEqual(["/usr/local/cuda-13.2/bin/nvcc"]);
    expect(p.shell.calls.some((c) => c.includes("--build"))).toBe(false);
    expect(await p.fs.exists(`/r/local/engine-builds/${engine.sha7}-sm120`)).toBe(false);
  });
  test("13.2.2's nvcc builds for sm_120, and 13.2.1's builds for a card it is not known to miscompile", async () => {
    const fixed = await setup();
    configuresWith(fixed.p, "/usr/local/cuda-13.2/bin/nvcc");
    fixed.p.gpu.toolkit = "13.2.86";
    expect((await fixed.uc.run({ gpu: 0 })).ok).toBe(true);
    const h100 = await setup();
    h100.p.gpu.card(0, { computeCap: "90" });
    h100.p.gpu.toolkit = "13.2.78";
    expect((await h100.uc.run({ gpu: 0 })).ok).toBe(true);
  });
});

describe("build, a card with a published prebuilt", () => {
  const sha = (text: string) => new Bun.CryptoHasher("sha256").update(text).digest("hex");
  const tarballName = `engine-sm120-${engine.sha7}.tar.gz`;
  const bytes: Record<string, string> = {
    [tarballName]: "the portable build",
    "cuda_cudart-archive.tar.xz": "nvidia cudart",
    "libcublas-archive.tar.xz": "nvidia cublas",
  };
  /** a download's fixture bytes, by file name */
  const body = (name: string): string => {
    const text = bytes[name];
    if (text === undefined) throw new Error(`no fixture for ${name}`);
    return text;
  };
  const prebuiltToml = `${compiledToml}
[cuda]
version = "13.3"
[[cuda.runtime]]
url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/cuda_cudart-archive.tar.xz"
sha256 = "${sha(body("cuda_cudart-archive.tar.xz"))}"
libs = ["libcudart.so.13"]
[[cuda.runtime]]
url = "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/libcublas-archive.tar.xz"
sha256 = "${sha(body("libcublas-archive.tar.xz"))}"
libs = ["libcublas.so.13", "libcublasLt.so.13"]
[[prebuilt]]
cap = "120"
url = "https://github.com/torad-labs/llama.cpp/releases/download/engine-${engine.sha7}/${tarballName}"
sha256 = "${sha(body(tarballName))}"
glibc = "2.35"
`;
  const downloads = "/r/local/downloads";
  const dir = `/r/local/engine-builds/${engine.sha7}-sm120`;
  /** engine-sm120-32e695e.tar.gz as published on Sep 27, 2026 (sha256 9df73c20…, engine.toml's [[prebuilt]]): its
   *  members by `tar -tzf`, not TARGETS. A published build never changes, so it holds the targets of its day: this one
   *  predates llama-perplexity (f183b82), and v0.1.11 refused to install it on every fresh machine. */
  const PUBLISHED_32E695E = [
    "libggml-base.so.0.21.0",
    "libggml-base.so.0",
    "libggml-base.so",
    "libggml-cpu.so.0.21.0",
    "libggml-cpu.so.0",
    "libggml-cpu.so",
    "libggml-cuda.so.0.21.0",
    "libggml-cuda.so.0",
    "libggml-cuda.so",
    "libggml.so.0.21.0",
    "libggml.so.0",
    "libggml.so",
    "libllama.so.0.2.0",
    "libllama.so.0",
    "libllama.so",
    "libllama-common.so.0.2.0",
    "libmtmd.so.0.2.0",
    "libmtmd.so.0",
    "libmtmd.so",
    "libllama-common.so.0",
    "libllama-common.so",
    "llama-kv-mean-center",
    "libllama-bench-impl.so",
    "libllama-server-impl.so",
    "llama-bench",
    "llama-server",
    "libgomp.so.1",
  ];

  /** curl writes each file's bytes (`served` overrides one); tar unpacks the build (`members`, TARGETS unless
   *  given), and each runtime archive the libraries its wildcards name (`omit` leaves one out) */
  async function prebuiltSetup(
    o: { served?: Record<string, string>; omit?: string; members?: readonly string[] } = {},
  ) {
    const s = await setup(prebuiltToml);
    s.p.shell.on(/^curl/, (cmd) => {
      const out = cmd[cmd.indexOf("-o") + 1]!;
      const name = cmd.at(-1)!.split("/").at(-1)!;
      s.p.fs.put(out, o.served?.[name] ?? body(name));
      return { code: 0, stdout: "", stderr: "" };
    });
    s.p.shell.on(/^tar -C (\S+) -xzf/, (cmd) => {
      for (const t of o.members ?? TARGETS) s.p.fs.put(`${cmd[2]}/${t}`, `elf ${t}`);
      return { code: 0, stdout: "", stderr: "" };
    });
    s.p.shell.on(/^tar -C (\S+) -xJf/, (cmd) => {
      for (const member of cmd.filter((arg) => arg.startsWith("*/lib/"))) {
        const lib = member.slice("*/lib/".length, -1);
        if (lib !== o.omit) s.p.fs.put(`${cmd[2]}/${lib}`, `so ${lib}`);
      }
      return { code: 0, stdout: "", stderr: "" };
    });
    return s;
  }

  test("installs it: the build and NVIDIA's runtime by sha256, unpacked together, never compiled", async () => {
    const { p, uc } = await prebuiltSetup();
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(true);
    expect(p.shell.calls.some((c) => c[0] === "cmake")).toBe(false);
    expect(p.shell.calls.filter((c) => c[0] === "curl").map((c) => c.at(-1))).toEqual([
      `https://github.com/torad-labs/llama.cpp/releases/download/engine-${engine.sha7}/${tarballName}`,
      "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/cuda_cudart-archive.tar.xz",
      "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/libcublas-archive.tar.xz",
    ]);
    for (const lib of ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "llama-server"])
      expect(await p.fs.exists(`${dir}/${lib}`)).toBe(true);
    // the symbol check ran over the build with the runtime already beside it
    expect(p.shell.calls.some((c) => c[0] === "ldd")).toBe(true);
    expect(p.fs.text(`${dir}/BUILD`)).toContain(`source=tarball:${tarballName}`);
    for (const name of Object.keys(bytes))
      expect(await p.fs.exists(`${downloads}/${name}`)).toBe(false);
  });
  test("installs a build published before llama-perplexity was a target: an install needs the targets that serve", async () => {
    const { p, uc } = await prebuiltSetup({ members: PUBLISHED_32E695E });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok ? r.value.marker : r.message).toContain(`source=tarball:${tarballName}`);
    for (const target of ["llama-server", "llama-bench", "llama-kv-mean-center"])
      expect(await p.fs.exists(`${dir}/${target}`)).toBe(true);
    expect(await p.fs.exists(`${dir}/llama-perplexity`)).toBe(false);
    // the symbol check ran over the targets the tarball holds, and asked nothing of the one it lacks
    const checked = p.shell.calls
      .filter((c) => c[0] === "ldd")
      .map((c) => c.at(-1)!.split("/").at(-1));
    expect(checked).toContain("llama-server");
    expect(checked).not.toContain("llama-perplexity");
  });
  test("an install still refuses a tarball missing a target that serves, by name", async () => {
    const members = PUBLISHED_32E695E.filter((m) => m !== "llama-bench");
    const { p, uc } = await prebuiltSetup({ members });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toContain("llama-bench is missing");
    expect(await p.fs.exists(dir)).toBe(false);
  });
  test("--cap installs with no card present, and a tarball given by hand gets the pin's CUDA runtime beside it", async () => {
    const { p, uc } = await prebuiltSetup();
    p.gpu.cards.clear(); // an image build: no card
    const local = `/r/local/prebuilt/${tarballName}`;
    p.fs.put(local, "the portable build");
    const r = await uc.run({ gpu: 0, cap: "120", fromTarball: local });
    expect(r.ok && [r.value.cap, r.value.dir]).toEqual(["120", dir]);
    for (const lib of ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "llama-server"])
      expect(await p.fs.exists(`${dir}/${lib}`)).toBe(true);
    expect(
      p.shell.calls.filter((c) => c[0] === "curl").map((c) => c.at(-1)?.split("/").at(-1)),
    ).toEqual(["cuda_cudart-archive.tar.xz", "libcublas-archive.tar.xz"]); // the runtime only: the build is the file given
    expect(await p.fs.exists(local)).toBe(true); // the operator's file is kept; the downloads are not
    for (const name of ["cuda_cudart-archive.tar.xz", "libcublas-archive.tar.xz"])
      expect(await p.fs.exists(`${downloads}/${name}`)).toBe(false);
    const again = await uc.run({ gpu: 0, cap: "120" });
    expect(again.ok && again.value.alreadyBuilt).toBe(true); // published under the pin, a later build finds it
  });

  // a candidate's `--portable` output: another commit's build, which the lab A/Bs against the pin
  const candidate = "/r/local/candidates/engine-sm120-0123abc.tar.gz";
  const lab = "/r/local/engine-lab/cand";
  test("a tarball off the pin is refused without --off-pin, the refusal naming the flag, and nothing is published", async () => {
    const { p, uc } = await prebuiltSetup();
    p.fs.put(candidate, "a candidate build");
    const r = await uc.run({ gpu: 0, fromTarball: candidate });
    expect(!r.ok && r.message).toContain("--off-pin");
    expect(await p.fs.exists(dir)).toBe(false);
    expect(await p.fs.exists(lab)).toBe(false);
    // and where the pin is already installed: "already built" would answer for a build that was not asked for
    expect((await uc.run({ gpu: 0 })).ok).toBe(true);
    const installed = await uc.run({ gpu: 0, fromTarball: candidate });
    expect(!installed.ok && installed.message).toContain("--off-pin");
  });
  test("--off-pin installs it as local/engine-lab/<name>/ with the pin's CUDA runtime beside it, never as the pin's build", async () => {
    const { p, uc } = await prebuiltSetup();
    p.fs.put(candidate, "a candidate build");
    // a tarball packed by hand can carry its builder's marker, naming the pin
    p.shell.on(/^tar -C (\S+) -xzf/, (cmd) => {
      for (const t of TARGETS) p.fs.put(`${cmd[2]}/${t}`, `elf ${t}`);
      p.fs.put(`${cmd[2]}/BUILD`, `fork=${engine.fork.sha} cap=sm_120`);
      return { code: 0, stdout: "", stderr: "" };
    });
    // one named for another card is refused by its name, as the pin's are
    const sm90 = "/r/local/candidates/engine-sm90-0123abc.tar.gz";
    p.fs.put(sm90, "a candidate build for sm_90");
    const wrong = await uc.run({ gpu: 0, fromTarball: sm90, offPin: "cand" });
    expect(!wrong.ok && wrong.message).toContain("a build for sm_90, not this card's sm_120");
    const r = await uc.run({ gpu: 0, fromTarball: candidate, offPin: "cand" });
    expect(r.ok && [r.value.dir, r.value.alreadyBuilt]).toEqual([lab, false]);
    for (const lib of ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "llama-server"])
      expect(await p.fs.exists(`${lab}/${lib}`)).toBe(true);
    const marker = p.fs.text(`${lab}/BUILD`);
    expect(marker).toContain("fork=off-pin");
    expect(marker).not.toContain(engine.fork.sha);
    expect(marker).toContain(`sha256=${sha("a candidate build")}`);
    // the symbol check ran over it with the runtime beside it, as over an installed build
    expect(p.shell.calls.some((c) => c[0] === "ldd")).toBe(true);
    expect(await p.fs.exists(dir)).toBe(false); // where env resolution looks: never there
    expect(await p.fs.exists(candidate)).toBe(true);
    for (const name of ["cuda_cudart-archive.tar.xz", "libcublas-archive.tar.xz"])
      expect(await p.fs.exists(`${downloads}/${name}`)).toBe(false);
  });
  test("--off-pin again with the same tarball is a no-op, and another tarball under that name is refused", async () => {
    const { p, uc } = await prebuiltSetup();
    p.fs.put(candidate, "a candidate build");
    expect((await uc.run({ gpu: 0, fromTarball: candidate, offPin: "cand" })).ok).toBe(true);
    const again = await uc.run({ gpu: 0, fromTarball: candidate, offPin: "cand" });
    expect(again.ok && again.value.alreadyBuilt).toBe(true);
    const other = "/r/local/candidates/engine-sm120-4567def.tar.gz";
    p.fs.put(other, "another candidate");
    const clash = await uc.run({ gpu: 0, fromTarball: other, offPin: "cand" });
    expect(!clash.ok && clash.message).toContain(`${lab} exists`);
    expect(p.fs.text(`${lab}/BUILD`)).toContain(`sha256=${sha("a candidate build")}`);
  });
  test("--off-pin without --from-tarball, or naming a path and not a directory, is a usage error before anything runs", async () => {
    const { p, uc } = await prebuiltSetup();
    p.gpu.cards.clear(); // refused before the card is asked for
    p.fs.put(candidate, "a candidate build");
    const bare = await uc.run({ gpu: 0, offPin: "cand" });
    expect(!bare.ok && [bare.code, bare.message.includes("--from-tarball")]).toEqual([
      ExitCode.Usage,
      true,
    ]);
    for (const name of ["../engine-builds/x", "a/b", "..", ""]) {
      const r = await uc.run({ gpu: 0, fromTarball: candidate, offPin: name });
      expect(!r.ok && r.code).toBe(ExitCode.Usage);
    }
    expect(p.shell.calls.length).toBe(0);
  });
  test("a download that is not the pinned bytes is refused and nothing is published; what verified is kept for a retry", async () => {
    const { p, uc } = await prebuiltSetup({ served: { "libcublas-archive.tar.xz": "tampered" } });
    const r = await uc.run({ gpu: 0 });
    expect(!r.ok && r.message).toContain("not the pinned CUDA runtime libcublas-archive.tar.xz");
    expect(await p.fs.exists(dir)).toBe(false);
    expect(await p.fs.exists(`${downloads}/${tarballName}`)).toBe(true);
    expect(await p.fs.exists(`${downloads}/libcublas-archive.tar.xz`)).toBe(false);
  });
  test("a retry fetches only what is not already verified on disk", async () => {
    const { p, uc } = await prebuiltSetup();
    p.fs.put(`${downloads}/${tarballName}`, body(tarballName));
    expect((await uc.run({ gpu: 0 })).ok).toBe(true);
    expect(p.shell.calls.filter((c) => c[0] === "curl").length).toBe(2);
  });
  test("an archive without a library it is pinned for refuses the build, by name", async () => {
    const { p, uc } = await prebuiltSetup({ omit: "libcublasLt.so.13" });
    const r = await uc.run({ gpu: 0 });
    expect(!r.ok && r.message).toContain("libcublas-archive.tar.xz holds no lib/libcublasLt.so.13");
    expect(await p.fs.exists(dir)).toBe(false);
    expect(await p.fs.exists(`${dir}.tmp-${process.pid}`)).toBe(false);
  });
  test("a machine below the prebuilt's glibc compiles, and says why, instead of installing a build its ldd -r would refuse", async () => {
    const { p, uc } = await prebuiltSetup();
    p.host.libc = "2.31";
    expect((await uc.run({ gpu: 0 })).ok).toBe(true);
    expect(p.shell.calls.some((c) => c[0] === "cmake")).toBe(true);
    expect(p.shell.calls.some((c) => c[0] === "curl")).toBe(false);
    expect(p.log.lines.some((l) => l.includes("glibc 2.31 on this machine"))).toBe(true);
  });
  test("a machine with no NVIDIA driver is refused by the driver library it lacks, never as a stale object", async () => {
    // 2026-10-03, the v0.1.13 fresh install in a card-less ubuntu:22.04: libcuda.so.1 not found read "a stale or empty object"
    const { p, uc } = await prebuiltSetup();
    p.shell.on(/^ldd -r /, {
      code: 0,
      stdout: "\tlibcuda.so.1 => not found\n",
      stderr: "undefined symbol: cuMemCreate\t(/r/libggml-cuda.so.0)\n",
    });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.message).toContain("llama-server does not resolve: libcuda.so.1 not found");
    expect(r.message).toContain("NVIDIA driver");
    expect(r.message).not.toContain("stale or empty object");
    expect(await p.fs.exists(dir)).toBe(false);
  });
  test("a library missing from the build dir is named, never read as a stale object", async () => {
    const { p, uc } = await prebuiltSetup();
    p.shell.on(/^ldd -r /, {
      code: 0,
      stdout:
        "\tlibcublasLt.so.13 => not found\n\tlibcuda.so.1 => /usr/lib/x86_64-linux-gnu/libcuda.so.1\n",
      stderr: "undefined symbol: cublasLtMatmul\t(/r/libggml-cuda.so.0)\n",
    });
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.message).toContain("llama-server does not resolve: libcublasLt.so.13 not found");
    expect(r.message).not.toContain("NVIDIA driver");
    expect(r.message).not.toContain("stale or empty object");
  });
  test("--compile builds from source over a pinned prebuilt, and --portable (which makes one) always does", async () => {
    for (const options of [{ compile: true }, { portable: true }]) {
      const { p, uc } = await prebuiltSetup();
      p.shell.on(/^tar -C \S+ -czf/, { code: 0, stdout: "", stderr: "" }); // --portable packs it
      expect((await uc.run({ gpu: 0, ...options })).ok).toBe(true);
      expect(p.shell.calls.some((c) => c[0] === "cmake")).toBe(true);
      expect(p.shell.calls.some((c) => c[0] === "curl")).toBe(false);
    }
  });
});

describe("build, on a machine with no NVIDIA driver (a compile in a container with no card)", () => {
  const STUB = "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so";
  const tree = `/r/local/engine-build-trees/${engine.sha7}-sm120`;
  const NO_DRIVER = {
    code: 0,
    stdout: "\tlibcuda.so.1 => not found\n",
    stderr: "undefined symbol: cuMemCreate\t(/r/libggml-cuda.so.0)\n",
  };

  /** configure leaves the entry cmake's FindCUDAToolkit writes: the libcuda stub the engine links against */
  function configures(p: Awaited<ReturnType<typeof setup>>["p"], entries: string) {
    p.shell.on(/^cmake -S/, () => {
      p.fs.put(
        `${tree}/CMakeCache.txt`,
        `CMAKE_HOME_DIRECTORY:INTERNAL=/r/engine/llama.cpp\nCMAKE_CACHEFILE_DIR:INTERNAL=${tree}\n${entries}`,
      );
      return { code: 0, stdout: "configured", stderr: "" };
    });
  }
  /** ldd as the loader answers it, recording what each call preloaded */
  function loader(
    p: Awaited<ReturnType<typeof setup>>["p"],
    answer: (preload: string | undefined) => { code: number; stdout: string; stderr: string },
  ) {
    const preloaded: (string | undefined)[] = [];
    p.shell.on(/^ldd -r /, (_cmd, opts?: RunOptions) => {
      preloaded.push(opts?.env?.LD_PRELOAD);
      return answer(opts?.env?.LD_PRELOAD);
    });
    return preloaded;
  }

  test("libcuda.so.1 resolves through the toolkit's stub the engine linked against, preloaded only after a check without it fails to find the driver", async () => {
    const { p, uc } = await setup();
    configures(p, `CUDA_cuda_driver_LIBRARY:FILEPATH=${STUB}\n`);
    const preloaded = loader(p, (preload) =>
      preload === STUB ? { code: 0, stdout: `\tlibcuda.so.1 => ${STUB}\n`, stderr: "" } : NO_DRIVER,
    );
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(true);
    expect(preloaded).toEqual(TARGETS.flatMap(() => [undefined, STUB]));
  });
  test("a machine with the driver resolves against it: the stub is never preloaded", async () => {
    const { p, uc } = await setup();
    configures(p, `CUDA_cuda_driver_LIBRARY:FILEPATH=${STUB}\n`);
    const preloaded = loader(p, () => ({
      code: 0,
      stdout: "\tlibcuda.so.1 => /usr/lib/x86_64-linux-gnu/libcuda.so.1\n",
      stderr: "",
    }));
    expect((await uc.run({ gpu: 0 })).ok).toBe(true);
    expect(preloaded).toEqual(TARGETS.map(() => undefined));
  });
  test("a symbol that is really missing is refused by name under the stub too, and nothing is published", async () => {
    const { p, uc } = await setup();
    configures(p, `CUDA_cuda_driver_LIBRARY:FILEPATH=${STUB}\n`);
    const missing =
      "undefined symbol: _Z14mul_mat_q_caseIL9ggml_type142EEvR25ggml_backend_cuda_contextRK8mmq_argsP11CUstream_st\t(/r/libggml-cuda.so.0)";
    loader(p, (preload) =>
      preload === STUB ? { code: 0, stdout: "", stderr: missing } : NO_DRIVER,
    );
    const r = await uc.run({ gpu: 0 });
    expect(!r.ok && r.message).toContain("llama-server does not resolve");
    expect(!r.ok && r.message).toContain("mul_mat_q_case");
    expect(await p.fs.exists(`/r/local/engine-builds/${engine.sha7}-sm120`)).toBe(false);
  });
  test("with no stub in the cache there is nothing to preload: the refusal names the driver library, not a stale object", async () => {
    const { p, uc } = await setup();
    configures(p, "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n");
    const preloaded = loader(p, () => NO_DRIVER);
    const r = await uc.run({ gpu: 0 });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.message).toContain("libcuda.so.1 not found");
    expect(r.message).toContain("NVIDIA driver");
    expect(r.message).not.toContain("stale or empty object");
    expect(preloaded.every((preload) => preload === undefined)).toBe(true);
  });
});
