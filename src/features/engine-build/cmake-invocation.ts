// The cmake command lines for the engine, as data. Configure: CUDA for one compute capability,
// the server and tools, nothing else; the launcher cache off (GGML_CCACHE=OFF: ggml auto-detects
// sccache, whose daemon cannot start inside buildgate's cgroup scope, 18 of 487 objects failed
// on 2026-09-20, and a pinned commit's build depends on no cache daemon); RUNPATH=$ORIGIN, not
// the build tree (cmake bakes <tree>/bin by default, so a published directory would load its
// own libraries only while the tree it came from exists, on the box it was built on). Native:
// -march=native here on purpose; --portable pins the AVX2 + FMA + F16C baseline, because a
// native tarball SIGILLs on the Zen 2 EPYCs that host most rented 5090s. Every compiler maps the
// source directory to `llama.cpp` (-ffile-prefix-map; nvcc hands it to the host compiler, which
// preprocesses the device pass too): ggml's asserts print __FILE__, and a build published for
// others would otherwise carry the path of the machine that built it, 81 times in libggml-cuda.
// --portable also builds without OpenSSL (llama-server's HTTPS client and TLS serving, neither of
// which rig uses): linked against this host's OpenSSL 3.5, libllama-common needed OPENSSL_3.3.0 and
// would not load on Ubuntu 24.04's 3.0 (2026-09-24). Both modes pass the flag, because they share a
// build tree and a flag one of them leaves out keeps whatever the other one cached.
import { TARGETS } from "../../shared/engine/engine.ts";

export interface ConfigureRequest {
  source: string;
  buildTree: string;
  /** compute capability, "120" for sm_120 */
  cap: string;
  portable: boolean;
}

export function configureArgv(request: ConfigureRequest): string[] {
  const native = request.portable
    ? ["-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON"]
    : ["-DGGML_NATIVE=ON"];
  const openssl = `-DLLAMA_OPENSSL=${request.portable ? "OFF" : "ON"}`;
  const prefixMap = `-ffile-prefix-map=${request.source}=llama.cpp`;
  return [
    "cmake",
    "-S",
    request.source,
    "-B",
    request.buildTree,
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CCACHE=OFF",
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    "-DCMAKE_INSTALL_RPATH=$ORIGIN",
    `-DCMAKE_C_FLAGS=${prefixMap}`,
    `-DCMAKE_CXX_FLAGS=${prefixMap}`,
    `-DCMAKE_CUDA_FLAGS=-Xcompiler=${prefixMap}`,
    "-DGGML_CUDA=ON",
    `-DCMAKE_CUDA_ARCHITECTURES=${request.cap}a`,
    ...native,
    openssl,
    "-DGGML_CUDA_FA=ON",
    "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
    "-DGGML_CUDA_GRAPHS=ON",
    "-DGGML_CUDA_COMPRESSION_MODE=size",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_COMMON=ON",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_MTMD=OFF",
    "-DLLAMA_BUILD_UI=OFF",
  ];
}

export interface BuildRequest {
  buildTree: string;
  jobs: number;
  /** the host's build semaphore on PATH, when it provides one: boxes memory and task count and
   *  admits one build at a time; it does not set the job count */
  gate: string | null;
}

export function buildArgv(request: BuildRequest): string[] {
  const gate = request.gate ? [request.gate] : [];
  return [
    ...gate,
    "cmake",
    "--build",
    request.buildTree,
    "--target",
    ...TARGETS,
    `-j${request.jobs}`,
  ];
}

/** every empty object, archive and shared library under the build tree, printed and deleted */
export function pruneArgv(buildTree: string): string[] {
  return [
    "find",
    buildTree,
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
  ];
}
