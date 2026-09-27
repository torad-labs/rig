import { describe, expect, test } from "bun:test";
import { fakePorts } from "../../../test/fakes/index.ts";
import { pinnedSource } from "../../../test/fakes/pinned-source.ts";
import { layoutAt } from "../layout.ts";
import { engineSource, loadEngine, miscompiles, type Prebuilt, prebuiltSkip } from "./engine.ts";

const root = `${import.meta.dir}/../../..`;
const engineToml = await Bun.file(`${root}/engine/engine.toml`).text();

async function engine() {
  const p = fakePorts();
  p.fs.put("/r/engine/engine.toml", engineToml);
  const e = await loadEngine(p.fs, layoutAt("/r"));
  if (!e.ok) throw new Error(e.message);
  return e.value;
}

/** the vector kernel's type cases with FA_ALL_QUANTS off: fattn.cu's #else branch of the FATTN_VEC_CASES_ALL_D list,
 *  and f32/f32 where FATTN_VEC_CASE takes an f32 K and V as the f16 case */
function faPairs(fattn: string): string[] {
  const branch =
    /#ifdef GGML_CUDA_FA_ALL_QUANTS\n[\s\S]*?#else\n([\s\S]*?)#endif \/\/ GGML_CUDA_FA_ALL_QUANTS/.exec(
      fattn,
    )?.[1];
  if (!branch) throw new Error("fattn.cu has no FA_ALL_QUANTS #else branch");
  const pairs = [
    ...branch.matchAll(/FATTN_VEC_CASES_ALL_D\(GGML_TYPE_(\w+),\s*GGML_TYPE_(\w+)\)/g),
  ].map((m) => `${m[1]}/${m[2]}`.toLowerCase());
  const f32AsF16 =
    /K->type == GGML_TYPE_F32 && \(type_K\) == GGML_TYPE_F16/.test(fattn) &&
    /V->type == GGML_TYPE_F32 && \(type_V\) == GGML_TYPE_F16/.test(fattn);
  return f32AsF16 && pairs.includes("f16/f16") ? [...pairs, "f32/f32"] : pairs;
}

describe("engine source", () => {
  test("the submodule is the source only at the pin and clean; dirty at the pin, the one-commit fetch is", async () => {
    const p = fakePorts();
    p.fs.put("/r/engine/engine.toml", engineToml);
    const layout = layoutAt("/r");
    const e = await loadEngine(p.fs, layout);
    if (!e.ok) throw new Error(e.message);
    p.git.heads.set("/r/engine/llama.cpp", e.value.fork.sha);
    expect(await engineSource(p.fs, p.git, layout, e.value)).toEqual({
      ok: true,
      value: "/r/engine/llama.cpp",
    });
    p.git.dirty.add("/r/engine/llama.cpp"); // uncommitted lines under the pinned sha (2026-09-21 04:41)
    const fetched = `/r/local/engine-sources/llama.cpp-${e.value.sha7}`;
    expect(await engineSource(p.fs, p.git, layout, e.value)).toEqual({ ok: true, value: fetched });
    expect(p.git.fetched).toEqual([
      { dir: fetched, repo: e.value.fork.repo, sha: e.value.fork.sha },
    ]);
    // the fetched tree dirty at the pin is not the source either, and is not re-fetched over
    p.git.dirty.add(fetched);
    const r = await engineSource(p.fs, p.git, layout, e.value);
    expect(!r.ok && r.message).toContain(`${fetched} is at the pin but carries changes`);
    expect(p.git.fetched.length).toBe(1);
  });
});

describe("engine pin", () => {
  test("engine.toml parses and names the measured cards", async () => {
    const e = await engine();
    expect(e.sha7).toBe(e.fork.sha.slice(0, 7));
    expect(e.supports("120")).toBe(true);
    expect(e.supports("90")).toBe(true);
    expect(e.supports("89")).toBe(false);
    expect(e.binDir("120")).toBe(`/r/local/engine-builds/${e.sha7}-sm120`);
  });
  test("engine.toml lists the cache formats the build runs, and a regex reads the K/V pairs out of fattn.cu", async () => {
    const e = await engine();
    expect(e.caches.fa_kv).toEqual(["f16/f16", "q4_0/q4_0", "q8_0/q8_0", "bf16/bf16", "f32/f32"]);
    expect(e.caches.state).toEqual(["f32", "q8_0", "f16", "bf16"]); // every type it parses since 1fd214cc6 (engine.toml)
    const snippet = `#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
#endif // GGML_CUDA_FA_ALL_QUANTS`;
    expect(faPairs(snippet)).toEqual(["f16/f16", "q4_0/q4_0"]);
    const f32 = `const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16);
const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16);`;
    expect(faPairs(`${f32}\n${snippet}`)).toEqual(["f16/f16", "q4_0/q4_0", "f32/f32"]); // f32 through the f16 case
    expect(() => faPairs("no dispatch here")).toThrow("no FA_ALL_QUANTS #else branch");
  });
  // CI fetches no engine source: there the test is skipped (and says so), and the literal list above holds the pin's
  // reading
  const pin = (Bun.TOML.parse(engineToml) as { fork: { sha: string } }).fork.sha;
  const source = pinnedSource(root, pin);
  test.skipIf(source === null)(
    "where the pinned source is on the box, fa_kv is fattn.cu's list with FA_ALL_QUANTS off",
    async () => {
      const e = await engine();
      const fattn = await Bun.file(`${source}/ggml/src/ggml-cuda/fattn.cu`).text();
      expect(e.caches.fa_kv).toEqual(faPairs(fattn));
    },
  );
  test("CUDA 13.2.1's nvcc is listed as miscompiling sm_120, with its evidence, and nothing else is", async () => {
    const e = await engine();
    expect(miscompiles(e, "13.2.78", "120")).toContain("torad-labs/llama.cpp#56");
    expect(miscompiles(e, "13.2.86", "120")).toBeUndefined();
    expect(miscompiles(e, "13.2.78", "90")).toBeUndefined();
    expect(miscompiles(e, null, "120")).toBeUndefined();
  });
  test("a miscompiler entry needs the full nvcc version and a card", async () => {
    for (const [entry, path] of [
      ['nvcc = "13.2"\ncaps = ["120"]\nwhy = "w"', "miscompilers.0.nvcc"],
      ['nvcc = "13.2.78"\ncaps = []\nwhy = "w"', "miscompilers.0.caps"],
    ] as const) {
      const p = fakePorts();
      p.fs.put(
        "/r/engine/engine.toml",
        engineToml.replace(/^\[\[miscompilers\]\]\n(?:.+\n)+/m, `[[miscompilers]]\n${entry}\n`),
      );
      const e = await loadEngine(p.fs, layoutAt("/r"));
      expect(!e.ok && e.message).toContain(path);
    }
  });
  test("the submodule at engine/llama.cpp is committed at the sha engine.toml pins — one pin, two readers", async () => {
    const e = await engine();
    const ls = Bun.spawnSync(["git", "ls-files", "-s", "engine/llama.cpp"], { cwd: root });
    const gitlink = /^160000 ([0-9a-f]{40}) /.exec(ls.stdout.toString())?.[1];
    expect(gitlink).toBe(e.fork.sha);
  });
});

describe("engine pin, prebuilt builds", () => {
  const base = engineToml.replace(/^\[cuda\][\s\S]*$/m, "");
  const cuda = `[cuda]
version = "13.3"
[[cuda.runtime]]
url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/a.tar.xz"
sha256 = "${"a".repeat(64)}"
libs = ["libcudart.so.13"]
`;
  const prebuilt = (cap: string, sha7: string) => `[[prebuilt]]
cap = "${cap}"
url = "https://github.com/torad-labs/llama.cpp/releases/download/x/engine-sm${cap}-${sha7}.tar.gz"
sha256 = "${"b".repeat(64)}"
glibc = "2.35"
`;
  async function load(toml: string) {
    const p = fakePorts();
    p.fs.put("/r/engine/engine.toml", toml);
    return loadEngine(p.fs, layoutAt("/r"));
  }

  test("the pin's own build for a measured card is found by its card", async () => {
    const sha7 = (await engine()).sha7;
    const e = await load(`${base}\n${cuda}${prebuilt("120", sha7)}`);
    expect(e.ok && e.value.prebuiltFor("120")?.cap).toBe("120");
    expect(e.ok && e.value.prebuiltFor("90")).toBeUndefined();
  });
  test("an entry a pin move left behind names the old commit and refuses to load", async () => {
    const e = await load(`${base}\n${cuda}${prebuilt("120", "0000000")}`);
    expect(!e.ok && e.message).toContain("not the build of the pin");
  });
  test("a prebuilt without [cuda], for an unlisted card, or listed twice refuses to load", async () => {
    const sha7 = (await engine()).sha7;
    const cases: [string, string][] = [
      [`${base}\n${prebuilt("120", sha7)}`, "needs [cuda]"],
      [`${base}\n${cuda}${prebuilt("89", sha7)}`, "a card [[archs]] does not list"],
      [`${base}\n${cuda}${prebuilt("120", sha7)}${prebuilt("120", sha7)}`, "listed twice"],
    ];
    for (const [toml, message] of cases) {
      const e = await load(toml);
      expect(!e.ok && e.message).toContain(message);
    }
  });
  test("a prebuilt names its glibc floor, and a machine below it (by number: 2.4 is below 2.35) or without glibc skips it", async () => {
    const sha7 = (await engine()).sha7;
    const noFloor = await load(
      `${base}\n${cuda}${prebuilt("120", sha7).replace(/^glibc = .*\n/m, "")}`,
    );
    expect(!noFloor.ok && noFloor.message).toContain("glibc");
    const entry = { cap: "120", url: "u", sha256: "s", glibc: "2.35" } as Prebuilt;
    expect(prebuiltSkip(entry, "2.35")).toBeUndefined();
    expect(prebuiltSkip(entry, "2.39")).toBeUndefined();
    expect(prebuiltSkip(entry, "3.0")).toBeUndefined();
    expect(prebuiltSkip(entry, "2.34")).toContain("glibc 2.34 on this machine");
    expect(prebuiltSkip(entry, "2.4")).toContain("needs glibc 2.35 or newer");
    expect(prebuiltSkip(entry, null)).toContain("not glibc");
  });
});
