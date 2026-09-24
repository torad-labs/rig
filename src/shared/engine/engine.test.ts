import { describe, expect, test } from "bun:test";
import { fakePorts } from "../../../test/fakes/index.ts";
import { layoutAt } from "../layout.ts";
import { engineSource, loadEngine } from "./engine.ts";

const root = `${import.meta.dir}/../../..`;
const engineToml = await Bun.file(`${root}/engine/engine.toml`).text();

async function engine() {
  const p = fakePorts();
  p.fs.put("/r/engine/engine.toml", engineToml);
  const e = await loadEngine(p.fs, layoutAt("/r"));
  if (!e.ok) throw new Error(e.message);
  return e.value;
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
});
