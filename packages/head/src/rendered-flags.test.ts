import { describe, expect, test } from "bun:test";
import { readdir } from "node:fs/promises";
import { argAliases } from "@rig/engine";
import { pinnedSource, repoRoot } from "@rig/testing";
import { cacheArgv } from "./cache-formats.ts";
import { RENDERED, RENDERED_FLAGS } from "./head-config.ts";
import aliases from "./rendered-flag-aliases.json";

const root = repoRoot;
const SHA = /sha = "([0-9a-f]{40})"/;
const pin = SHA.exec(await Bun.file(`${root}/engine/engine.toml`).text())![1]!;

/** the engine a head pins (its own [engine] sha, else the default pin) and the flags rig renders for it: the core
 *  serverArgv always renders, plus the conditional ones its own [cache] turns on. The conditional ones come from
 *  cacheArgv itself, the renderer whose output depends on the head's config rather than on the machine, so this
 *  cannot drift from what serve actually passes. Takes the file's text, so a synthetic head proves it can fail. */
export function headPinAndFlags(
  toml: string,
  defaultPin: string,
): { sha: string; flags: string[] } {
  const core = ["-m", "-ngl", "--jinja", "-fa", "-c", "-np", "--kv-unified", "--cache-ram"];
  const own = /\[engine\][^[]*?sha = "([0-9a-f]{40})"/s.exec(toml)?.[1];
  const cache = /\[cache\][^[]*/s.exec(toml)?.[0] ?? "";
  const named = (key: string) => new RegExp(`^${key} = "([a-z0-9_]+)"`, "m").exec(cache)?.[1];
  const formats = {
    k: named("k") ?? "f16",
    v: named("v") ?? "f16",
    ...(named("s") ? { s: named("s") } : {}),
    ...(named("idx") ? { idx: named("idx") } : {}),
  };
  const rendered = cacheArgv(
    { cache: { mean_center: undefined }, path: (p: string) => p } as never,
    formats as never,
  ).filter((token) => token.startsWith("-"));
  return { sha: own ?? defaultPin, flags: [...core, ...rendered] };
}

/** the check itself: of `flags`, the ones the engine `sha` declares none of. An engine the manifest has no block for
 *  declares nothing, so every flag is named rather than none. Pure, so the mutation arms below exercise THIS and not
 *  whichever conditional flags cacheArgv happens to render today. */
function undeclared(flags: readonly string[], sha: string): string[] {
  const block = (aliases.engines as Record<string, Record<string, string[]>>)[sha];
  return block === undefined ? [...flags] : flags.filter((flag) => !(flag in block));
}

async function headsWithPins(
  dir: string,
): Promise<Array<{ name: string; sha: string; flags: string[] }>> {
  const out = [];
  for (const entry of await readdir(`${dir}/heads`, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const toml = await Bun.file(`${dir}/heads/${entry.name}/head.toml`).text();
    out.push({ name: entry.name, ...headPinAndFlags(toml, pin) });
  }
  return out;
}

describe("rendered flags", () => {
  test("the alias table parses on and off groups and every spelling of a group, and refuses a flag the engine does not declare", () => {
    const table = `
    add_opt(common_arg(
        {"-np", "--parallel"}, "N",
        "number of parallel sequences to decode (default: %d)",
        [](common_params & params, int value) { params.n_parallel = value; }
    ).set_env("LLAMA_ARG_N_PARALLEL"));
    add_opt(common_arg(
        {"--cache-idle-slots"},
        {"--no-cache-idle-slots"},
        "save idle slots to the prompt cache on new task",
        [](common_params & params, bool value) { params.cache_idle_slots = value; }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
        "number of layers to store in VRAM for the draft model",
        [](common_params & params, int value) { params.speculative.n_gpu_layers = value; }
    ));`;
    expect(argAliases(table, ["-np", "--no-cache-idle-slots", "-ngld"])).toEqual({
      "-np": ["--parallel", "-np"],
      "--no-cache-idle-slots": ["--cache-idle-slots", "--no-cache-idle-slots"],
      "-ngld": ["--gpu-layers-draft", "--n-gpu-layers-draft", "--spec-draft-ngl", "-ngld"],
    });
    expect(() => argAliases(table, ["-c"])).toThrow("-c is not declared");
  });
  test("the manifest is at the engine pin and covers every rendered flag under every spelling", () => {
    expect(aliases.engine).toBe(pin); // a pin move regenerates it: bun tools/rendered-flag-aliases.ts
    expect(Object.keys(aliases.aliases).sort()).toEqual([...RENDERED].sort());
    for (const [flag, spellings] of Object.entries(aliases.aliases)) {
      expect(spellings).toContain(flag);
      for (const s of spellings) expect(RENDERED_FLAGS).toContain(s);
    }
    // the holes the hand-listed set had (2026-09-21): the long forms of what serve renders
    for (const s of [
      "--model",
      "--ctx-size",
      "--parallel",
      "--gpu-layers",
      "--flash-attn",
      "-cram",
      "-kvu",
      "--model-draft",
      "--draft-p-min",
    ])
      expect(RENDERED_FLAGS).toContain(s);
  });
  // A head carries its own [engine] sha, and serves on THAT engine: a flag rig renders for it has to be declared
  // there, not in whatever the default pin happens to declare. The manifest's own per-engine blocks answer that, so
  // this runs in CI with no engine source — the check the pin's arg table gives is below, and it can only run on a
  // box that has the source.
  test("every flag rig renders for a head is declared by the engine THAT HEAD pins", async () => {
    const heads = await headsWithPins(root);
    expect(heads.length).toBeGreaterThan(1); // a one-head repo would prove nothing about per-head keying
    for (const { name, sha, flags } of heads) {
      const block = (aliases.engines as Record<string, unknown>)[sha];
      expect(
        block,
        `${name} pins ${sha.slice(0, 7)}, which the manifest has no block for: bun tools/rendered-flag-aliases.ts`,
      ).toBeDefined();
      expect(
        undeclared(flags, sha),
        `${name} renders flags its engine ${sha.slice(0, 7)} does not declare`,
      ).toEqual([]);
    }
  });
  test("the default pin's block is the manifest's flat table, so one generator writes both", () => {
    expect((aliases.engines as Record<string, unknown>)[pin]).toEqual(aliases.aliases);
  });
  // Proving the check above CAN fail is the point. It was found by -ctki against the GLM head's pin before that pin
  // declared it (the head named idx while it pinned 3d40ae99c, and the check named -ctki); a pin that declares it
  // turns that case green, so the undeclared arm is a flag no engine declares. Both arms, so the gate is not green by
  // always answering the same way, and on the engine the head serves on (its own [engine], else the default pin), so a
  // manifest block that declared everything would fail.
  test("a flag the head's own engine does not declare is named, one it declares is not, and an unknown engine declares nothing", async () => {
    const glm = headPinAndFlags(
      await Bun.file(`${root}/heads/glm-5.3-flash/head.toml`).text(),
      pin,
    ).sha;
    expect(undeclared(["-m", "--no-engine-declares-this"], glm)).toEqual([
      "--no-engine-declares-this",
    ]);
    expect(undeclared(["-m", "--cache-type-k"], glm)).toEqual([]);
    expect(undeclared(["-m"], "f".repeat(40))).toEqual(["-m"]);
  });
  // and the wiring the arms above take as given: a head's [cache] reaches the flag set through cacheArgv, so the
  // day a head names a cache its engine lacks, the real-heads check above is what goes red
  test("a head's own [cache] is what grows its flag set, through the renderer serve uses", async () => {
    const bare = `[engine]\nsha = "${"0".repeat(40)}"\n`;
    expect(headPinAndFlags(bare, pin).flags).not.toContain("-cts");
    expect(
      headPinAndFlags(`${bare}\n[cache]\nk = "q8_0"\nv = "q8_0"\ns = "q8_0"\n`, pin).flags,
    ).toContain("-cts");
  });
  // CI fetches no engine source: there the test is skipped (and says so), and the pin check above still holds
  const source = pinnedSource(root, pin);
  test.skipIf(source === null)(
    "where the pinned source is on the box, the manifest equals its arg table",
    async () => {
      const table = await Bun.file(`${source}/common/arg.cpp`).text();
      expect(argAliases(table, RENDERED)).toEqual(aliases.aliases);
    },
  );
});
