import { describe, expect, test } from "bun:test";
import { pinnedSource } from "../../../test/fakes/pinned-source.ts";
import { argAliases } from "../engine/arg-aliases.ts";
import { RENDERED, RENDERED_FLAGS } from "./head-config.ts";
import aliases from "./rendered-flag-aliases.json";

const root = `${import.meta.dir}/../../..`;
const pin = /sha = "([0-9a-f]{40})"/.exec(await Bun.file(`${root}/engine/engine.toml`).text())![1]!;

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
    expect(aliases.engine).toBe(pin); // a pin move regenerates it: bun scripts/rendered-flag-aliases.ts
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
