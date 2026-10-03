import { describe, expect, test } from "bun:test";
import { ExitCode, fail, ok } from "@rig/core";
import type { TagRelease } from "@rig/engine";
import { FakeLog } from "@rig/testing";
import { parseArgs, UsageError, unknownFlags } from "../cli/args.ts";
import { tagCommand } from "./tag.command.ts";

/** the command over a tagger that tags, or refuses when `refuses` */
function command(refuses = false) {
  let runs = 0;
  const tagger = {
    run: async () => {
      runs++;
      return refuses
        ? fail(ExitCode.Failure, "no fresh-machine e2e passed on 54d558e")
        : ok({
            tag: "v0.1.13",
            commit: "54d558e0a1b2c3d4e5f60718293a4b5c6d7e8f90",
            receipt: "/r/local/release/e2e-54d558e.json",
          });
    },
  } as unknown as TagRelease;
  const log = new FakeLog();
  return { runs: () => runs, log, cmd: tagCommand(tagger, log) };
}

describe("tag", () => {
  test("tags, naming the tag, the commit and the receipt; takes --json", async () => {
    const { log, cmd } = command();
    expect(unknownFlags(parseArgs(["--json"]), cmd.usage)).toEqual([]);
    expect(await cmd.run(parseArgs([]))).toBe(ExitCode.Ok);
    expect(log.lines).toEqual([
      "info v0.1.13 pushed on 54d558e (/r/local/release/e2e-54d558e.json)",
    ]);
  });
  test("a refusal is its exit code and its reason", async () => {
    const { log, cmd } = command(true);
    expect(await cmd.run(parseArgs([]))).toBe(ExitCode.Failure);
    expect(log.lines).toEqual(["error no fresh-machine e2e passed on 54d558e"]);
  });
  test("an argument is a usage error, and nothing is tagged", async () => {
    const { runs, cmd } = command();
    await expect(cmd.run(parseArgs(["../public"]))).rejects.toBeInstanceOf(UsageError);
    expect(runs()).toBe(0);
  });
});
