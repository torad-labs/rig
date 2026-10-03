import { describe, expect, test } from "bun:test";
import { ExitCode, fail, ok } from "@rig/core";
import type { DriverOnlyGate, DriverOnlyGateOptions } from "@rig/engine";
import { FakeLog } from "@rig/testing";
import { parseArgs, UsageError, unknownFlags } from "../cli/args.ts";
import { e2eCommand } from "./e2e.command.ts";

/** the command over a recording gate that passes, or fails when `fails` */
function command(lock?: string, fails = false) {
  const asked: DriverOnlyGateOptions[] = [];
  const gate = {
    run: async (options: DriverOnlyGateOptions) => {
      asked.push(options);
      return fails
        ? fail(ExitCode.Failure, "the driver-only gate failed (exit 1)")
        : ok({
            base: options.base,
            prebuilt: null,
            log: "/r/local/logs/e2e-driver-only.log",
            receipt: "/r/local/release/e2e-54d558e.json",
          });
    },
  } as unknown as DriverOnlyGate;
  const log = new FakeLog();
  return { asked, log, cmd: e2eCommand(gate, lock, log) };
}

describe("e2e", () => {
  test("takes the pack, and --prebuilt --gpu --base --head, the card 0 and ubuntu:22.04 unless given", async () => {
    const { asked, cmd } = command();
    const args = parseArgs(["/m/pack.gguf"]);
    expect(
      unknownFlags(
        parseArgs([
          "/m/p.gguf",
          "--prebuilt",
          "/t",
          "--gpu",
          "1",
          "--base",
          "i",
          "--head",
          "h",
          "--json",
        ]),
        cmd.usage,
      ),
    ).toEqual([]);
    expect(await cmd.run(args)).toBe(ExitCode.Ok);
    expect(asked).toEqual([
      {
        pack: "/m/pack.gguf",
        prebuilt: undefined,
        gpu: 0,
        base: "ubuntu:22.04",
        head: undefined,
        lock: undefined,
      },
    ]);
    await cmd.run(
      parseArgs([
        "/m/p.gguf",
        "--prebuilt",
        "/p/t.tar.gz",
        "--gpu",
        "1",
        "--base",
        "ubuntu:24.04",
        "--head",
        "bonsai-2-27b",
      ]),
    );
    expect(asked[1]).toEqual({
      pack: "/m/p.gguf",
      prebuilt: "/p/t.tar.gz",
      gpu: 1,
      base: "ubuntu:24.04",
      head: "bonsai-2-27b",
      lock: undefined,
    });
  });
  test("the gate lock is the one main read from RIG_GATE_LOCK", async () => {
    const { asked, cmd } = command("/run/gate.lock");
    await cmd.run(parseArgs(["/m/pack.gguf"]));
    expect(asked[0]?.lock).toBe("/run/gate.lock");
  });
  test("a pass names the receipt `rig tag` reads", async () => {
    const { log, cmd } = command();
    await cmd.run(parseArgs(["/m/pack.gguf"]));
    expect(log.lines.join("\n")).toContain("receipt /r/local/release/e2e-54d558e.json");
  });
  test("a failed gate is its exit code", async () => {
    const { cmd } = command(undefined, true);
    expect(await cmd.run(parseArgs(["/m/pack.gguf"]))).toBe(ExitCode.Failure);
  });
  test("no pack, a second pack, or a flag with no value are usage errors", async () => {
    for (const argv of [
      [],
      ["/a.gguf", "/b.gguf"],
      ["/a.gguf", "--prebuilt"],
      ["/a.gguf", "--base"],
    ]) {
      const { asked, cmd } = command();
      await expect(cmd.run(parseArgs(argv))).rejects.toBeInstanceOf(UsageError);
      expect(asked).toEqual([]);
    }
  });
});
