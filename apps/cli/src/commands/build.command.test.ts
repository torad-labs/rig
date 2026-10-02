import { describe, expect, test } from "bun:test";
import { ExitCode, ok } from "@rig/core";
import type {
  BuildEngine,
  BuildEngineOptions,
  BuildPrebuilt,
  BuildPrebuiltOptions,
} from "@rig/engine";
import { FakeLog } from "@rig/testing";
import { parseArgs, UsageError, unknownFlags } from "../cli/args.ts";
import { buildEngineCommand } from "./build.command.ts";

const SHA = "3d40ae99ca0b9becce1b91134ad2bdd5f8aadf26";

/** the command over two recording services */
function command() {
  const asked: { build: BuildEngineOptions[]; prebuilt: BuildPrebuiltOptions[] } = {
    build: [],
    prebuilt: [],
  };
  const build = {
    run: async (options: BuildEngineOptions) => {
      asked.build.push(options);
      return ok({
        dir: "/r/local/engine-lab/cand",
        cap: "120",
        alreadyBuilt: false,
        marker: "fork=off-pin",
      });
    },
  } as unknown as BuildEngine;
  const prebuilt = {
    run: async (options: BuildPrebuiltOptions) => {
      asked.prebuilt.push(options);
      return ok({ image: "rig-prebuilt:0123456789ab", sha: SHA, tarball: "/r/t.tar.gz" });
    },
  } as unknown as BuildPrebuilt;
  return { asked, cmd: buildEngineCommand(build, prebuilt, new FakeLog()) };
}

describe("build", () => {
  test("--off-pin is a flag of the command and reaches the build with the tarball it names a lab directory for", async () => {
    const { asked, cmd } = command();
    const args = parseArgs([
      "--from-tarball",
      "/c/engine-sm120-0123abc.tar.gz",
      "--off-pin",
      "cand",
    ]);
    expect(unknownFlags(args, cmd.usage)).toEqual([]);
    expect(await cmd.run(args)).toBe(ExitCode.Ok);
    expect([asked.build[0]?.fromTarball, asked.build[0]?.offPin]).toEqual([
      "/c/engine-sm120-0123abc.tar.gz",
      "cand",
    ]);
  });
  test("--prebuilt builds in the release image with --gpu, --jobs (6 unless given) and --sha, and not the host's build", async () => {
    const { asked, cmd } = command();
    const args = parseArgs(["--prebuilt", "--sha", SHA, "--gpu", "1"]);
    expect(unknownFlags(args, cmd.usage)).toEqual([]);
    expect(await cmd.run(args)).toBe(ExitCode.Ok);
    expect(asked.prebuilt).toEqual([{ gpu: 1, jobs: 6, sha: SHA }]);
    expect(asked.build).toEqual([]);
    await cmd.run(parseArgs(["--prebuilt", "--jobs", "4"]));
    expect(asked.prebuilt[1]).toEqual({ gpu: 0, jobs: 4, sha: undefined });
  });
  test("--sha without --prebuilt, and --prebuilt with a head or a host build's flags, are usage errors", async () => {
    for (const argv of [
      ["--sha", SHA],
      ["glm-5.3-flash", "--prebuilt"],
      ["--prebuilt", "--portable"],
      ["--prebuilt", "--from-tarball", "/c/t.tar.gz"],
      ["--prebuilt", "--cap", "120"],
    ]) {
      const { asked, cmd } = command();
      await expect(cmd.run(parseArgs(argv))).rejects.toBeInstanceOf(UsageError);
      expect([asked.build, asked.prebuilt]).toEqual([[], []]);
    }
  });
});
