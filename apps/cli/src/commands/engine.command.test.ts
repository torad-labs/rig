import { describe, expect, test } from "bun:test";
import { ExitCode, layoutAt } from "@rig/core";
import { EngineLab } from "@rig/engine";
import { fakePorts } from "@rig/testing";
import { parseArgs, UsageError } from "../cli/args.ts";
import { engineLabCommand } from "./engine.command.ts";

function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  const command = engineLabCommand({
    lab: new EngineLab(p, layout),
    lease: p.cardLease,
    self: ["/bin/rig"],
    treesDir: layout.engineBuildTreesDir,
    // 2026-10-01 21:04 UTC
    now: () => Date.UTC(2026, 9, 1, 21, 4),
    log: p.log,
  });
  return { p, command };
}

describe("rig engine", () => {
  test("a run holds its cards once: rig again under the lease, its run named and its paths absolute", async () => {
    const { p, command } = setup();
    p.shell.on(/^\/bin\/rig engine/, { code: 0, stdout: "", stderr: "" });
    const code = await command.run(
      parseArgs([
        "ab",
        "--tree",
        "glm",
        "--gpu",
        "0,1",
        "--model",
        "/m.gguf",
        "--b-env",
        "GGML_CUDA_X_LEGACY=1",
        "--",
        "-p",
        "4096",
      ]),
    );
    expect(code).toBe(0);
    expect(p.cardLease.leases).toEqual([
      {
        cards: [0, 1],
        terms: { label: "rig engine ab ab-20261001T2104Z", etaMin: 10, maxHoldMin: 120 },
        cmd: [
          "/bin/rig",
          "engine",
          "ab",
          "--tree=glm",
          "--gpu=0,1",
          "--model=/m.gguf",
          "--b-env=GGML_CUDA_X_LEGACY=1",
          "--run=ab-20261001T2104Z",
          "--held",
          "--",
          "-p",
          "4096",
        ],
      },
    ]);
  });

  test("--eta tells the card queue the run's expected minutes, and the run under the lease is given it too", async () => {
    const { p, command } = setup();
    p.shell.on(/^\/bin\/rig engine/, { code: 0, stdout: "", stderr: "" });
    await command.run(
      parseArgs([
        "test",
        "--tree",
        "glm",
        "--gpu",
        "1",
        "--ops",
        "ADD",
        "--eta",
        "25",
        "--run",
        "t",
      ]),
    );
    expect(p.cardLease.leases.map((lease) => lease.terms)).toEqual([
      { label: "rig engine test t", etaMin: 25, maxHoldMin: 30 },
    ]);
    expect(p.cardLease.leases[0]?.cmd).toContain("--eta=25");
  });

  test("the lease's exit code is the run's", async () => {
    const { p, command } = setup();
    p.shell.on(/^\/bin\/rig engine/, { code: 1, stdout: "", stderr: "FAILED card 0" });
    const code = await command.run(
      parseArgs(["test", "--tree", "glm", "--gpu", "0", "--ops", "ADD"]),
    );
    expect(code).toBe(1);
  });

  test("a flag another subcommand takes is refused, nothing leased", async () => {
    const { p, command } = setup();
    await expect(
      command.run(
        parseArgs(["test", "--tree", "glm", "--gpu", "0", "--ops", "ADD", "--pairs", "3"]),
      ),
    ).rejects.toThrow(UsageError);
    expect(p.cardLease.leases).toEqual([]);
  });

  test("an unknown subcommand is a usage error", async () => {
    const { command } = setup();
    expect(await command.run(parseArgs(["bench"]))).toBe(ExitCode.Usage);
  });

  test("relink holds no cards", async () => {
    const { p, command } = setup();
    p.fs.put("/r/local/engine-build-trees/glm/CMakeCache.txt", "nothing\n");
    const code = await command.run(
      parseArgs(["relink", "--tree", "glm", "--head", "ggml/src/x.cuh"]),
    );
    expect(code).toBe(ExitCode.Failure);
    expect(p.log.lines.join("\n")).toContain("not a configured cmake tree");
    expect(p.cardLease.leases).toEqual([]);
  });
});
