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

  test("--b-args is one argument under the lease, whitespace and all, and only ab takes it", async () => {
    const { p, command } = setup();
    p.shell.on(/^\/bin\/rig engine/, { code: 0, stdout: "", stderr: "" });
    await command.run(
      parseArgs([
        "ab",
        "--tree",
        "glm",
        "--gpu",
        "0",
        "--model",
        "/m.gguf",
        "--b-args=-ctk q8_0 -ctv q8_0",
        "--run",
        "kv",
      ]),
    );
    expect(p.cardLease.leases[0]?.cmd).toContain("--b-args=-ctk q8_0 -ctv q8_0");
    await expect(
      command.run(
        parseArgs([
          "kld",
          "--tree",
          "glm",
          "--gpu",
          "0",
          "--model",
          "/m.gguf",
          "--text",
          "/t",
          "--base",
          "/b",
          "--tag",
          "x",
          "--b-args=-ctk q8_0",
        ]),
      ),
    ).rejects.toThrow(UsageError);
  });

  test("held, --a-args and --b-args are the arguments of that side's bench and of no other", async () => {
    const { p, command } = setup();
    const tree = "/r/local/engine-build-trees/glm";
    p.fs.put(`${tree}/bin/libggml-cuda.so.0`, "the tree's library");
    p.fs.put(`${tree}/bin/llama-bench`, "elf");
    p.shell.on(/^ldd /, (_cmd, opts) => ({
      code: 0,
      stdout: `\tlibggml-cuda.so.0 => ${opts?.env?.LD_LIBRARY_PATH?.split(":")[0]}/libggml-cuda.so.0 (0x7f)\n`,
      stderr: "",
    }));
    const benches: string[] = [];
    // -o jsonl is on the bench argv and not on the ldd one, which also names llama-bench
    p.shell.on(/-o jsonl/, (cmd) => {
      benches.push(cmd.slice(cmd.indexOf("-p")).join(" "));
      return { code: 0, stdout: `{"samples_ts": [1, 100, 100]}\n`, stderr: "" };
    });
    const code = await command.run(
      parseArgs([
        "ab",
        "--tree",
        "glm",
        "--gpu",
        "0",
        "--model",
        "/m.gguf",
        "--a-args=-ctk f16",
        "--b-args=  -ctk q8_0   -ctv q8_0 ",
        "--pairs",
        "1",
        "--run",
        "kv",
        "--held",
        "--",
        "-p",
        "0",
      ]),
    );
    expect(code).toBe(0);
    // pair 1 runs a then b; the b value's stray spaces are not arguments
    expect(benches).toEqual(["-p 0 -ctk f16", "-p 0 -ctk q8_0 -ctv q8_0"]);
    expect(p.log.lines.join("\n")).toContain("with -ctk q8_0 -ctv q8_0");
  });

  test("--median rates each arm by the median of its repetitions, under the lease and held; only ab takes it", async () => {
    const { p, command } = setup();
    p.shell.on(/^\/bin\/rig engine/, { code: 0, stdout: "", stderr: "" });
    const ab = ["ab", "--tree", "glm", "--gpu", "0", "--model", "/m.gguf", "--run", "med"];
    await command.run(parseArgs([...ab, "--median"]));
    expect(p.cardLease.leases[0]?.cmd).toContain("--median");
    const tree = "/r/local/engine-build-trees/glm";
    p.fs.put(`${tree}/bin/libggml-cuda.so.0`, "the tree's library");
    p.fs.put(`${tree}/bin/llama-bench`, "elf");
    p.shell.on(/^ldd /, (_cmd, opts) => ({
      code: 0,
      stdout: `\tlibggml-cuda.so.0 => ${opts?.env?.LD_LIBRARY_PATH?.split(":")[0]}/libggml-cuda.so.0 (0x7f)\n`,
      stderr: "",
    }));
    // one slow repetition: its mean is 80, its median 100
    p.shell.on(/-o jsonl/, () => ({
      code: 0,
      stdout: `{"samples_ts": [1, 100, 100, 40]}\n`,
      stderr: "",
    }));
    const held = [...ab, "--pairs", "1", "--held", "--", "-p", "0"];
    expect(await command.run(parseArgs([...held.slice(0, 9), "--median", ...held.slice(9)]))).toBe(
      0,
    );
    const said = p.log.lines.join("\n");
    expect(said).toContain("pair 1: a 100.0, b 100.0 t/s");
    expect(said).toContain("each arm the median of its repetitions after the first");
    await expect(
      command.run(
        parseArgs([
          "kld",
          "--tree",
          "glm",
          "--gpu",
          "0",
          "--model",
          "/m.gguf",
          "--text",
          "/t",
          "--base",
          "/b",
          "--tag",
          "x",
          "--median",
        ]),
      ),
    ).rejects.toThrow(UsageError);
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
