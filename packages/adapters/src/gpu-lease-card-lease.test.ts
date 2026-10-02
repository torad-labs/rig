import { describe, expect, test } from "bun:test";
import { FakeShell } from "@rig/testing";
import { GpuLeaseCardLease } from "./gpu-lease-card-lease.ts";

const ok = { code: 0, stdout: "", stderr: "" };

describe("GpuLeaseCardLease", () => {
  test("holds one card or both through gpu-lease, its queue told the expected minutes, the command after --", async () => {
    const shell = new FakeShell().on(/.*/, ok);
    shell.tools.add("gpu-lease");
    const lease = new GpuLeaseCardLease(shell);
    await lease.run([1], { label: "rig engine test", etaMin: 6, maxHoldMin: 20 }, [
      "test-backend-ops",
      "-o",
      "ADD",
    ]);
    await lease.run([1, 0], { label: "rig engine ab", etaMin: 12, maxHoldMin: 90 }, [
      "rig",
      "engine",
      "ab",
    ]);
    expect(shell.calls).toEqual([
      [
        "gpu-lease",
        "run",
        "--card",
        "1",
        "--eta",
        "6m",
        "--max-hold",
        "20m",
        "--label",
        "rig engine test",
        "--",
        "test-backend-ops",
        "-o",
        "ADD",
      ],
      [
        "gpu-lease",
        "run",
        "--card",
        "both",
        "--eta",
        "12m",
        "--max-hold",
        "90m",
        "--label",
        "rig engine ab",
        "--",
        "rig",
        "engine",
        "ab",
      ],
    ]);
  });
  test("a set gpu-lease cannot hold is refused by name, nothing run", async () => {
    const shell = new FakeShell().on(/.*/, ok);
    shell.tools.add("gpu-lease");
    const result = await new GpuLeaseCardLease(shell).run(
      [0, 2],
      { label: "x", etaMin: 1, maxHoldMin: 5 },
      ["true"],
    );
    expect(result.code).toBe(1);
    expect(result.stderr).toContain("not 0,2");
    expect(shell.calls).toEqual([]);
  });
  test("without gpu-lease the command runs as it is", async () => {
    const shell = new FakeShell().on(/.*/, ok);
    await new GpuLeaseCardLease(shell).run([0, 1], { label: "x", etaMin: 1, maxHoldMin: 5 }, [
      "llama-bench",
      "-p",
      "512",
    ]);
    expect(shell.calls).toEqual([["llama-bench", "-p", "512"]]);
  });
});
