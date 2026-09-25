import { describe, expect, test } from "bun:test";
import { FakeShell } from "../../../test/fakes/index.ts";
import { NvidiaSmiGpu } from "./nvidia-smi-gpu.ts";

// nvidia-smi's csv, noheader, nounits: the shapes this box's driver 610.43.02 prints
describe("NvidiaSmiGpu", () => {
  test("query reads the card's total and what is in use on it now", async () => {
    const shell = new FakeShell();
    shell.on(/--query-gpu=name,memory.total,memory.used,compute_cap,driver_version/, {
      code: 0,
      stdout: "NVIDIA GeForce RTX 5070 Ti, 16303, 2318, 12.0, 610.43.02\n",
      stderr: "",
    });
    expect(await new NvidiaSmiGpu(shell).query(1)).toEqual({
      index: 1,
      name: "NVIDIA GeForce RTX 5070 Ti",
      memoryMiB: 16303,
      usedMiB: 2318,
      computeCap: "120",
      driver: "610.43.02",
    });
  });
  test("processMiB is the pid's own row on that card, 0 when it holds none", async () => {
    const shell = new FakeShell();
    shell.on(/^nvidia-smi -i 0 --query-compute-apps=pid,used_memory/, {
      code: 0,
      stdout: "3919564, 13784\n4750, 284\n",
      stderr: "",
    });
    const gpu = new NvidiaSmiGpu(shell);
    expect(await gpu.processMiB(0, 3919564)).toBe(13784);
    expect(await gpu.processMiB(0, 4750)).toBe(284);
    expect(await gpu.processMiB(0, 391956)).toBe(0);
  });
});
