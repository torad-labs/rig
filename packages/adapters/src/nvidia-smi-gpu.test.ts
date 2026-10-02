import { describe, expect, test } from "bun:test";
import { FakeShell } from "@rig/testing";
import { NvidiaSmiGpu } from "./nvidia-smi-gpu.ts";

// nvidia-smi's csv, noheader, nounits: the shapes this box's driver 610.43.02 prints
describe("NvidiaSmiGpu", () => {
  test("query reads the card's total and what is in use on it now", async () => {
    const shell = new FakeShell();
    shell.on(
      /^nvidia-smi -i 1 --query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version/,
      {
        code: 0,
        stdout: "1, NVIDIA GeForce RTX 5070 Ti, 16303, 2318, 12.0, 610.43.02\n",
        stderr: "",
      },
    );
    expect(await new NvidiaSmiGpu(shell).query(1)).toEqual({
      index: 1,
      name: "NVIDIA GeForce RTX 5070 Ti",
      memoryMiB: 16303,
      usedMiB: 2318,
      computeCap: "120",
      driver: "610.43.02",
    });
  });
  test("list is every card by its index, as this box's two print; none without a driver", async () => {
    const shell = new FakeShell();
    shell.on(/^nvidia-smi --query-gpu=index,/, {
      code: 0,
      stdout:
        "0, NVIDIA GeForce RTX 5080, 16303, 5, 12.0, 610.43.02\n1, NVIDIA GeForce RTX 5070 Ti, 16303, 1557, 12.0, 610.43.02\n",
      stderr: "",
    });
    const cards = await new NvidiaSmiGpu(shell).list();
    expect(cards.map((card) => [card.index, card.name, card.usedMiB])).toEqual([
      [0, "NVIDIA GeForce RTX 5080", 5],
      [1, "NVIDIA GeForce RTX 5070 Ti", 1557],
    ]);
    const none = new FakeShell();
    none.on(/^nvidia-smi/, { code: 9, stdout: "", stderr: "NVIDIA-SMI has failed" });
    expect(await new NvidiaSmiGpu(none).list()).toEqual([]);
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
  test("toolkitCuda is the compiler's build number, of nvcc on PATH or of the compiler named", async () => {
    const version = (release: string, build: string) => ({
      code: 0,
      stdout: `nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2026 NVIDIA Corporation\nCuda compilation tools, release ${release}, V${build}\nBuild cuda_${release}.r${release}/compiler.37061995_0\n`,
      stderr: "",
    });
    const shell = new FakeShell();
    shell.tools.add("nvcc");
    shell.on(/^nvcc --version$/, version("13.2", "13.2.78"));
    shell.on(/^\/usr\/local\/cuda-13\.3\/bin\/nvcc --version$/, version("13.3", "13.3.33"));
    const gpu = new NvidiaSmiGpu(shell);
    expect(await gpu.toolkitCuda()).toBe("13.2.78");
    expect(await gpu.toolkitCuda("/usr/local/cuda-13.3/bin/nvcc")).toBe("13.3.33");
    shell.tools.delete("nvcc");
    expect(await gpu.toolkitCuda()).toBeNull();
  });
});
