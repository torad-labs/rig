import { describe, expect, test } from "bun:test";
import { FakeShell } from "../../../test/fakes/index.ts";
import { VastAiRental } from "./vast-ai-rental.ts";

const listing = (ids: number[]) => ({
  code: 0,
  stdout: JSON.stringify(ids.map((id) => ({ id, actual_status: "running", label: "rig" }))),
  stderr: "",
});

describe("VastAiRental.show", () => {
  test("a failed show falls back to the listing: its row when listed, null when the listing confirms the instance is gone, thrown when the listing fails too", async () => {
    const shell = new FakeShell();
    const vast = new VastAiRental(shell);
    shell.on(/^vastai show instance 1000 --raw$/, {
      code: 0,
      stdout: JSON.stringify({ id: 1000, actual_status: "running", label: "rig", gpu_util: 97 }),
      stderr: "",
    });
    expect(await vast.show(1000)).toMatchObject({ id: 1000, status: "running", gpuUtil: 97 });
    // a 429 on show while the listing still has the box: the box is not gone, and the listing's
    // row is the market read, its card reading included
    shell.on(/^vastai show instance 1000 --raw$/, {
      code: 1,
      stdout: "",
      stderr: "429 Too Many Requests",
    });
    shell.on(/^vastai show instances --raw$/, {
      code: 0,
      stdout: JSON.stringify([{ id: 1000, actual_status: "running", label: "rig", gpu_util: 97 }]),
      stderr: "",
    });
    expect(await vast.show(1000)).toMatchObject({ id: 1000, status: "running", gpuUtil: 97 });
    // the listing unreadable too (an expired key): nothing confirms the box is gone
    shell.on(/^vastai show instances --raw$/, { code: 1, stdout: "", stderr: "401 Unauthorized" });
    await expect(vast.show(1000)).rejects.toThrow("401 Unauthorized");
    // destroyed: show fails and the listing has no such box
    shell.on(/^vastai show instances --raw$/, listing([56]));
    expect(await vast.show(1000)).toBeNull();
  });
});
