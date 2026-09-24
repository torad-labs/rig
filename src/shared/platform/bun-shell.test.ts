import { describe, expect, test } from "bun:test";
import { BunShell } from "./bun-shell.ts";

describe("BunShell.run", () => {
  test("a missing executable is exit 127, not an exception (systemctl on a machine without systemd)", async () => {
    const r = await new BunShell().run(["rig-test-no-such-executable", "--user", "show"]);
    expect(r).toEqual({
      code: 127,
      stdout: "",
      stderr: "rig-test-no-such-executable: command not found",
    });
  });
  test("an executable that runs reports its own code and output", async () => {
    const r = await new BunShell().run(["sh", "-c", "echo out; echo err >&2; exit 3"]);
    expect(r).toEqual({ code: 3, stdout: "out\n", stderr: "err\n" });
  });
});
