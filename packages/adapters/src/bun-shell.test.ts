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
  test("a timeout asks the process to stop before it kills it, so it can stop what it started", async () => {
    // the trap runs only on a TERM, and stops the helper it started, as nsys stops its agent on one
    const r = await new BunShell().run(
      ["sh", "-c", "sleep 30 > /dev/null 2>&1 & trap 'kill $!; echo stopped; exit 143' TERM; wait"],
      { timeoutMs: 200 },
    );
    expect(r).toEqual({ code: 143, stdout: "stopped\n", stderr: "" });
  });
  test("onLine sees each line while the process still runs, and the result still holds the whole output", async () => {
    const seen: Array<[string, number]> = [];
    const started = Date.now();
    const r = await new BunShell().run(
      ["sh", "-c", "echo one; echo two >&2; sleep 0.5; printf 'three\\rfour'"],
      { onLine: (line) => seen.push([line, Date.now() - started]) },
    );
    expect(r).toEqual({ code: 0, stdout: "one\nthree\rfour", stderr: "two\n" });
    expect(seen.map(([line]) => line).sort()).toEqual(["four", "one", "three", "two"]);
    // the first two arrived before the half second the process then slept
    expect(seen.find(([line]) => line === "one")![1]).toBeLessThan(400);
  });
  test("onLine says which stream each line came from", async () => {
    const seen: string[] = [];
    await new BunShell().run(["sh", "-c", "echo out; echo err >&2"], {
      onLine: (line, stream) => seen.push(`${stream} ${line}`),
    });
    expect(seen.sort()).toEqual(["stderr err", "stdout out"]);
  });
  test("a process that ignores the TERM is killed after the grace", async () => {
    const started = Date.now();
    const r = await new BunShell({ killAfterMs: 300 }).run(
      ["sh", "-c", "trap '' TERM; exec sleep 30"],
      { timeoutMs: 200 },
    );
    expect(r.code).toBe(137);
    expect(Date.now() - started).toBeLessThan(5_000);
  });
});
