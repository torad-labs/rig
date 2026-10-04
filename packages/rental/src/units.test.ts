import { describe, expect, test } from "bun:test";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { renderStopService } from "./units.ts";

/** a unit's ExecStart= commands, in the order systemd runs them */
const execs = (unit: string) =>
  unit
    .split("\n")
    .filter((line) => line.startsWith("ExecStart="))
    .map((line) => line.slice("ExecStart=".length));

describe("the hard stop", () => {
  test("rig's down may fail; the two lines after it run no rig code, and either failing runs the stop again", () => {
    const unit = renderStopService({
      self: ["/r/dist/rig"],
      vastai: "/home/u/.local/bin/vastai",
      instanceId: 54030694,
    });
    const [rig, destroy, check, ...rest] = execs(unit);
    expect(rest).toEqual([]);
    // "-": rig's exit, a tree that does not load included, never stops the lines after it (Type=oneshot runs them in
    // order and stops at the first failure without one; checked on systemd 257, 2026-10-03)
    expect(rig).toBe("-/r/dist/rig vast down --box 54030694");
    expect(destroy).toBe("/home/u/.local/bin/vastai destroy instance 54030694 -y");
    expect(check).toMatch(/^python3 -c "[^"]*"$/);
    for (const line of [destroy!, check!]) {
      expect(line.startsWith("-")).toBe(false);
      expect(line).not.toContain("/r/dist/rig");
    }
    expect(unit).toContain("Type=oneshot\n");
    expect(unit).toContain("Restart=on-failure\n");
  });

  test("its check passes only once vast reads every page and no longer lists the box", () => {
    const dir = mkdtempSync(join(tmpdir(), "rig-hard-stop-"));
    const vastai = join(dir, "vastai");
    const unit = renderStopService({ self: ["/r/dist/rig"], vastai, instanceId: 54030694 });
    const code = /^python3 -c "(.*)"$/.exec(execs(unit)[2]!)![1]!;
    // systemd would read a specifier, a variable, an escape or a quote in the argument; it holds none
    expect(code).not.toMatch(/["\\%$]/);
    // a vastai that answers `show instances-v1 --raw` with `first`, and with `next` when asked for a page by its token
    writeFileSync(
      vastai,
      `#!/bin/sh\necho "$*" >> ${dir}/calls\necho 'this CLI is DEPRECATED' >&2\ncase "$*" in *--next-token*) cat ${dir}/next ;; *) cat ${dir}/first ;; esac\n`,
    );
    chmodSync(vastai, 0o755);
    const check = (first: string, next = "") => {
      writeFileSync(join(dir, "first"), first);
      writeFileSync(join(dir, "next"), next);
      writeFileSync(join(dir, "calls"), "");
      const run = Bun.spawnSync(["python3", "-c", code]);
      return { exit: run.exitCode, said: run.stdout.toString().trim() };
    };
    const page = (ids: number[], token: string | null = null) =>
      JSON.stringify({
        success: true,
        instances: ids.map((id) => ({ id, actual_status: "running" })),
        next_token: token,
      });
    expect(check(page([51000001, 54030694]))).toEqual({
      exit: 1,
      said: "vast lists box 54030694: stop again",
    });
    expect(readFileSync(join(dir, "calls"), "utf8")).toBe("show instances-v1 --raw\n");
    for (const gone of [page([51000001]), page([])])
      expect(check(gone)).toEqual({ exit: 0, said: "vast no longer lists box 54030694" });
    // the box on the second page: every page is read, each by the token the one before named
    expect(check(page([51000001], "page-2"), page([54030694]))).toEqual({
      exit: 1,
      said: "vast lists box 54030694: stop again",
    });
    expect(readFileSync(join(dir, "calls"), "utf8")).toBe(
      "show instances-v1 --raw\nshow instances-v1 --next-token page-2 --raw\n",
    );
    expect(check(page([51000001], "page-2"), page([1]))).toEqual({
      exit: 0,
      said: "vast no longer lists box 54030694",
    });
    // vast not read: vastai prints vast's refusal on stderr and exits 0 with nothing on stdout; an error object; the old
    // command's bare array; a page cut short; a token that never ends
    expect(check("")).toEqual({ exit: 1, said: "vast not read: stop again" });
    // said, not only exited 1: a check that crashes on a shape exits 1 too, and would pass for one that read it
    for (const unread of [
      '{"error": true, "status_code": 502}',
      JSON.stringify([{ id: 51000001 }]),
    ])
      expect(check(unread)).toEqual({ exit: 1, said: "vast not read: stop again" });
    expect(check('{"instances": [{"id": 5403').exit).not.toBe(0);
    expect(check(page([1], "again"), page([2], "again"))).toEqual({
      exit: 1,
      said: "vast not read: stop again",
    });
    // about 120 python3 and sh processes, the never-ending token alone 100 pages: 5,011 ms at load 43 against bun's
    // default 5,000 (rig-glm, Oct 3)
  }, 60_000);
});
