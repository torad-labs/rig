import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, statfsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BunFileSystem } from "./bun-file-system.ts";

describe("BunFileSystem.freeBytes", () => {
  test("what an unprivileged writer can still write, asked of a path not created yet: its nearest existing ancestor's filesystem", async () => {
    const dir = import.meta.dir;
    const free = await new BunFileSystem().freeBytes(join(dir, "not-yet", "packs", "head"));
    const fs = statfsSync(dir);
    // other writers move the disk between the two reads: slack for them, far below the reserve
    // a superuser-only count (bfree) or the whole disk (blocks) would add
    expect(Math.abs(free - fs.bavail * fs.bsize)).toBeLessThan(256 * 1024 * 1024);
    expect(free).toBeLessThan(fs.blocks * fs.bsize);
  });
});

describe("BunFileSystem.claim", () => {
  test("writes only where nothing is: the first claim of a path wins and the second leaves the first's text", async () => {
    const fs = new BunFileSystem();
    const path = join(mkdtempSync(join(tmpdir(), "rig-claim-")), "ports", "8100");
    const claims = await Promise.all([fs.claim(path, "first"), fs.claim(path, "second")]);
    expect(claims.filter(Boolean)).toHaveLength(1);
    expect(readFileSync(path, "utf8")).toBe(claims[0] ? "first" : "second");
    expect(await fs.claim(path, "third")).toBe(false);
    await fs.remove(path);
    expect(await fs.claim(path, "fourth")).toBe(true);
  });
});
