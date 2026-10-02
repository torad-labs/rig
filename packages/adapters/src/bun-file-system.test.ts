import { describe, expect, test } from "bun:test";
import { statfsSync } from "node:fs";
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
