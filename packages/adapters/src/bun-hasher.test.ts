import { describe, expect, test } from "bun:test";
import { chmodSync, mkdtempSync, renameSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BunHasher } from "./bun-hasher.ts";

const sha = (text: string) => new Bun.CryptoHasher("sha256").update(text).digest("hex");
const scratch = () => mkdtempSync(join(tmpdir(), "rig-hasher-"));

// Over real files: the worker reads them, and the memory of a hash is keyed on what stat says
describe("BunHasher", () => {
  test("the sha256 of the file's bytes, read once while the file stays the same, a rename included", async () => {
    const dir = scratch();
    writeFileSync(join(dir, "a.part"), "the pinned bytes");
    const hasher = new BunHasher();
    expect(await hasher.sha256File(join(dir, "a.part"))).toBe(sha("the pinned bytes"));
    renameSync(join(dir, "a.part"), join(dir, "a"));
    expect(await hasher.sha256File(join(dir, "a"))).toBe(sha("the pinned bytes"));
    expect(hasher.reads).toBe(1);
  });

  test("a write that changes the size, or only the bytes and the time, is read again", async () => {
    const dir = scratch();
    const file = join(dir, "a");
    writeFileSync(file, "one");
    const hasher = new BunHasher();
    await hasher.sha256File(file);
    writeFileSync(file, "one more");
    expect(await hasher.sha256File(file)).toBe(sha("one more"));
    writeFileSync(file, "two more");
    utimesSync(file, new Date(2_000_000_000_000), new Date(2_000_000_000_000));
    expect(await hasher.sha256File(file)).toBe(sha("two more"));
    expect(hasher.reads).toBe(3);
  });

  test("files hashed at once each take a worker, and each gets its own answer", async () => {
    const dir = scratch();
    const names = ["a", "b", "c"];
    for (const name of names) writeFileSync(join(dir, name), `shard ${name}`);
    const hasher = new BunHasher();
    const hashes = await Promise.all(names.map((name) => hasher.sha256File(join(dir, name))));
    expect(hashes).toEqual(names.map((name) => sha(`shard ${name}`)));
  });

  test("a missing file throws ENOENT and an unreadable one EACCES, the codes artifact.ts tells apart", async () => {
    const dir = scratch();
    const hasher = new BunHasher();
    await expect(hasher.sha256File(join(dir, "none"))).rejects.toMatchObject({ code: "ENOENT" });
    writeFileSync(join(dir, "locked"), "x");
    chmodSync(join(dir, "locked"), 0o000);
    if (process.getuid?.() !== 0)
      await expect(hasher.sha256File(join(dir, "locked"))).rejects.toMatchObject({
        code: "EACCES",
      });
  });
});
