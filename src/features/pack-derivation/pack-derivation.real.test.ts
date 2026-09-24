// R3 for the bake: the TypeScript ablation must reproduce the pinned served pack from the real
// source pack, byte for byte, or the transcription of the numpy reference is wrong somewhere in
// its arithmetic order. Needs the 7.2 GB source pack, ~75 s and 7.2 GB of scratch; runs only
// when RIG_REAL_PACK names the source file (proved 2026-09-20: sha e7b99670…, 22,674,442 digits
// flipped of 5,908,725,760).
import { describe, expect, test } from "bun:test";
import { rmSync } from "node:fs";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { realPorts } from "../../shared/platform/index.ts";
import { DerivePack } from "./pack-derivation.service.ts";

const source = process.env.RIG_REAL_PACK;
const root = `${import.meta.dir}/../../..`;

describe.skipIf(!source)("derive (real pack)", () => {
  test("the bake reproduces the pinned pack byte for byte", async () => {
    const ports = realPorts();
    const localDir = `${root}/local/test-derive`;
    rmSync(localDir, { recursive: true, force: true });
    const head = await loadHead(ports.fs, layoutAt(root, localDir), "bonsai-2-27b");
    if (!head.ok) throw new Error(head.message);
    await ports.fs.mkdirp(head.value.packsDir);
    await ports.fs.linkOrCopy(source!, head.value.sourcePath); // a hard link: no second 7 GB copy of the source
    try {
      const r = await new DerivePack(ports).run(head.value);
      expect(r).toEqual({
        ok: true,
        value: {
          path: head.value.servedPath,
          state: "derived",
          flipped: 22_674_442,
          digits: 5_908_725_760,
        },
      });
      expect(await ports.hasher.sha256File(head.value.servedPath)).toBe(head.value.served.sha256);
    } finally {
      rmSync(localDir, { recursive: true, force: true });
    }
  }, 600_000);
});
