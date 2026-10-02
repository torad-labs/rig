import { describe, expect, test } from "bun:test";
import { lstatSync } from "node:fs";
import { join } from "node:path";
import { BunFileSystem } from "@rig/adapters";
import { fakePorts, repoRoot, sha256Of } from "@rig/testing";
import { excludedBy, loadPublicRules, type PublicRules, publicFile } from "./public-rules.ts";

const enc = (text: string) => new TextEncoder().encode(text);
const dec = (bytes: Uint8Array) => new TextDecoder().decode(bytes);
const BINARY = new Uint8Array([0x7f, 0x45, 0, 1, 2]);

const rules: PublicRules = {
  exclude: ["PRODUCT.md", "heads/*/evidence/lens-*/**"],
  rewrite: [{ from: "/home/op/private/lens", to: "<lens-dir>" }],
  deny: ["/home/op", "private-host"],
  deny_sha256: [sha256Of("a leaked key")],
  allow_binary_sha256: [new Bun.CryptoHasher("sha256").update(BINARY).digest("hex")],
};

describe("excludedBy", () => {
  test("names the glob that removes a path, and none for a path that ships", () => {
    expect(excludedBy(rules, "PRODUCT.md")).toBe("PRODUCT.md");
    expect(excludedBy(rules, "heads/b/evidence/lens-legs-1/x.json")).toBe(
      "heads/*/evidence/lens-*/**",
    );
    expect(excludedBy(rules, "heads/b/head.toml")).toBeUndefined();
  });
});

describe("publicFile", () => {
  test("a rewrite gives a private string its public form before the deny scan reads the line", () => {
    const out = publicFile(rules, "heads/b/head.toml", enc('args = ["/home/op/private/lens"]\n'));
    expect(dec(out.bytes)).toBe('args = ["<lens-dir>"]\n');
    expect(out.hits).toEqual([]);
  });
  test("a deny pattern no rewrite covers refuses, by path and line", () => {
    const out = publicFile(rules, "docs/a.md", enc("one\nssh private-host\n"));
    expect(out.hits).toEqual(["docs/a.md:2: private-host"]);
  });
  test("a path that names a private place refuses whatever the file holds", () => {
    expect(publicFile(rules, "private-host/notes.md", enc("clean")).hits).toEqual([
      "private-host/notes.md: the path itself holds private-host",
    ]);
  });
  test("denied content refuses by its sha256", () => {
    expect(publicFile(rules, "k.txt", enc("a leaked key")).hits).toEqual([
      `k.txt: denied content, sha256 ${sha256Of("a leaked key")}`,
    ]);
  });
  test("a binary ships unchanged only when named, and its raw bytes are still scanned", () => {
    const named = publicFile(rules, "a.bin", BINARY);
    expect(named).toEqual({ bytes: BINARY, hits: [] });
    const other = new Uint8Array([0, ...enc("/home/op/x")]);
    const unnamed = publicFile(rules, "b.bin", other);
    expect(unnamed.bytes).toBe(other);
    expect(unnamed.hits).toHaveLength(2);
    expect(unnamed.hits[0]).toContain("an unlisted binary");
    expect(unnamed.hits[1]).toBe("b.bin: binary holds /home/op");
  });
});

describe("loadPublicRules", () => {
  test("reads the rules, and refuses a file without them or with an empty deny list", async () => {
    const p = fakePorts();
    expect(await loadPublicRules(p.fs, "/r")).toMatchObject({ ok: false });
    p.fs.put(
      "/r/public-export.toml",
      'exclude = ["PRODUCT.md"]\ndeny = ["/home/op"]\ndeny_sha256 = []\nallow_binary_sha256 = []\n[[rewrite]]\nfrom = "a"\nto = "b"\n',
    );
    expect(await loadPublicRules(p.fs, "/r")).toEqual({
      ok: true,
      value: {
        exclude: ["PRODUCT.md"],
        rewrite: [{ from: "a", to: "b" }],
        deny: ["/home/op"],
        deny_sha256: [],
        allow_binary_sha256: [],
      },
    });
    p.fs.put(
      "/r/public-export.toml",
      "exclude = []\ndeny = []\ndeny_sha256 = []\nallow_binary_sha256 = []\nrewrite = []\n",
    );
    expect(await loadPublicRules(p.fs, "/r")).toMatchObject({ ok: false });
  });
});

// The export and `rig image` refuse a private string at publication; this finds it at the commit
// that adds it. Only the private repo carries public-export.toml (the public copy excludes it, and
// has nothing left to guard).
const root = repoRoot;
const guarded = await loadPublicRules(new BunFileSystem(), root);
describe.skipIf(!guarded.ok)("the tracked tree", () => {
  test("every file the rules do not remove clears them as it stands in the checkout", async () => {
    if (!guarded.ok) return;
    const tracked = Bun.spawnSync(["git", "-C", root, "ls-files", "-z"]).stdout.toString();
    const hits: string[] = [];
    for (const path of tracked.split("\0").filter(Boolean)) {
      if (excludedBy(guarded.value, path) !== undefined) continue;
      const file = join(root, path);
      const stat = lstatSync(file, { throwIfNoEntry: false });
      if (!stat?.isFile()) continue; // a submodule, a deleted file; the export reports symlinks
      const bytes = new Uint8Array(await Bun.file(file).arrayBuffer());
      hits.push(...publicFile(guarded.value, path, bytes).hits);
    }
    expect(hits).toEqual([]);
  });
});
