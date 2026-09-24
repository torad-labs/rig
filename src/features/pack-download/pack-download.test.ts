import { describe, expect, test } from "bun:test";
import { withSidecarDraft } from "../../../test/fakes/head-fixtures.ts";
import { fakePorts, sha256Of } from "../../../test/fakes/index.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { DownloadPack } from "./pack-download.service.ts";

const headToml = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();
const BYTES = "the pinned pack bytes";
const DRAFT = "the pinned draft head bytes";
/** the head with a sidecar draft (the in-pack head has nothing of its own to fetch) */
async function setup(draftPresent = true, toml = withSidecarDraft(headToml)) {
  const p = fakePorts();
  const layout = layoutAt("/r");
  p.fs.put(
    "/r/heads/bonsai-2-27b/head.toml",
    toml
      .replace(/^sha256 = "3cb3.*$/m, `sha256 = "${sha256Of(BYTES)}"`)
      .replace(/^sha256 = "9dd1.*$/m, `sha256 = "${sha256Of(DRAFT)}"`),
  );
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  if (!head.ok) throw new Error(head.message);
  if (draftPresent && head.value.draftPath) p.fs.put(head.value.draftPath, DRAFT);
  return { p, head: head.value, uc: new DownloadPack(p) };
}

describe("fetch", () => {
  test("a present, matching pack is left alone", async () => {
    const { p, head, uc } = await setup();
    p.fs.put(head.sourcePath, BYTES);
    const r = await uc.run(head);
    expect(r.ok && r.value.state).toBe("present");
    expect(r.ok && r.value.draft?.state).toBe("present");
    expect(p.shell.calls).toEqual([]);
  });
  test("the in-pack head has no draft of its own to fetch: the pack is the whole download", async () => {
    const { p, head, uc } = await setup(false, headToml);
    p.fs.put(head.sourcePath, BYTES);
    const r = await uc.run(head);
    expect(r.ok && r.value).toEqual({ path: head.sourcePath, state: "present" });
    expect(p.shell.calls).toEqual([]);
  });
  test("the draft head is fetched after the pack, by its own pin, and reported beside it", async () => {
    const { p, head, uc } = await setup(false);
    p.fs.put(head.sourcePath, BYTES);
    p.shell.on(/^curl/, (cmd) => {
      p.fs.put(cmd[cmd.indexOf("-o") + 1]!, cmd.at(-1)!.includes("DFlash2") ? DRAFT : BYTES);
      return { code: 0, stdout: "", stderr: "" };
    });
    const r = await uc.run(head);
    expect(r.ok && r.value).toEqual({
      path: head.sourcePath,
      state: "present",
      draft: { path: head.draftPath!, state: "fetched" },
    });
    expect(p.shell.calls.map((c) => c.at(-1))).toEqual([
      "https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/resolve/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b/Bonsai-2-27B-DFlash2-Q8_0.gguf",
    ]);
    expect(p.fs.renames).toEqual([[`${head.draftPath}.part`, head.draftPath!]]);
  });
  test("a draft head with the wrong bytes is removed and named, never renamed into place", async () => {
    const { p, head, uc } = await setup(false);
    p.fs.put(head.sourcePath, BYTES);
    p.shell.on(/^curl/, (cmd) => {
      p.fs.put(cmd[cmd.indexOf("-o") + 1]!, "something else");
      return { code: 0, stdout: "", stderr: "" };
    });
    const r = await uc.run(head);
    expect(!r.ok && r.message).toContain("draft head");
    expect(await p.fs.exists(head.draftPath!)).toBe(false);
    expect(await p.fs.exists(`${head.draftPath}.part`)).toBe(false);
  });
  test("downloads to .part and publishes by rename only after the sha matches", async () => {
    const { p, head, uc } = await setup();
    p.shell.on(/^curl/, (cmd) => {
      p.fs.put(cmd[cmd.indexOf("-o") + 1]!, BYTES);
      return { code: 0, stdout: "", stderr: "" };
    });
    const r = await uc.run(head);
    expect(r.ok && r.value.state).toBe("fetched");
    expect(p.fs.renames).toEqual([[`${head.sourcePath}.part`, head.sourcePath]]);
    expect(p.shell.calls[0]![p.shell.calls[0]!.length - 1]).toBe(
      "https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/resolve/efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
    );
  });
  test("a download with the wrong bytes is removed and reported, never renamed into place", async () => {
    const { p, head, uc } = await setup();
    p.shell.on(/^curl/, (cmd) => {
      p.fs.put(cmd[cmd.indexOf("-o") + 1]!, "something else");
      return { code: 0, stdout: "", stderr: "" };
    });
    const r = await uc.run(head);
    expect(r.ok).toBe(false);
    expect(await p.fs.exists(head.sourcePath)).toBe(false);
    expect(await p.fs.exists(`${head.sourcePath}.part`)).toBe(false);
  });
  test("--from adopts a local file only when its sha matches", async () => {
    const { p, head, uc } = await setup();
    p.fs.put("/elsewhere/pack.gguf", BYTES);
    p.fs.put("/elsewhere/other.gguf", "nope");
    expect((await uc.run(head, { from: "/elsewhere/other.gguf" })).ok).toBe(false);
    const r = await uc.run(head, { from: "/elsewhere/pack.gguf" });
    expect(r.ok && r.value.state).toBe("adopted");
    expect(p.fs.text(head.sourcePath)).toBe(BYTES);
  });
  test("a pack that is there but unreadable is refused by its errno, never fetched over", async () => {
    const { p, head, uc } = await setup();
    p.fs.put(head.sourcePath, BYTES);
    p.hasher.unreadable.set(head.sourcePath, "EACCES");
    const r = await uc.run(head);
    expect(!r.ok && r.message).toContain("unreadable (EACCES) — refusing to fetch over it");
    expect(p.shell.calls).toEqual([]);
    expect(p.fs.renames).toEqual([]);
  });
  test("prefers aria2c when present", async () => {
    const { p, head, uc } = await setup();
    p.shell.tools.add("aria2c");
    p.shell.on(/^aria2c/, (cmd) => {
      p.fs.put(`${cmd[cmd.indexOf("-d") + 1]}/${cmd[cmd.indexOf("-o") + 1]}`, BYTES);
      return { code: 0, stdout: "", stderr: "" };
    });
    const r = await uc.run(head);
    expect(r.ok && r.value.state).toBe("fetched");
  });
});
