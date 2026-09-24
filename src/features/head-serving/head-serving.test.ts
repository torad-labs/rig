import { describe, expect, test } from "bun:test";
import { putHead, withSidecarDraft } from "../../../test/fakes/head-fixtures.ts";
import { fakePorts } from "../../../test/fakes/index.ts";
import { loadEngine } from "../../shared/engine/engine.ts";
import { loadHead } from "../../shared/head/head.ts";
import { draftSidecar } from "../../shared/head/head-config.ts";
import { layoutAt } from "../../shared/layout.ts";
import { ExitCode } from "../../shared/result.ts";
import { defaultCacheRam, pickTier } from "./geometry.ts";
import golden from "./golden-argv.json";
import { ServeHead, type ServePlan, serveLogLine } from "./head-serving.service.ts";

const root = "/opt/rig"; // the golden was captured on the operator's checkout and re-rooted here; only the argv order and values are compared
const headToml = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();
const engineToml = await Bun.file(`${import.meta.dir}/../../../engine/engine.toml`).text();

async function setup(localDir = `${root}/local`, toml = headToml) {
  const p = fakePorts();
  const layout = layoutAt(root, localDir);
  putHead(p.fs, root, toml);
  p.fs.put(`${root}/engine/engine.toml`, engineToml);
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  return { p, head: head.value, engine: engine.value, uc: new ServeHead(p, engine.value) };
}

describe("argv", () => {
  test("GOLDEN: the plan for the 5080 at cache-ram 8192 is the live head's command line, byte for byte", async () => {
    // the live head (2026-09-20) served the pack from the repo root and the binary from bin/; the
    // plan below maps local/ onto those paths so only the argv order and values are compared
    const { head, engine, uc } = await setup();
    expect(head.runtime.lens?.enabled).toBe(false); // the golden was captured lens-off; a flipped switch left in head.toml must fail here
    const r = await uc.plan(head, { gpu: 0, cacheRam: 8192 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const mapped = r.value.argv.map((a) =>
      a
        .replace(`${root}/local/packs/bonsai-2-27b/`, `${root}/`)
        .replace(`${root}/local/engine-builds/${engine.sha7}-sm120/`, `${root}/bin/da69dc5-sm120/`)
        .replace(`${root}/heads/bonsai-2-27b/assets/`, `${root}/`),
    );
    expect(mapped).toEqual(golden as string[]);
    expect(r.value.env).toEqual({
      CUDA_DEVICE_ORDER: "PCI_BUS_ID",
      CUDA_VISIBLE_DEVICES: "0",
      LD_LIBRARY_PATH: `${root}/local/engine-builds/${engine.sha7}-sm120`,
    });
  });
  test("the in-pack MTP head's flags sit after the runtime args and before the geometry, on every tier", async () => {
    const { p, head, uc } = await setup();
    p.gpu.card(1, { name: "NVIDIA GeForce RTX 5090", memoryMiB: 32607 });
    const r = await uc.plan(head, { gpu: 1, cacheRam: 8192 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value).toMatchObject({ speculative: true, slots: 8, ctx: 786432 });
    const i = r.value.argv.indexOf("--spec-type");
    expect(r.value.argv.slice(i, i + 8)).toEqual([
      "--spec-type",
      "draft-mtp",
      "--spec-draft-n-max",
      "2",
      "-ctkd",
      "q4_0",
      "-ctvd",
      "q4_0",
    ]);
    expect(r.value.argv).not.toContain("-md"); // the head is in the pack
    expect(r.value.argv).not.toContain("--spec-draft-p-min");
    expect(r.value.argv).not.toContain("--spec-draft-chain-p-min");
    expect(r.value.argv.indexOf("--checkpoint-every")).toBeLessThan(i);
    expect(r.value.argv.indexOf("-c")).toBeGreaterThan(i);
    const local = await uc.plan(head, { gpu: 0, cacheRam: 8192 });
    expect(local.ok && local.value.speculative).toBe(true);
    expect(local.ok && local.value.argv).toContain("draft-mtp");
  });
  test("lens defaults off, can be omitted, and restores the whole bundle when enabled", async () => {
    const bundle = [
      "--lens-layers",
      "48,52,56,60,62,63",
      "--lens-out",
      "local/lens/bonsai-2-27b",
      "--lens-top",
      "12",
      "--lens-channels",
      "--pull-layers",
      "48,52,56,60,62,63",
      "--pull-action",
      "log",
    ];
    for (const toml of [
      headToml,
      headToml.replace(/\[runtime\.lens\][\s\S]*?(?=\[client\])/, ""),
    ]) {
      const { head, uc } = await setup(undefined, toml);
      const r = await uc.plan(head, { gpu: 0, cacheRam: 8192 });
      expect(r.ok).toBe(true);
      if (!r.ok) continue;
      for (const flag of bundle.filter((arg) => arg.startsWith("--")))
        expect(r.value.argv).not.toContain(flag);
      expect(r.value.argv.slice(-2)).toEqual(["-lv", "4"]);
      expect(r.value.argv).not.toContain("--spec-draft-mtp-decode-only");
      expect(r.value.argv).not.toContain("--spec-draft-mtp-window");
    }
    const { head, uc } = await setup(
      undefined,
      headToml.replace("enabled = false", "enabled = true"),
    );
    const r = await uc.plan(head, { gpu: 0, cacheRam: 8192 });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.argv.slice(-bundle.length - 2)).toEqual(["-lv", "4", ...bundle]);
  });
  test("enabled lens resolves head-relative asset paths", async () => {
    const { head, uc } = await setup(
      undefined,
      headToml
        .replace("enabled = false", "enabled = true")
        .replace("local/lens/bonsai-2-27b", "assets/lens"),
    );
    const r = await uc.plan(head, { gpu: 0, cacheRam: 8192 });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.argv).toContain(head.path("assets/lens"));
  });
  test("a sidecar draft names its file with -md and keeps it on the card with -ngld", async () => {
    const { p, head, uc } = await setup(undefined, withSidecarDraft(headToml));
    p.gpu.card(1, { name: "NVIDIA GeForce RTX 5090", memoryMiB: 32607 });
    const r = await uc.plan(head, { gpu: 1, cacheRam: 8192 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const i = r.value.argv.indexOf("--spec-type");
    expect(r.value.argv.slice(i, i + 8)).toEqual([
      "--spec-type",
      "draft-dflash",
      "-md",
      head.draftPath!,
      "--spec-draft-n-max",
      "3",
      "-ngld",
      "999",
    ]);
  });
});

describe("draft cutoffs", () => {
  test("a head that sets p_min and chain_p_min renders both after the draft length", async () => {
    const { head, uc } = await setup(
      undefined,
      headToml.replace(
        'type = "draft-mtp"\n',
        'type = "draft-mtp"\np_min = 0.5\nchain_p_min = 0.3\n',
      ),
    );
    const r = await uc.plan(head, { gpu: 0, cacheRam: 8192 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const i = r.value.argv.indexOf("--spec-draft-n-max");
    expect(r.value.argv.slice(i, i + 6)).toEqual([
      "--spec-draft-n-max",
      "2",
      "--spec-draft-p-min",
      "0.5",
      "--spec-draft-chain-p-min",
      "0.3",
    ]);
  });
});

describe("geometry", () => {
  test("the four measured cards land on their tiers", async () => {
    const { head } = await setup();
    for (const [vram, slots, ctx] of [
      [16303, 4, 294912],
      [32607, 8, 786432],
      [81559, 16, 2883584],
      [97887, 16, 3538944],
    ] as const) {
      const t = pickTier(head, vram);
      expect(t.ok && [t.value.slots, t.value.ctx]).toEqual([slots, ctx]);
    }
  });
  test("a card below the smallest tier is refused with exit 3", async () => {
    const { head } = await setup();
    const t = pickTier(head, 12000);
    expect(!t.ok && t.code).toBe(ExitCode.Unsupported);
  });
  test("cache-ram defaults to RAM/4 capped at 32 GiB", () => {
    expect(defaultCacheRam(62818)).toBe(15704);
    expect(defaultCacheRam(200000)).toBe(32768);
  });
});

describe("serve", () => {
  test("the log line names undrived when set, and stays silent about it otherwise", async () => {
    const { head } = await setup();
    const plan: ServePlan = {
      argv: [],
      env: {},
      binDir: "/v/engine-builds/abc123-sm120",
      gpu: 0,
      slots: 4,
      ctx: 294912,
      cacheRam: 8192,
      vramMiB: 16303,
      speculative: false,
    };
    expect(serveLogLine(head, plan)).not.toContain("UNDRIVED");
    const undrivedHead = { ...head, undrived: "the adapter is missing: serving the source pack" };
    expect(serveLogLine(undrivedHead, plan)).toContain(
      "UNDRIVED: the adapter is missing: serving the source pack",
    );
  });
});

describe("verify", () => {
  test("passes only with a complete build, the pinned pack and every asset", async () => {
    const { p, head, engine, uc } = await setup("/v");
    const bin = `/v/engine-builds/${engine.sha7}-sm120`;
    p.fs.put(`${bin}/llama-server`, "x"); // no marker yet
    expect((await uc.verify(head, 0)).ok).toBe(false);
    p.fs.put(`${bin}/BUILD`, "fork=… cap=sm_120");
    p.fs.put(head.servedPath, "pack");
    p.hasher.pinned.set(head.servedPath, head.served.sha256);
    let r = await uc.verify(head, 0);
    expect(!r.ok && r.message).toContain("asset");
    p.fs.put(head.path("assets/kv-mean-center-PQ2_0.gguf"), "b");
    p.fs.put(head.path("assets/chat-template.jinja"), "t");
    r = await uc.verify(head, 0);
    expect(r.ok).toBe(true);
    p.hasher.pinned.set(head.servedPath, "0".repeat(64));
    r = await uc.verify(head, 0);
    expect(!r.ok && r.message).toContain("sha256 differs");
  });
  test("verify checks the assets every list on the command line names, the enabled lens bundle included", async () => {
    const lensToml = headToml
      .replace("enabled = false", "enabled = true")
      .replace("local/lens/bonsai-2-27b", "assets/lens-out");
    const { p, head, engine, uc } = await setup("/v", lensToml);
    const bin = `/v/engine-builds/${engine.sha7}-sm120`;
    p.fs.put(`${bin}/llama-server`, "x");
    p.fs.put(`${bin}/BUILD`, "fork=… cap=sm_120");
    p.fs.put(head.servedPath, "pack");
    p.hasher.pinned.set(head.servedPath, head.served.sha256);
    p.fs.put(head.path("assets/kv-mean-center-PQ2_0.gguf"), "b");
    p.fs.put(head.path("assets/chat-template.jinja"), "t");
    let r = await uc.verify(head, 0);
    expect(!r.ok && r.message).toContain("asset assets/lens-out is missing");
    p.fs.put(head.path("assets/lens-out"), "");
    r = await uc.verify(head, 0);
    expect(r.ok).toBe(true);
  });
  test("a card whose tier loads a sidecar draft refuses to start without the pinned draft", async () => {
    const { p, head, engine, uc } = await setup("/v", withSidecarDraft(headToml));
    p.gpu.card(1, { name: "NVIDIA GeForce RTX 5090", memoryMiB: 32607 });
    const bin = `/v/engine-builds/${engine.sha7}-sm120`;
    p.fs.put(`${bin}/llama-server`, "x");
    p.fs.put(`${bin}/BUILD`, "fork=… cap=sm_120");
    p.fs.put(head.servedPath, "pack");
    p.hasher.pinned.set(head.servedPath, head.served.sha256);
    p.fs.put(head.path("assets/kv-mean-center-PQ2_0.gguf"), "b");
    p.fs.put(head.path("assets/chat-template.jinja"), "t");
    let r = await uc.verify(head, 1);
    expect(!r.ok && r.message).toContain("draft head is missing");
    p.fs.put(head.draftPath!, "draft");
    p.hasher.pinned.set(head.draftPath!, "0".repeat(64));
    r = await uc.verify(head, 1);
    expect(!r.ok && r.message).toContain("draft head is not the pinned bytes");
    p.hasher.pinned.set(head.draftPath!, draftSidecar(head.speculative!)!.sha256);
    r = await uc.verify(head, 1);
    expect(r.ok && r.value.draftPath).toBe(head.draftPath!);
  });
  test("verify names the undrived reason as a WARNING, pass or fail", async () => {
    const { p, engine, uc } = await setup("/v");
    const bin = `/v/engine-builds/${engine.sha7}-sm120`;
    p.fs.put(`${bin}/llama-server`, "x");
    p.fs.put(`${bin}/BUILD`, "fork=… cap=sm_120");
    await p.fs.remove(`${root}/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf`);
    const layout = layoutAt(root, "/v");
    const undrived = await loadHead(p.fs, layout, "bonsai-2-27b");
    if (!undrived.ok) throw new Error(undrived.message);
    expect(undrived.value.undrived).toBeDefined();
    const r = await uc.verify(undrived.value, 0); // fails on the missing source pack: still warns
    expect(r.ok).toBe(false);
    expect(p.log.lines).toContain(`warn ${undrived.value.undrived}`);
  });
  test("a --pack naming neither pinned pack is refused by name", async () => {
    const { p, head, engine, uc } = await setup("/v");
    const bin = `/v/engine-builds/${engine.sha7}-sm120`;
    p.fs.put(`${bin}/llama-server`, "x");
    p.fs.put(`${bin}/BUILD`, "fork=… cap=sm_120");
    const r = await uc.verify(head, 0, "/v/local/packs/bonsai-2-27b/some-other.gguf");
    expect(!r.ok && r.message).toContain("is neither the pinned source pack");
    expect(!r.ok && r.message).toContain(head.sourcePath);
    expect(!r.ok && r.message).toContain(head.declaredServed.path);
  });
  test("failing state (a): a unit rendered on the derived pack, then the adapter and the derived pack both disappear — verify without --pack passes on the source pack (the ExecStart bug); --pack the unit's own -m path catches the now-missing file", async () => {
    const { p, head, engine, uc } = await setup("/v");
    const bin = `/v/engine-builds/${engine.sha7}-sm120`;
    p.fs.put(`${bin}/llama-server`, "x");
    p.fs.put(`${bin}/BUILD`, "fork=… cap=sm_120");
    p.fs.put(head.path("assets/kv-mean-center-PQ2_0.gguf"), "b");
    p.fs.put(head.path("assets/chat-template.jinja"), "t");
    const packAtInstall = head.servedPath; // the unit was rendered while this machine served the derived pack
    p.fs.put(head.sourcePath, "source");
    p.hasher.pinned.set(head.sourcePath, head.source.sha256);
    p.fs.put(head.servedPath, "pack");
    p.hasher.pinned.set(head.servedPath, head.served.sha256);
    // the adapter and the derived pack both go missing: this machine goes undrived
    await p.fs.remove(`${root}/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf`);
    await p.fs.remove(head.servedPath);
    const layout = layoutAt(root, "/v");
    const undrived = await loadHead(p.fs, layout, "bonsai-2-27b");
    if (!undrived.ok) throw new Error(undrived.message);
    expect(undrived.value.undrived).toBeDefined();
    expect(undrived.value.servedPath).toBe(head.sourcePath);
    const noPack = await uc.verify(undrived.value, 0);
    expect(noPack.ok).toBe(true); // the bug: passes, though ExecStart's own -m no longer exists
    const withPack = await uc.verify(undrived.value, 0, packAtInstall);
    expect(!withPack.ok && withPack.message).toContain("missing");
    expect(!withPack.ok && withPack.message).toContain(packAtInstall);
  });
  test("failing state (b): a unit rendered on the source pack (undrived at install), then the adapter is pulled — verify without --pack refuses on the never-produced derived pack (the outage bug); --pack the unit's own -m path passes on the intact source pack, with a WARNING naming the drift and the remedy", async () => {
    const { p, engine, uc } = await setup("/v");
    const bin = `/v/engine-builds/${engine.sha7}-sm120`;
    p.fs.put(`${bin}/llama-server`, "x");
    p.fs.put(`${bin}/BUILD`, "fork=… cap=sm_120");
    const layout = layoutAt(root, "/v");
    await p.fs.remove(`${root}/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf`);
    const atInstall = await loadHead(p.fs, layout, "bonsai-2-27b");
    if (!atInstall.ok) throw new Error(atInstall.message);
    expect(atInstall.value.undrived).toBeDefined();
    const packAtInstall = atInstall.value.servedPath; // the source path, at install
    p.fs.put(atInstall.value.sourcePath, "source");
    p.hasher.pinned.set(atInstall.value.sourcePath, atInstall.value.source.sha256);
    p.fs.put(atInstall.value.path("assets/kv-mean-center-PQ2_0.gguf"), "b");
    p.fs.put(atInstall.value.path("assets/chat-template.jinja"), "t");
    // the adapter is pulled later: no longer undrived, but the derived pack is not on disk yet
    p.fs.put(`${root}/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf`, "adapter");
    const now = await loadHead(p.fs, layout, "bonsai-2-27b");
    if (!now.ok) throw new Error(now.message);
    expect(now.value.undrived).toBeUndefined();
    expect(now.value.servedPath).not.toBe(packAtInstall);
    const noPack = await uc.verify(now.value, 0);
    expect(!noPack.ok && noPack.message).toContain("missing"); // the bug: an outage though -m is intact
    const withPack = await uc.verify(now.value, 0, packAtInstall);
    expect(withPack.ok).toBe(true);
    expect(
      p.log.lines.some(
        (l) =>
          l.startsWith("warn") && l.includes(packAtInstall) && l.includes(now.value.servedPath),
      ),
    ).toBe(true);
    expect(p.log.lines.some((l) => l.includes("rig derive"))).toBe(true);
  });
});
