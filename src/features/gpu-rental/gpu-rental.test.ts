import { describe, expect, test } from "bun:test";
import { putHead } from "../../../test/fakes/head-fixtures.ts";
import { fakePorts } from "../../../test/fakes/index.ts";
import { loadEngine } from "../../shared/engine/engine.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import type { Offer } from "../../shared/ports/index.ts";
import { ok } from "../../shared/result.ts";
import { type LiveGate, RentGpu } from "./gpu-rental.service.ts";
import { loadVastConfig, offerQuery } from "./rental-config.ts";
import { gpuRentalCommand, SUBCOMMAND_FLAGS } from "./vast.command.ts";

const root = `${import.meta.dir}/../../..`;
const headToml = await Bun.file(`${root}/heads/bonsai-2-27b/head.toml`).text();
const engineToml = await Bun.file(`${root}/engine/engine.toml`).text();
// the pin every expected path below is built from, read once from engine.toml like the use case does
const engine = await (async () => {
  const p = fakePorts();
  p.fs.put("/r/engine/engine.toml", engineToml);
  const e = await loadEngine(p.fs, layoutAt("/r"));
  if (!e.ok) throw new Error(e.message);
  return e.value;
})();
const vastToml = await Bun.file(`${root}/vast.toml`).text();

const h100: Offer = {
  id: 50262229,
  gpu: "H100 SXM",
  gpus: 1,
  gpuRamMiB: 81559,
  computeCap: "90",
  dph: 1.975,
  geo: "Germany, DE",
  cpu: "AMD EPYC 9454",
  ramGiB: 92,
  bandwidth: 2887,
  cudaMaxGood: 13.2,
  reliability: 0.99,
  downMbps: 219,
};

/** boxServes: the pack the box answers /props with — its public pack, the default a box derives
 *  without the private adapter, or this machine's served pack (--private) */
async function setup(boxServes: "public" | "served" = "public") {
  const p = fakePorts();
  const layout = layoutAt("/r");
  putHead(p.fs, "/r", headToml); // ships the adapter: this fixture's head is drived, not undrived
  p.fs.put("/r/engine/engine.toml", engineToml);
  p.fs.put("/r/vast.toml", vastToml);
  p.fs.put("/r/dist/rig", "binary");
  p.fs.put("/home/u/.ssh/id_ed25519.pub", "ssh-ed25519 AAAAKEY marcos");
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  const gateRuns: Array<{ live: string; only: string[] }> = [];
  const gate: LiveGate = {
    run: async (_h, o) => {
      gateRuns.push(o);
      return ok({ dir: "/r/local/gate-runs/bonsai-2-27b/live", pass: true });
    },
  };
  p.rental.offers = [h100];
  p.shell.on(/^tar /, { code: 0, stdout: "", stderr: "" });
  p.http.json(/8100\/health$/, { status: "ok" });
  const boxPack = boxServes === "served" ? head.value.served.file : head.value.public?.file;
  p.http.json(/8100\/props$/, {
    model_path: `/workspace/rig/local/packs/bonsai-2-27b/${boxPack}`,
    total_slots: 16,
  });
  const deps = { ...p, gate, self: ["/r/dist/rig"], home: "/home/u" };
  const uc = new RentGpu(deps, layout, engine);
  return { p, head: head.value, uc, gateRuns, layout, deps };
}

describe("vast.toml", () => {
  test("parses; the query carries the class's bandwidth floor and price ceiling", async () => {
    const { p, layout } = await setup();
    const cfg = await loadVastConfig(p.fs, layout);
    expect(cfg.ok).toBe(true);
    if (!cfg.ok) return;
    expect(offerQuery(cfg.value, "H100_SXM", { diskGb: 40 })).toBe(
      "gpu_name=H100_SXM num_gpus=1 verified=true rentable=true reliability>0.98 cuda_max_good>=13.0 cpu_ram>=48 inet_down>=200 direct_port_count>=2 disk_space>=40 gpu_mem_bw>=2700 dph_total<=2.9",
    );
    expect(offerQuery(cfg.value, "RTX_4090", { diskGb: 40, maxDph: 0.5, geo: "US" })).toContain(
      "gpu_mem_bw>=0 dph_total<=0.5 geolocation=US",
    );
    // a multi-GPU box is the same query with num_gpus raised — and the count lives in ONE place,
    // so asking for four can never also ask for one
    const four = offerQuery(cfg.value, "RTX_5090", { diskGb: 40, gpus: 4 });
    expect(four).toContain("num_gpus=4");
    expect(four).not.toContain("num_gpus=1");
    const one = offerQuery(cfg.value, "RTX_5090", { diskGb: 40 });
    expect(one.split("num_gpus=1").length - 1).toBe(1);
    // the class ceiling is written per card; a box of four costs four times it. An explicit
    // --max-price is the box price the operator named and is not scaled.
    expect(four).toContain("dph_total<=2.8");
    expect(offerQuery(cfg.value, "RTX_5090", { diskGb: 40, gpus: 4, maxDph: 2.0 })).toContain(
      "dph_total<=2",
    );
  });
});

describe("vast up", () => {
  test("dry run: funds, key, the cheapest matching offer, nothing created", async () => {
    const { p, head, uc } = await setup();
    const r = await uc.up(head, { gpu: "H100_SXM", dryRun: true });
    expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(50262229);
    expect(p.rental.ops).toEqual([
      "register-key",
      `search ${offerQuery((await loadVastConfig(p.fs, layoutAt("/r"))).ok ? ((await loadVastConfig(p.fs, layoutAt("/r"))) as { ok: true; value: never }).value : ({} as never), "H100_SXM", { diskGb: 40 })}`,
    ]);
    expect(p.rental.instances.size).toBe(0);
    expect(p.fs.text("/r/local/rented-box/offers.json")).toContain("50262229");
  });
  test("rents, ships rig + head (its private adapter kept here) + engine pin, brings the head up on the box with rig's own steps, opens the tunnel and arms the idle timer", async () => {
    const { p, head, uc } = await setup();
    p.ssh.on(/rig build/, { code: 0, stdout: "", stderr: "" });
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(r.ok).toBe(true);
    if (!r.ok || r.value.kind !== "up") return;
    expect(r.value).toMatchObject({
      instanceId: 1000,
      gpu: "H100 SXM",
      cap: "90",
      sshHost: "ssh5.vast.ai",
      sshPort: 12345,
      localUrl: "http://127.0.0.1:8100",
      serving: { model: head.public?.file, slots: 16 },
    });
    expect(p.rental.ops.at(-1)).toBe("create 50262229 nvidia/cuda:13.0.3-devel-ubuntu24.04 40");
    // a rented box is someone else's machine: the adapter stays here, the box derives the public pack
    expect(p.shell.calls.find((c) => c[0] === "tar")).toEqual([
      "tar",
      "-C",
      "/r",
      "--exclude=heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf",
      "-czf",
      "/r/local/rented-box/payload.tar.gz",
      "dist/rig",
      "heads/bonsai-2-27b",
      "engine/engine.toml",
    ]);
    expect(p.ssh.pushed).toEqual([
      ["/r/local/rented-box/payload.tar.gz", "/workspace/rig/payload.tar.gz"],
    ]);
    const rigCalls = p.ssh.calls
      .filter((c) => c.includes("/workspace/rig/dist/rig "))
      .map((c) => c.replace(/.*\/dist\/rig /, "").split(" >>")[0]);
    expect(rigCalls).toEqual([
      "prepare --gpu 0",
      "fetch bonsai-2-27b",
      "build --gpu 0 --portable",
      "derive bonsai-2-27b --json",
      "serve bonsai-2-27b --gpu 0",
    ]);
    expect(p.ssh.pulled).toEqual([
      [
        `/workspace/rig/local/engine-builds/engine-sm90-${engine.sha7}.tar.gz`,
        `/r/local/rented-box/cached-builds/engine-sm90-${engine.sha7}.tar.gz`,
      ],
    ]);
    expect(p.fs.text("/r/local/rented-box/tunnel.env")).toBe("HOST=ssh5.vast.ai\nPORT=12345\n");
    expect(JSON.parse(p.fs.text("/r/local/rented-box/instance.json")!)).toMatchObject({
      instanceId: 1000,
      cap: "90",
      head: "bonsai-2-27b",
      sshPort: 12345,
    });
    expect(p.systemd.ops).toEqual([
      "daemon-reload",
      "restart rig-vast-tunnel.service",
      "enable rig-vast-idle.timer",
      "restart rig-vast-idle.timer",
    ]);
    // each unit replaced by a rename: a reload another process triggers never reads half a file
    expect(p.fs.replaced).toEqual(
      ["rig-vast-tunnel.service", "rig-vast-idle.service", "rig-vast-idle.timer"].map(
        (unit) => `/home/u/.config/systemd/user/${unit}`,
      ),
    );
    expect(p.fs.text("/home/u/.config/systemd/user/rig-vast-tunnel.service")).toContain(
      // biome-ignore lint/suspicious/noTemplateCurlyInString: systemd expands ${PORT} from the EnvironmentFile
      "-L 127.0.0.1:8100:127.0.0.1:8099 -p ${PORT} root@${HOST}",
    );
    expect(p.fs.text("/home/u/.config/systemd/user/rig-vast-idle.service")).toContain(
      "ExecStart=/r/dist/rig vast idle-check",
    );
  });
  test("--private ships the adapter, and the box must serve this machine's pack", async () => {
    const { p, head, uc } = await setup("served");
    const r = await uc.up(head, { gpu: "H100_SXM", private: true });
    expect(r.ok && r.value.kind === "up" && r.value.serving.model).toBe(head.served.file);
    expect(p.shell.calls.find((c) => c[0] === "tar")?.some((a) => a.startsWith("--exclude"))).toBe(
      false,
    );
    // and the box answering with its public pack is refused as a lesser pack than asked for
    const other = await setup("public");
    const refused = await other.uc.up(other.head, { gpu: "H100_SXM", private: true });
    expect(!refused.ok && refused.message).toContain(`not the pinned ${head.served.file}`);
  });
  test("a card the pin publishes a prebuilt for installs it on the box: no compile, no cached build shipped", async () => {
    const { p, head, uc } = await setup();
    expect(engine.prebuiltFor("120")).toBeDefined(); // engine.toml pins one; the case below depends on it
    p.rental.offers = [{ ...h100, gpu: "RTX 5090", computeCap: "120", bandwidth: 1790, dph: 0.6 }];
    p.fs.put(`/r/local/rented-box/cached-builds/engine-sm120-${engine.sha7}.tar.gz`, "tgz");
    const r = await uc.up(head, { gpu: "RTX_5090" });
    expect(r.ok).toBe(true);
    const rigCalls = p.ssh.calls
      .filter((c) => c.includes("/workspace/rig/dist/rig "))
      .map((c) => c.replace(/.*\/dist\/rig /, "").split(" >>")[0] ?? "");
    expect(rigCalls).toContain("build --gpu 0");
    expect(rigCalls.some((c) => c.includes("--portable") || c.includes("--from-tarball"))).toBe(
      false,
    );
    expect(p.ssh.pushed.map((x) => x[1])).toEqual(["/workspace/rig/payload.tar.gz"]);
    expect(p.ssh.pulled).toEqual([]);
  });
  test("a cached tarball for the card's sm is pushed and used instead of a build", async () => {
    const { p, head, uc } = await setup();
    p.fs.put(`/r/local/rented-box/cached-builds/engine-sm90-${engine.sha7}.tar.gz`, "tgz");
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(r.ok).toBe(true);
    expect(p.ssh.pushed.map((x) => x[1])).toEqual([
      "/workspace/rig/payload.tar.gz",
      `/workspace/rig/local/engine-builds/engine-sm90-${engine.sha7}.tar.gz`,
    ]);
    expect(
      p.ssh.calls.some((c) =>
        c.includes(
          `build --gpu 0 --from-tarball /workspace/rig/local/engine-builds/engine-sm90-${engine.sha7}.tar.gz`,
        ),
      ),
    ).toBe(true);
    expect(p.ssh.pulled).toEqual([]);
  });
  test("an offer whose tier the head cannot serve on is refused before anything is rented, dry run included", async () => {
    const { p, head: served, uc } = await setup();
    // the H100 tier in a K/V pair the pinned engine has no CUDA flash attention for
    putHead(
      p.fs,
      "/r",
      headToml.replace("ctx = 2883584\n", 'ctx = 2883584\ncache = { v = "q4_1" }\n'),
    );
    const loaded = await loadHead(p.fs, layoutAt("/r"), "bonsai-2-27b");
    if (!loaded.ok) throw new Error(loaded.message);
    for (const dryRun of [true, false]) {
      const r = await uc.up(loaded.value, { gpu: "H100_SXM", dryRun });
      expect(!r.ok && r.code).toBe(3);
      expect(!r.ok && r.message).toContain("no CUDA flash attention for K/V q4_0/q4_1");
    }
    p.rental.offers = [{ ...h100, gpu: "RTX 3060", gpuRamMiB: 12288, computeCap: "90" }];
    const small = await uc.up(served, { gpu: "RTX_3060", dryRun: true });
    expect(!small.ok && small.message).toContain("REFUSING to rent RTX 3060: bonsai-2-27b");
    expect(p.rental.ops.some((op) => op.startsWith("create"))).toBe(false);
  });
  test("an unmeasured card is refused with exit 3 unless --allow-arch; a second box is refused; no funds is refused", async () => {
    const { p, head, uc } = await setup();
    p.rental.offers = [{ ...h100, gpu: "RTX 4090", computeCap: "89" }];
    let r = await uc.up(head, { gpu: "RTX_4090", dryRun: true });
    expect(!r.ok && r.code).toBe(3);
    r = await uc.up(head, { gpu: "RTX_4090", dryRun: true, allowArch: "89" });
    expect(r.ok).toBe(true);
    p.fs.put("/r/local/rented-box/instance.json", JSON.stringify({ instanceId: 7 }));
    r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("already exists"); // a real rental, not a dry run
    await p.fs.remove("/r/local/rented-box/instance.json");
    p.rental.balance = 0.5;
    r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("top up");
    expect(p.rental.instances.size).toBe(0);
  });
  test("a box that never runs, or never answers ssh, is destroyed and the state cleared", async () => {
    const { p, head, uc } = await setup();
    p.rental.statuses = ["loading"];
    let r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("never reached running");
    expect(p.rental.ops.at(-1)).toBe("destroy 1000");
    expect(await p.fs.exists("/r/local/rented-box/instance.json")).toBe(false);
    expect(p.clock.slept.filter((ms) => ms === 10_000).length).toBe(90);
    const s2 = await setup();
    s2.p.ssh.on(/^true$/, { code: 255, stdout: "", stderr: "refused" });
    r = await s2.uc.up(s2.head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("ssh never answered");
    expect(s2.p.rental.instances.size).toBe(0);
  });
  test("a failed remote step leaves the box running for inspection and says so", async () => {
    const { p, head, uc } = await setup();
    p.ssh.on(/rig derive/, {
      code: 1,
      stdout: "",
      stderr: "the derive step produced a DIFFERENT edit",
    });
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("derive failed on box 1000");
    expect(p.rental.instances.size).toBe(1);
    expect(p.log.lines.join("\n")).toContain("DIFFERENT edit");
  });
  test("READY refuses when the box serves a different pack than this local head resolves, box left running for inspection", async () => {
    const { p, head, uc } = await setup();
    p.http.json(/8100\/props$/, {
      model_path: "/workspace/rig/local/packs/bonsai-2-27b/some-other-pack.gguf",
      total_slots: 16,
    });
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.code).toBe(1);
    expect(!r.ok && r.message).toContain(
      `the box serves some-other-pack.gguf, not the pinned ${head.public?.file}`,
    );
    expect(!r.ok && r.message).toContain("box 1000 left running for inspection");
    expect(p.rental.instances.size).toBe(1); // left running, not destroyed
  });
  test("the box's own derive going undrived (its own missing adapter, not ours) is warned here, not swallowed like every other step's stdout", async () => {
    // the source pack ("undrived") or the [public] pack, derived there ("derived"): a lesser pack either way
    for (const [state, file] of [
      ["undrived", "source.gguf"],
      ["derived", "public.gguf"],
    ]) {
      const { p, head, uc } = await setup();
      p.ssh.on(/rig derive/, {
        code: 0,
        stdout: JSON.stringify({
          path: `/workspace/rig/local/packs/bonsai-2-27b/${file}`,
          state,
          reason: "the [derive] adapter assets/lora/bonsai-abliterate-lora.gguf is not on the box",
        }),
        stderr: "",
      });
      const r = await uc.up(head, { gpu: "H100_SXM" });
      expect(r.ok).toBe(true);
      expect(
        p.log.lines.some(
          (l) => l.startsWith("warn") && l.includes("box 1000") && l.includes("is not on the box"),
        ),
      ).toBe(true);
    }
  });
});

describe("vast down / status / idle", () => {
  test("down destroys, re-reads the listing, reports the bill and clears the state; --all sweeps forgotten boxes", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    p.clock.t += 2 * 3_600_000;
    p.rental.instances.set(55, { id: 55, status: "running", label: "rig", dph: 1 });
    p.rental.instances.set(56, { id: 56, status: "running", label: "other", dph: 1 });
    const r = await uc.down({ all: true });
    expect(r).toEqual({ ok: true, value: { destroyed: [1000, 55], hours: 2, cost: 3.95 } });
    expect(await p.fs.exists("/r/local/rented-box/instance.json")).toBe(false);
    expect(p.systemd.ops.slice(-3)).toEqual([
      "disable rig-vast-idle.timer",
      "stop rig-vast-idle.timer",
      "stop rig-vast-tunnel.service",
    ]);
  });
  test("down refuses to believe a destroy the listing contradicts, and leaves the box's cost control running", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    p.rental.destroy = async () => {}; // vast says nothing, the box stays listed
    const before = p.systemd.ops.length;
    const r = await uc.down();
    expect(!r.ok && r.message).toContain("STILL listed");
    expect(await p.fs.exists("/r/local/rented-box/instance.json")).toBe(true);
    expect(p.systemd.ops.slice(before)).toEqual([]);
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(true);
  });
  test("down: a destroy vast refuses (a 429, a 5xx) keeps the state and the idle timer; a box vast no longer lists is gone whatever the call answers", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    p.rental.destroy = async () => {
      throw new Error("vastai destroy instance 1000: 429 Too Many Requests");
    };
    const before = p.systemd.ops.length;
    const refused = await uc.down();
    expect(!refused.ok && refused.message).toContain(
      "box 1000 is STILL listed after destroy (the destroy failed: vastai destroy instance 1000: 429 Too Many Requests) — it is billing, and its idle timer stays armed",
    );
    expect(await p.fs.exists("/r/local/rented-box/instance.json")).toBe(true);
    expect(p.systemd.ops.slice(before)).toEqual([]);
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(true);
    // destroyed from vast's console meanwhile: the next down's call refuses the gone id
    p.rental.instances.delete(1000);
    const gone = await uc.down();
    expect(gone.ok && gone.value.destroyed).toEqual([1000]);
    expect(await p.fs.exists("/r/local/rented-box/instance.json")).toBe(false);
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(false);
  });
  test("an engine.toml this binary cannot read stops up and bench, never the cost control of a box that bills", async () => {
    const { p, head, uc, deps, layout } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    // the checkout moves on to a pin this binary cannot read (2026-09-25, 8:59-11:09 PM CT:
    // fourteen idle checks exited 1 on `miscompilers: Invalid key` while box 52647843 billed)
    p.fs.put("/r/engine/engine.toml", `${engineToml}\n[future]\nkey = 1\n`);
    const unreadable = await loadEngine(p.fs, layout);
    if (unreadable.ok) throw new Error("fixture: the unknown key loaded");
    const stale = new RentGpu(deps, layout, unreadable);
    expect(await stale.up(head, { gpu: "H100_SXM", dryRun: true })).toEqual(unreadable);
    expect(await stale.bench(head)).toEqual(unreadable);
    expect(await stale.status()).toMatchObject({ ok: true, value: { listed: true } });
    p.http.on(/8100\/metrics$/, () => ({
      status: 200,
      text: "llamacpp:prompt_tokens_total 1\nllamacpp:tokens_predicted_total 1\nllamacpp:requests_processing 0\n",
    }));
    expect(await stale.idleCheck()).toEqual({ ok: true, value: { action: "changed" } });
    p.clock.t += 45 * 60_000;
    expect(await stale.idleCheck()).toEqual({
      ok: true,
      value: { action: "destroyed", idleMinutes: 45 },
    });
    expect(p.rental.instances.size).toBe(0);
  });
  test("idle-check: counters unchanged for idle_minutes destroy the box; traffic or a busy slot resets the clock", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    let metrics =
      "llamacpp:prompt_tokens_total 100\nllamacpp:tokens_predicted_total 50\nllamacpp:requests_processing 0\n";
    p.http.on(/8100\/metrics$/, () => ({ status: 200, text: metrics }));
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "changed" } });
    p.clock.t += 20 * 60_000;
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "idle", idleMinutes: 20 } });
    metrics = metrics.replace("100", "160");
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "changed" } });
    p.clock.t += 44 * 60_000;
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "idle", idleMinutes: 44 } });
    metrics = metrics.replace("requests_processing 0", "requests_processing 1");
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
    metrics = metrics.replace("requests_processing 1", "requests_processing 0");
    p.clock.t += 45 * 60_000;
    expect(await uc.idleCheck()).toEqual({
      ok: true,
      value: { action: "destroyed", idleMinutes: 45 },
    });
    expect(p.rental.instances.size).toBe(0);
    expect(await p.fs.exists("/r/local/rented-box/instance.json")).toBe(false);
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "no-box" } });
  });
  test("--disk-gb sizes the query and the box; --idle-minutes is this box's own budget, read by every idle check", async () => {
    const { p, head, uc } = await setup();
    const r = await uc.up(head, { gpu: "H100_SXM", diskGb: 300, idleMinutes: 720 });
    expect(r.ok).toBe(true);
    expect(p.rental.ops.find((op) => op.startsWith("search "))).toContain("disk_space>=300");
    expect(p.rental.ops.at(-1)).toBe("create 50262229 nvidia/cuda:13.0.3-devel-ubuntu24.04 300");
    expect(JSON.parse(p.fs.text("/r/local/rented-box/instance.json")!)).toMatchObject({
      idleMinutes: 720,
    });
    expect(p.fs.text("/home/u/.config/systemd/user/rig-vast-idle.service")).toContain(
      "after 720 idle minutes",
    );
    // the READY line prices the budget: what the box can bill idle before it is destroyed
    expect(p.log.lines.join("\n")).toContain("720 min: up to ~$23.70 idle before it is destroyed");
    p.http.on(/8100\/metrics$/, () => ({
      status: 200,
      text: "llamacpp:prompt_tokens_total 1\nllamacpp:tokens_predicted_total 1\nllamacpp:requests_processing 0\n",
    }));
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "changed" } });
    p.clock.t += 45 * 60_000; // vast.toml's 45 would destroy it here
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "idle", idleMinutes: 45 } });
    p.clock.t += 675 * 60_000;
    expect(await uc.idleCheck()).toEqual({
      ok: true,
      value: { action: "destroyed", idleMinutes: 720 },
    });
  });
  test("idle-check: a server that does not answer is no evidence of idleness; a busy card keeps the box", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    // no /metrics route: the box never served (it runs other work on its card)
    const listed = p.rental.instances.get(1000)!;
    p.rental.instances.set(1000, { ...listed, gpuUtil: 97 });
    for (let check = 0; check < 80; check++) {
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
      p.clock.t += 10 * 60_000;
    }
    expect(p.rental.instances.size).toBe(1);
    p.rental.instances.set(1000, { ...listed, gpuUtil: 0 });
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "idle", idleMinutes: 10 } });
    p.clock.t += 35 * 60_000;
    expect(await uc.idleCheck()).toEqual({
      ok: true,
      value: { action: "destroyed", idleMinutes: 45 },
    });
    expect(p.rental.instances.size).toBe(0);
  });
  test("idle-check: neither the server nor the market readable counts nothing, fails the run, and never destroys", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    // no /metrics route (the server stopped for `vast bench`'s gates) and vast answering 429s
    const show = p.rental.show.bind(p.rental);
    p.rental.show = async () => {
      throw new Error("vastai show instance 1000: 429 Too Many Requests");
    };
    for (let check = 0; check < 80; check++) {
      const r = await uc.idleCheck();
      expect(!r.ok && r.message).toBe(
        "neither the server nor vast answered: box 1000 not counted, its idle clock unchanged",
      );
      p.clock.t += 10 * 60_000;
    }
    expect(p.rental.instances.size).toBe(1);
    expect(p.log.lines.join("\n")).toContain(
      "warn vast could not be read: vastai show instance 1000: 429",
    );
    // the market back: the card's own reading decides again
    p.rental.show = show;
    const listed = p.rental.instances.get(1000)!;
    p.rental.instances.set(1000, { ...listed, gpuUtil: 97 });
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
  });
  test("idle-check: a card the market did not read is no evidence it is idle, whatever the server says", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    // vast lists the box running with no utilization sample (its gpu_util is number | null) and
    // the server does not answer: the 2026-09-24 box, its card busy with other work
    const { gpuUtil: _, ...unsampled } = p.rental.instances.get(1000)!;
    p.rental.instances.set(1000, unsampled);
    for (let check = 0; check < 80; check++) {
      const r = await uc.idleCheck();
      expect(!r.ok && r.message).toBe(
        "the server did not answer and vast listed no GPU reading: box 1000 not counted, its idle clock unchanged",
      );
      p.clock.t += 10 * 60_000;
    }
    // the server answers, idle, but the card is still unread: not counted either
    p.http.on(/8100\/metrics$/, () => ({
      status: 200,
      text: "llamacpp:prompt_tokens_total 1\nllamacpp:tokens_predicted_total 1\nllamacpp:requests_processing 0\n",
    }));
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "changed" } });
    for (let check = 0; check < 80; check++) {
      p.clock.t += 10 * 60_000;
      const r = await uc.idleCheck();
      expect(!r.ok && r.message).toBe(
        "the server is idle but vast listed no GPU reading: box 1000 not counted, its idle clock unchanged",
      );
    }
    // vast unreadable while the server is idle: the card unread the same way
    const show = p.rental.show.bind(p.rental);
    p.rental.show = async () => {
      throw new Error("vastai show instance 1000: 429 Too Many Requests");
    };
    const r = await uc.idleCheck();
    expect(!r.ok && r.message).toBe(
      "the server is idle but vast could not be read: box 1000 not counted, its idle clock unchanged",
    );
    expect(p.rental.instances.size).toBe(1);
    // the card read idle: the idle clock decides again, from the server's last change
    p.rental.show = show;
    p.rental.instances.set(1000, { ...unsampled, gpuUtil: 0 });
    expect(await uc.idleCheck()).toMatchObject({ ok: true, value: { action: "destroyed" } });
    expect(p.rental.instances.size).toBe(0);
  });
  test("status reads the state, the market and the tunnel", async () => {
    const { p, head, uc } = await setup();
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { box: null, listed: false, healthy: true },
    });
    await uc.up(head, { gpu: "H100_SXM" });
    p.clock.t += 3_600_000;
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { listed: true, hours: 1, cost: 1.98 },
    });
  });
  test("status re-arms the idle timer of a box that bills without it, and says so; a box gone from the listing is left alone", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    expect(await uc.status()).toMatchObject({ ok: true, value: { idleTimer: "active" } });
    p.systemd.active.delete("rig-vast-idle.timer"); // 2026-09-24: dead at 09:08, no stop logged
    const before = p.systemd.ops.length;
    expect(await uc.status()).toMatchObject({ ok: true, value: { idleTimer: "re-armed" } });
    expect(p.systemd.ops.slice(before)).toEqual([
      "enable rig-vast-idle.timer",
      "restart rig-vast-idle.timer",
    ]);
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(true);
    expect(p.log.lines.join("\n")).toContain("rig-vast-idle.timer was not running: re-armed");
    // destroyed elsewhere: nothing bills, so nothing is armed
    p.systemd.active.delete("rig-vast-idle.timer");
    p.rental.instances.delete(1000);
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { listed: false, idleTimer: "inactive" },
    });
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(false);
  });
  test("status: a market that cannot be read is no evidence the box is gone, so a dead timer is re-armed", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    p.systemd.active.delete("rig-vast-idle.timer");
    p.rental.show = async () => {
      throw new Error("vastai show instance 1000: 401 key expired");
    };
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { listed: "unread", idleTimer: "re-armed" },
    });
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(true);
    expect(p.log.lines.join("\n")).toContain(
      "box 1000 may be billing and rig-vast-idle.timer was not running: re-armed",
    );
  });
  test("status reports an active timer whose last check failed: a check that exits 1 controls nothing", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { idleTimer: "active", idleCheck: "ok" },
    });
    p.systemd.results.set("rig-vast-idle.service", "exit-code");
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { idleTimer: "active", idleCheck: "failed" },
    });
    expect(p.log.lines.join("\n")).toContain(
      "rig-vast-idle.service's last run ended exit-code: the box's cost control is not running",
    );
    p.systemd.results.set("rig-vast-idle.service", null);
    expect(await uc.status()).toMatchObject({ ok: true, value: { idleCheck: "unread" } });
  });
  test("a dry run prices the next box while one is still up", async () => {
    const { p, uc, head } = await setup();
    p.fs.put(
      "/r/local/rented-box/instance.json",
      JSON.stringify({
        instanceId: 1,
        offerId: 1,
        gpu: "RTX PRO 6000 Max-Q",
        cap: "120",
        dph: 1,
        geo: "CZ",
        createdAt: 0,
      }),
    );
    const r = await uc.up(head, { gpu: "RTX_5090", gpus: 4, dryRun: true });
    expect(r.ok && r.value.kind === "dry-run").toBe(true);
    expect(r.ok && r.value.kind === "dry-run" ? r.value.query : "").toContain("num_gpus=4");
  });

  test("starting the server releases the ssh channel and records the server's own pid", async () => {
    const { p, uc, head } = await setup();
    await uc.up(head, { gpu: "RTX_5090" });
    const start = p.ssh.calls.find((c) => /rig serve bonsai-2-27b/.test(c))!;
    // the `&` backgrounds a SUBSHELL, and a subshell whose own stdout/stderr are still the ssh
    // channel's pipes never lets ssh see EOF: the bring-up hangs before it installs the tunnel.
    // Measured on box 51727683 (2026-09-20): /proc/<subshell>/fd/{1,2} -> pipe, ssh alive 17 min.
    expect(start).toMatch(/\}\s*>\s*\/dev\/null\s*2>&1\s*$/);
    // and $! must be the nohup'd server, not the subshell — stopRemote kills what this records
    expect(start).toMatch(/< \/dev\/null & echo \$! > \S+\/local\/server\.pid/);
  });

  test("bench runs the gates on the box with the server stopped, pulls the run, restarts, then the live probes through the tunnel", async () => {
    const { p, head, uc, gateRuns } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    p.ssh.on(/rig gate/, {
      code: 0,
      stdout: JSON.stringify({
        dir: "/workspace/rig/local/gate-runs/bonsai-2-27b/20260920T000000Z",
        pass: true,
      }),
      stderr: "",
    });
    const r = await uc.bench(head);
    expect(r).toEqual({
      ok: true,
      value: {
        remote: "/r/local/rented-box/pulled-runs/h100-sxm-1000",
        local: "/r/local/gate-runs/bonsai-2-27b/live",
        pass: true,
      },
    });
    const seq = p.ssh.calls
      .filter((c) => /kill -INT|rig gate|rig serve/.test(c))
      .map((c) => (/kill/.test(c) ? "stop" : /gate/.test(c) ? "gate" : "serve"));
    expect(seq).toEqual(["serve", "stop", "gate", "serve"]);
    expect(p.ssh.pulled.at(-1)).toEqual([
      "/workspace/rig/gate-run.tar.gz",
      "/r/local/rented-box/pulled-runs/h100-sxm-1000/gate-run.tar.gz",
    ]);
    expect(gateRuns).toEqual([
      { live: "http://127.0.0.1:8100", only: ["sessions", "concurrency"] },
    ]);
  });
});

describe("vast command", () => {
  test("a subcommand refuses another subcommand's flag before it acts: down --dry-run destroys nothing", async () => {
    const { p, head, uc } = await setup();
    p.fs.put("/r/local/rented-box/instance.json", JSON.stringify({ instanceId: 7 })); // a box to destroy
    const vast = gpuRentalCommand(uc, async () => ok(head), p.log);
    const code = await vast.run({ positionals: ["down"], flags: { "dry-run": true }, dashed: [] });
    expect(code).toBe(64);
    expect(p.log.lines.join("\n")).toContain("vast down does not take --dry-run");
    expect(p.rental.ops.some((op) => op.startsWith("destroy"))).toBe(false);
  });
  test("the subcommands' flags are exactly the ones the usage line names", async () => {
    const source = await Bun.file(`${import.meta.dir}/vast.command.ts`).text();
    const usage = /const USAGE =\s*"([^"]*)"/.exec(source)![1]!;
    const named = new Set([...usage.matchAll(/--([a-z][a-z-]*)/g)].map((m) => m[1]!));
    expect(new Set(Object.values(SUBCOMMAND_FLAGS).flat())).toEqual(named);
  });
});
