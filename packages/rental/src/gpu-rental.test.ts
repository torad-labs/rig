import { describe, expect, test } from "bun:test";
import type { Offer, SshResult, SshTarget } from "@rig/core";
import { layoutAt, ok } from "@rig/core";
import { loadEngine } from "@rig/engine";
import { loadHead } from "@rig/head";
import { FakeSsh, fakePorts, type InMemoryFileSystem, putHead, repoRoot } from "@rig/testing";
import { headDiskGb, headPackBytes } from "./box-template.ts";
import { type LiveGate, RentGpu, VM_ONSTART } from "./gpu-rental.service.ts";
import { loadVastConfig, offerQuery } from "./rental-config.ts";

const root = repoRoot;
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
/** HEAD's tree in the fixture checkout, and the stamp of a dist/rig built from it */
const TREE = "47c0853456eff6e35f31bf988a784c15f8690f19";

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
  downCostPerGb: 0.003,
  storagePerHour: 0.01,
  machineId: 41200,
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
  p.fs.put("/r/.git/HEAD", "ref: refs/heads/main\n"); // a checkout: rig's own .git at its root
  p.git.heads.set("/r", TREE);
  p.shell.on(/^\/r\/dist\/rig --built-from$/, { code: 0, stdout: `${TREE}\n`, stderr: "" });
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
  const boxPack = boxServes === "served" ? head.value.served?.file : head.value.public?.file;
  p.http.json(/8100\/props$/, {
    model_path: `/workspace/rig/local/packs/bonsai-2-27b/${boxPack}`,
    total_slots: 16,
  });
  const deps = {
    ...p,
    gate,
    self: ["/r/dist/rig"],
    vastai: "/home/u/.local/bin/vastai",
    home: "/home/u",
  };
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
  test("a download floor replaces the query's own inet_down term when it is higher, and never lowers it", async () => {
    const { p, layout } = await setup();
    const cfg = await loadVastConfig(p.fs, layout);
    if (!cfg.ok) throw new Error("fixture");
    const floored = offerQuery(cfg.value, "H100_SXM", { diskGb: 40, minDownMbps: 2000 });
    expect(floored).toContain("inet_down>=2000 direct_port_count>=2");
    expect(floored.split("inet_down").length - 1).toBe(1);
    // vast caps a search at 64 rows, so the floor has to be in the query: a filter on the rows read could miss the fast ones
    expect(offerQuery(cfg.value, "H100_SXM", { diskGb: 40, minDownMbps: 100 })).toBe(
      offerQuery(cfg.value, "H100_SXM", { diskGb: 40 }),
    );
    const bare = { ...cfg.value, rental: { ...cfg.value.rental, query: "verified=true" } };
    expect(offerQuery(bare, "H100_SXM", { diskGb: 40, minDownMbps: 2000 })).toContain(
      "verified=true inet_down>=2000",
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
      "prepare bonsai-2-27b --gpu 0",
      "fetch bonsai-2-27b",
      "build bonsai-2-27b --gpu 0 --portable",
      "derive bonsai-2-27b --json",
      "serve bonsai-2-27b --gpu auto",
    ]);
    expect(p.ssh.pulled).toEqual([
      [
        `/workspace/rig/local/engine-builds/engine-sm90-${engine.sha7}.tar.gz`,
        `/r/local/rented-box/cached-builds/engine-sm90-${engine.sha7}.tar.gz`,
      ],
    ]);
    expect(p.fs.text("/r/local/rented-box/boxes/1000/tunnel.env")).toBe(
      "HOST=ssh5.vast.ai\nPORT=12345\n",
    );
    const saved = JSON.parse(p.fs.text("/r/local/rented-box/boxes/1000/instance.json")!);
    expect(saved).toMatchObject({
      instanceId: 1000,
      cap: "90",
      head: "bonsai-2-27b",
      sshPort: 12345,
      localPort: 8100,
    });
    // armed at create, before provisioning can fail, the hard stop with the reaper; the reaper again at READY,
    // which is idempotent
    expect(p.systemd.ops).toEqual([
      "daemon-reload",
      "enable rig-vast-idle-1000.timer",
      "restart rig-vast-idle-1000.timer",
      "enable rig-vast-stop-1000.timer",
      "restart rig-vast-stop-1000.timer",
      "restart rig-vast-tunnel-1000.service",
      "enable rig-vast-idle-1000.timer",
      "restart rig-vast-idle-1000.timer",
    ]);
    // each unit replaced by a rename: a reload another process triggers never reads half a file
    const units = "/home/u/.config/systemd/user";
    expect(p.fs.replaced.filter((path) => path.startsWith(units))).toEqual(
      [
        "rig-vast-tunnel-1000.service",
        "rig-vast-idle-1000.service",
        "rig-vast-idle-1000.timer",
        "rig-vast-stop-1000.service",
        "rig-vast-stop-1000.timer",
      ].map((unit) => `${units}/${unit}`),
    );
    expect(p.fs.text(`${units}/rig-vast-tunnel-1000.service`)).toContain(
      // biome-ignore lint/suspicious/noTemplateCurlyInString: systemd expands ${PORT} from the EnvironmentFile
      "-L 127.0.0.1:8100:127.0.0.1:8099 -p ${PORT} root@${HOST}",
    );
    expect(p.fs.text(`${units}/rig-vast-tunnel-1000.service`)).toContain(
      "EnvironmentFile=/r/local/rented-box/boxes/1000/tunnel.env",
    );
    expect(p.fs.text(`${units}/rig-vast-idle-1000.service`)).toContain(
      "ExecStart=/r/dist/rig vast idle-check --box 1000",
    );
    // the hard stop: vast.toml's max_hours after the create, a persistent timer that runs rig's own down for the box,
    // then the vast CLI found here, with no rig code, whatever rig's exit (units.test.ts)
    expect(saved.stopAt - saved.createdAt).toBe(12 * 3_600_000);
    const stop = p.fs.text(`${units}/rig-vast-stop-1000.timer`)!;
    const at = new Date(saved.stopAt).toISOString().slice(0, 19).replace("T", " ");
    expect(stop).toContain(`OnCalendar=${at} UTC`);
    expect(stop).toContain("Persistent=true");
    expect(stop).toContain("WantedBy=timers.target");
    const stopService = p.fs.text(`${units}/rig-vast-stop-1000.service`)!;
    expect(stopService).toContain("ExecStart=-/r/dist/rig vast down --box 1000\n");
    expect(stopService).toContain("ExecStart=/home/u/.local/bin/vastai destroy instance 1000 -y\n");
    expect(stopService).toContain(
      "ExecStart=python3 -c \"import json,subprocess,sys;run=lambda token: subprocess.run(['/home/u/.local/bin/vastai','show','instances-v1']",
    );
    // The card is a point sample, so the box is read often enough that a run of a few minutes cannot fall between
    // two reads (2026-10-02: a box in use by short runs read under 10 % at every ten-minute check and was destroyed).
    const timer = p.fs.text(`${units}/rig-vast-idle-1000.timer`);
    expect(timer).toContain("OnUnitActiveSec=3min");
    expect(timer).toContain("every 3 minutes");
    expect(timer).not.toContain("10min");
  });
  test("--private ships the adapter, and the box must serve this machine's pack", async () => {
    const { p, head, uc } = await setup("served");
    const r = await uc.up(head, { gpu: "H100_SXM", private: true });
    expect(r.ok && r.value.kind === "up" && r.value.serving.model).toBe(head.servedFiles[0]!.file);
    expect(p.shell.calls.find((c) => c[0] === "tar")?.some((a) => a.startsWith("--exclude"))).toBe(
      false,
    );
    // and the box answering with its public pack is refused as a lesser pack than asked for
    const other = await setup("public");
    const refused = await other.uc.up(other.head, { gpu: "H100_SXM", private: true });
    expect(!refused.ok && refused.message).toContain(`not the pinned ${head.servedFiles[0]!.file}`);
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
    expect(rigCalls).toContain("build bonsai-2-27b --gpu 0");
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
          `build bonsai-2-27b --gpu 0 --from-tarball /workspace/rig/local/engine-builds/engine-sm90-${engine.sha7}.tar.gz`,
        ),
      ),
    ).toBe(true);
    expect(p.ssh.pulled).toEqual([]);
  });
  test("an offer whose profile the head cannot serve on is refused before anything is rented, dry run included", async () => {
    const { p, head: served, uc } = await setup();
    // the H100 profile in a K/V pair the pinned engine has no CUDA flash attention for
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
  test("an unmeasured card is refused with exit 3 unless --allow-arch; none beside a one-box rig's box; no funds is refused", async () => {
    const { p, head, uc } = await setup();
    p.rental.offers = [{ ...h100, gpu: "RTX 4090", computeCap: "89" }];
    let r = await uc.up(head, { gpu: "RTX_4090", dryRun: true });
    expect(!r.ok && r.code).toBe(3);
    r = await uc.up(head, { gpu: "RTX_4090", dryRun: true, allowArch: "89" });
    expect(r.ok).toBe(true);
    // a box a rig that held one rented: its units act on the only box, so it stays alone
    p.fs.put("/r/local/rented-box/instance.json", JSON.stringify({ instanceId: 7 }));
    r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("rig vast down --box 7"); // a real rental, not a dry run
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
  test("the idle timer is armed at create, not after provisioning, and an abandoned box leaves none armed", async () => {
    // a box bills from the moment vast creates it, so the reaper has to be armed there: box 53784847
    // was created on Oct 1 with rig-vast-idle.timer down, and only a later `vast status` armed it.
    const { p, head, uc } = await setup();
    p.rental.statuses = ["loading"]; // provisioning dies before anything is served
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("never reached running");
    for (const timer of ["rig-vast-idle-1000.timer", "rig-vast-stop-1000.timer"]) {
      const armed = p.systemd.ops.indexOf(`enable ${timer}`);
      expect(armed).toBeGreaterThanOrEqual(0);
      expect(armed).toBeLessThan(p.systemd.ops.indexOf(`disable ${timer}`));
      expect(await p.systemd.isActive(timer)).toBe(false);
      expect(await p.fs.exists(`/home/u/.config/systemd/user/${timer}`)).toBe(false);
    }
  });
  test("a box abandoned in provisioning keeps its reaper and hard stop while vast still lists it", async () => {
    // vastai exits 0 when vast refuses a destroy (a 429, a 5xx), so the call's answer is not the box's fate
    const { p, head, uc } = await setup();
    p.rental.statuses = ["loading"];
    p.rental.destroy = async (id) => {
      p.rental.ops.push(`destroy ${id}`);
    };
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("never reached running");
    expect(!r.ok && r.message).toContain("box 1000 is STILL listed after destroy");
    expect(p.rental.ops).toContain("destroy 1000");
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(true);
    for (const timer of ["rig-vast-idle-1000.timer", "rig-vast-stop-1000.timer"])
      expect(await p.systemd.isActive(timer)).toBe(true);
  });
  test("a box with less disk than it was sold is refused before the fetch starts, both numbers named", async () => {
    // box 53786017 was created for 160 GB and died 91 GB into a 134 GB pack: the sold disk is asked
    // for, not trusted, and the refusal comes in seconds rather than eleven minutes of download
    const { p, head, uc } = await setup();
    p.ssh.on(/df -B1/, { code: 0, stdout: "1000000000\n0\n", stderr: "" }); // 1 GB free, nothing fetched
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("has 1.0 GB free");
    expect(!r.ok && r.message).toContain("left to fetch plus");
    expect(p.ssh.calls.some((c) => c.includes("rig fetch"))).toBe(false);
  });
  test("a partly fetched pack is measured against what is left, not the whole pack", async () => {
    const { p, head, uc } = await setup();
    // everything but 1 GB of the pack is already on the box, and 8 GB is free: that is enough
    const present = headPackBytes(head) - 1e9;
    p.ssh.on(/df -B1/, { code: 0, stdout: `8000000000\n${present}\n`, stderr: "" });
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(p.ssh.calls.some((c) => c.includes("rig fetch"))).toBe(true);
    expect(r.ok || !(r.message ?? "").includes("left to fetch")).toBe(true);
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

describe("the rig shipped to a box", () => {
  // rental 3: its dist/rig was built before #168, so `rig engine ab --median` exited 64 on the box, on paid cards
  const stamped = (p: ReturnType<typeof fakePorts>, said: { code: number; stdout: string }) =>
    p.shell.on(/^\/r\/dist\/rig --built-from$/, { ...said, stderr: "" });
  /** nothing searched or rented: the refusal came before the market */
  const rentedNothing = (p: ReturnType<typeof fakePorts>) => {
    expect(p.rental.ops.filter((op) => /^(search|create) /.test(op))).toEqual([]);
    expect(p.rental.instances.size).toBe(0);
  };
  test("lab refuses a dist/rig built from another tree than HEAD's, before any rent, and names the rebuild", async () => {
    const { p, uc } = await setup();
    stamped(p, { code: 0, stdout: "0f1e2d3c4b5a69788796a5b4c3d2e1f0a9b8c7d6\n" });
    const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.code).toBe(1);
    expect(r.message).toContain("/r/dist/rig was built from tree 0f1e2d3c4b5a");
    expect(r.message).toContain(`HEAD's tree is ${TREE.slice(0, 12)}`);
    expect(r.message).toContain("bun run build");
    rentedNothing(p);
  });
  test("up refuses one built with changes HEAD does not hold", async () => {
    const { p, head, uc } = await setup();
    stamped(p, { code: 0, stdout: `${TREE}-dirty\n` });
    const r = await uc.up(head, { gpu: "H100_SXM" });
    expect(!r.ok && r.message).toContain("changes HEAD does not hold");
    expect(!r.ok && r.message).toContain("bun run build");
    rentedNothing(p);
  });
  test("a dist/rig that does not say what it was built from is refused: one compiled before the stamp, or by hand", async () => {
    const { p, uc } = await setup();
    stamped(p, { code: 64, stdout: "" }); // rig: unknown command "--built-from"
    const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
    expect(!r.ok && r.message).toContain("does not say what it was built from");
    expect(!r.ok && r.message).toContain("bun run build");
    rentedNothing(p);
  });
  test("a checkout with no HEAD has nothing to check it against, and is refused", async () => {
    const { p, uc } = await setup();
    p.git.heads.delete("/r");
    const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
    expect(!r.ok && r.message).toContain("/r is a git checkout with no HEAD");
    rentedNothing(p);
  });
  describe("an install, a release unpacked by install.sh with no checkout", () => {
    /** no .git at the root: a HEAD git finds above it (a home directory kept in git) is not rig's */
    const installed = async () => {
      const s = await setup();
      s.p.fs.files.delete("/r/.git/HEAD");
      s.p.fs.dirs.delete("/r/.git");
      s.p.git.heads.set("/r", "1111111111111111111111111111111111111111");
      return s;
    };
    test("ships the release's own binary, stamped clean with the tag's tree", async () => {
      const { p, uc } = await installed();
      stamped(p, { code: 0, stdout: "0f1e2d3c4b5a69788796a5b4c3d2e1f0a9b8c7d6\n" });
      const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
      expect(r.ok && r.value.kind).toBe("lab");
    });
    test("refuses an unstamped binary, a release from before the stamp", async () => {
      const { p, uc } = await installed();
      stamped(p, { code: 64, stdout: "" });
      const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
      expect(!r.ok && r.message).toContain("does not say what it was built from");
      expect(!r.ok && r.message).toContain("install a release");
      rentedNothing(p);
    });
    test("refuses a dirty one: no commit has its code", async () => {
      const { p, uc } = await installed();
      stamped(p, { code: 0, stdout: `${TREE}-dirty\n` });
      const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
      expect(!r.ok && r.message).toContain("changes no commit has");
      rentedNothing(p);
    });
    test("refuses what is not a stamp at all", async () => {
      const { p, uc } = await installed();
      stamped(p, { code: 0, stdout: "rig 0.1.13\n" });
      const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
      expect(!r.ok && r.message).toContain("does not say what it was built from");
      rentedNothing(p);
    });
  });
  test("a stamp of HEAD's tree is shipped, and a dry run, which ships nothing, does not ask", async () => {
    const { p, uc } = await setup();
    const r = await uc.lab({ gpu: "H100_SXM", maxDph: 3 });
    expect(r.ok && r.value.kind).toBe("lab");
    expect(p.shell.calls).toContainEqual(["/r/dist/rig", "--built-from"]);
    const dry = await setup();
    stamped(dry.p, { code: 0, stdout: "0f1e2d3c4b5a69788796a5b4c3d2e1f0a9b8c7d6\n" });
    expect((await dry.uc.lab({ gpu: "H100_SXM", maxDph: 3, dryRun: true })).ok).toBe(true);
    expect(dry.p.shell.calls).not.toContainEqual(["/r/dist/rig", "--built-from"]);
  });
});

describe("vast lab", () => {
  const rtx5090: Offer = {
    ...h100,
    id: 50263001,
    gpu: "RTX 5090",
    gpuRamMiB: 32607,
    computeCap: "120",
    dph: 0.41,
    geo: "Texas, US",
    bandwidth: 1792,
  };
  test("dry run: the cheapest matching offer for the class, no head read, nothing created", async () => {
    const { p, uc } = await setup();
    p.rental.offers = [rtx5090];
    const r = await uc.lab({ gpu: "RTX_5090", maxDph: 0.55, dryRun: true });
    expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(50263001);
    expect(p.rental.instances.size).toBe(0);
    expect(p.ssh.calls).toEqual([]);
  });
  describe("a download floor", () => {
    const slow = { ...rtx5090, id: 50263101, dph: 0.41, downMbps: 251 };
    const quick = { ...rtx5090, id: 50263102, dph: 0.6, downMbps: 5436, geo: "Spain, ES" };
    const quicker = { ...rtx5090, id: 50263103, dph: 0.7, downMbps: 8000 };
    test("skips a cheaper offer below it and rents the cheapest one above it", async () => {
      const { p, uc } = await setup();
      p.rental.offers = [slow, quick, quicker];
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, minDownMbps: 2000 });
      expect(r.ok && r.value.kind === "lab" && r.value.instanceId).toBe(1000);
      expect(p.rental.ops.filter((op) => op.startsWith("search "))).toEqual([
        expect.stringContaining("inet_down>=2000"),
      ]);
      expect(p.rental.ops.filter((op) => op.startsWith("create "))).toEqual([
        expect.stringContaining("create 50263102 "),
      ]);
    });
    test("without one the cheapest offer is rented, slow or not", async () => {
      const { p, uc } = await setup();
      p.rental.offers = [slow, quick];
      expect((await uc.lab({ gpu: "RTX_5090", maxDph: 1 })).ok).toBe(true);
      expect(p.rental.ops.filter((op) => op.startsWith("create "))).toEqual([
        expect.stringContaining("create 50263101 "),
      ]);
    });
    test("an offer exactly at it is kept", async () => {
      const { p, uc } = await setup();
      p.rental.offers = [{ ...slow, downMbps: 2000 }];
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, minDownMbps: 2000, dryRun: true });
      expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(50263101);
    });
    test("with none above it the command fails naming the fastest offer and its price, and rents nothing", async () => {
      const { p, uc } = await setup();
      p.rental.offers = [slow, { ...quick, downMbps: 1200, dph: 0.55 }];
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, minDownMbps: 2000 });
      expect(r.ok).toBe(false);
      if (r.ok) return;
      expect(r.message).toContain("2000 Mb/s");
      expect(r.message).toContain("offer 50263102");
      expect(r.message).toContain("1200 Mb/s");
      expect(r.message).toContain("$0.550/h");
      expect(r.message).toContain("nothing rented");
      // the market was asked with the floor first, and again without it to name the fastest
      const searches = p.rental.ops.filter((op) => op.startsWith("search "));
      expect(searches.length).toBe(2);
      expect(searches[0]).toContain("inet_down>=2000");
      expect(searches[1]).toContain("inet_down>=200 ");
      expect(p.rental.instances.size).toBe(0);
      expect(p.rental.ops.some((op) => op.startsWith("create"))).toBe(false);
    });
    test("a dry run under it fails the same way, and the offers it read are still kept", async () => {
      const { p, uc } = await setup();
      p.rental.offers = [slow];
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, minDownMbps: 2000, dryRun: true });
      expect(r.ok).toBe(false);
      expect(p.fs.text("/r/local/rented-box/offers.json")).toContain("50263101");
    });
    // 2026-10-03 rental 2c: `--min-down-mbps 2000` rented box 54020392, Japan, declaring 7,754 Mb/s; it pulled the pack at
    // about 53 MiB/s by its own sampler, while Texas (declaring 5,422) pulled it at 437 MiB/s. The declaration is a
    // figure the host wrote, and a host rig has measured is ranked on what it measured.
    const RATES = "/r/local/rented-box/download-rates.json";
    const measured = (rows: Array<{ machineId: number; geo: string; mbps: number }>) =>
      JSON.stringify(rows.map((row, index) => ({ instanceId: 54020392 + index, at: 0, ...row })));
    test("a host measured slower than the floor is skipped whatever it declares, and the log says what it pulled at", async () => {
      const { p, uc } = await setup();
      const japan = { ...quicker, id: 50263104, dph: 0.45, geo: "Japan, JP", machineId: 7754 };
      p.rental.offers = [japan, quick];
      p.fs.put(RATES, measured([{ machineId: 7754, geo: "Japan, JP", mbps: 447 }]));
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, minDownMbps: 2000, dryRun: true });
      expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(50263102);
      expect(p.log.lines.join("\n")).toContain(
        "offer 50263104 at 447 Mb/s measured on its host (declares 8000): under the floor",
      );
    });
    test("a host never measured takes its region's measured rate, and its own outranks the region's", async () => {
      const { p, uc } = await setup();
      const japan = { ...quicker, id: 50263105, dph: 0.45, geo: "Japan, JP", machineId: 9001 };
      p.rental.offers = [japan, quick];
      p.fs.put(RATES, measured([{ machineId: 7754, geo: "Japan, JP", mbps: 447 }]));
      const regional = await uc.lab({
        gpu: "RTX_5090",
        maxDph: 1,
        minDownMbps: 2000,
        dryRun: true,
      });
      expect(regional.ok && regional.value.kind === "dry-run" && regional.value.pick.id).toBe(
        50263102,
      );
      expect(p.log.lines.join("\n")).toContain("447 Mb/s measured in Japan, JP");
      p.fs.put(
        RATES,
        measured([
          { machineId: 7754, geo: "Japan, JP", mbps: 447 },
          { machineId: 9001, geo: "Japan, JP", mbps: 3600 },
        ]),
      );
      const own = await uc.lab({ gpu: "RTX_5090", maxDph: 1, minDownMbps: 2000, dryRun: true });
      expect(own.ok && own.value.kind === "dry-run" && own.value.pick.id).toBe(50263105);
    });
    test("with every offer measured under it the failure names the fastest by what it pulled at", async () => {
      const { p, uc } = await setup();
      p.rental.offers = [{ ...quicker, machineId: 7754 }];
      p.fs.put(RATES, measured([{ machineId: 7754, geo: "Texas, US", mbps: 447 }]));
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, minDownMbps: 2000, dryRun: true });
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.message).toContain("offer 50263103 at 447 Mb/s measured");
    });
    test("vast up takes the same floor", async () => {
      const { p, head, uc } = await setup();
      p.rental.offers = [h100, { ...h100, id: 50262230, dph: 2.4, downMbps: 4000 }];
      const r = await uc.up(head, { gpu: "H100_SXM", minDownMbps: 2000, dryRun: true });
      expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(50262230);
      const none = await uc.up(head, { gpu: "H100_SXM", minDownMbps: 9000, dryRun: true });
      expect(none.ok).toBe(false);
      if (!none.ok) expect(none.message).toContain("offer 50262230");
    });
  });
  describe("--pack: the head whose pack the box will pull, priced", () => {
    // a host bills its downloads per GB, 17 times apart (0.003 to 0.051 $/GB, 2026-09-29): the cheapest by the hour
    // is not the cheapest box once the pack is pulled onto it
    const dearPull = { ...rtx5090, id: 50263201, dph: 0.41, downCostPerGb: 0.051, downMbps: 5000 };
    const cheapPull = { ...rtx5090, id: 50263202, dph: 0.5, downCostPerGb: 0.003, downMbps: 5000 };
    test("ranks by the session all in, the pack's download at the host's price per GB included", async () => {
      const { p, head, uc } = await setup();
      p.rental.offers = [dearPull, cheapPull];
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, pack: head, hours: 2, dryRun: true });
      expect(r.ok).toBe(true);
      if (!r.ok || r.value.kind !== "dry-run") return;
      expect(r.value.pick.id).toBe(50263202);
      const gb = headPackBytes(head) / 1e9;
      const [first, second] = r.value.ranked ?? [];
      expect(first?.offer.id).toBe(50263202);
      expect(second?.offer.id).toBe(50263201);
      // the per-GB price is in the total: the gap is the pack's bytes at 0.048 $/GB less the hour's 0.09
      expect(second!.dollars - first!.dollars).toBeGreaterThan(0.048 * gb - 0.09 * 3);
      expect(p.log.lines.join("\n")).toContain("$0.051/GB down");
    });
    test("ranks on the measured rate: a host that pulled slowly is billed the hours its download takes", async () => {
      const { p, head, uc } = await setup();
      const slowHost = { ...cheapPull, id: 50263203, machineId: 7754 };
      // another region: a host never measured takes its region's rate, and Texas is measured slow here
      const fastHost = { ...cheapPull, id: 50263204, dph: 0.52, machineId: 5422, geo: "Spain, ES" };
      p.rental.offers = [slowHost, fastHost];
      p.fs.put(
        "/r/local/rented-box/download-rates.json",
        JSON.stringify([{ instanceId: 1, machineId: 7754, geo: "Texas, US", mbps: 20, at: 0 }]),
      );
      const r = await uc.lab({ gpu: "RTX_5090", maxDph: 1, pack: head, dryRun: true });
      expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(50263204);
      if (!r.ok || r.value.kind !== "dry-run") return;
      const slow = r.value.ranked?.find((each) => each.offer.id === 50263203);
      // the hours of a pull at 20 Mb/s, not at the 5,000 the host declares
      expect(slow?.minutesToServe).toBeGreaterThan((headPackBytes(head) * 8) / 20e6 / 60);
      expect(p.log.lines.join("\n")).toContain("pulls at 20 Mb/s measured on its host");
    });
    test("the box's disk holds the pack: no --disk-gb sizes it for the head, and a smaller one is refused", async () => {
      const { p, head, uc } = await setup();
      // vast.toml's disk under the head's own (bonsai-2-27b: 15.3 GB, a 30 GB disk)
      p.fs.put("/r/vast.toml", vastToml.replace(/^disk_gb = \d+/m, "disk_gb = 20"));
      p.rental.offers = [cheapPull];
      await uc.lab({ gpu: "RTX_5090", maxDph: 1, pack: head, dryRun: true });
      const gb = headDiskGb(head);
      expect(gb).toBeGreaterThan(20);
      expect(p.rental.ops.filter((op) => op.startsWith("search "))).toEqual([
        expect.stringContaining(`disk_space>=${gb}`),
      ]);
      const small = await uc.lab({ gpu: "RTX_5090", pack: head, diskGb: 10, dryRun: true });
      expect(small.ok).toBe(false);
      if (!small.ok) expect(small.message).toContain(`bonsai-2-27b's own ${gb} GB`);
    });
  });
  test("rents a card and ships rig and the pin to it, serving nothing: no head, no fetch, no derive, no tunnel", async () => {
    const { p, uc } = await setup();
    p.rental.offers = [rtx5090];
    const r = await uc.lab({ gpu: "RTX_5090", maxDph: 0.55, idleMinutes: 45 });
    expect(r.ok).toBe(true);
    if (!r.ok || r.value.kind !== "lab") return;
    expect(r.value).toMatchObject({
      instanceId: 1000,
      gpu: "RTX 5090",
      cap: "120",
      dph: 0.41,
      sshHost: "ssh5.vast.ai",
      sshPort: 12345,
    });
    // only the binary and the pin go: a rented box is someone else's machine, and nothing of a head is on it
    expect(p.shell.calls.find((c) => c[0] === "tar")).toEqual([
      "tar",
      "-C",
      "/r",
      "-czf",
      "/r/local/rented-box/payload.tar.gz",
      "dist/rig",
      "engine/engine.toml",
    ]);
    expect(p.ssh.pushed).toEqual([
      ["/r/local/rented-box/payload.tar.gz", "/workspace/rig/payload.tar.gz"],
    ]);
    const rigCalls = p.ssh.calls.filter((c) => c.includes("/workspace/rig/dist/rig "));
    expect(rigCalls).toEqual([]);
    const saved = JSON.parse(p.fs.text("/r/local/rented-box/boxes/1000/instance.json")!);
    expect(saved).toMatchObject({ instanceId: 1000, cap: "120", sshPort: 12345, idleMinutes: 45 });
    expect(saved.head).toBeUndefined();
    expect(saved.localPort).toBeUndefined();
    // the reaper and the hard stop are armed and their units written; the tunnel to a server that is not there is
    // neither
    expect(p.systemd.ops.some((op) => op.includes("tunnel"))).toBe(false);
    expect(p.fs.replaced.filter((path) => path.includes("/systemd/"))).toEqual(
      [
        "rig-vast-idle-1000.service",
        "rig-vast-idle-1000.timer",
        "rig-vast-stop-1000.service",
        "rig-vast-stop-1000.timer",
      ].map((unit) => `/home/u/.config/systemd/user/${unit}`),
    );
    for (const timer of ["rig-vast-idle-1000.timer", "rig-vast-stop-1000.timer"]) {
      expect(p.systemd.ops).toContain(`enable ${timer}`);
      expect(await p.systemd.isActive(timer)).toBe(true);
    }
    expect(p.log.lines.join("\n")).toContain("idle timer armed (45 min");
  });
  test("a lab box nobody uses is destroyed by the idle check on the card's reading alone, and a busy card keeps it", async () => {
    const { p, uc } = await setup();
    p.rental.offers = [rtx5090];
    await uc.lab({ gpu: "RTX_5090", idleMinutes: 45 });
    const listed = p.rental.instances.get(1000)!;
    p.rental.instances.set(1000, { ...listed, gpuUtil: 90 });
    for (let check = 0; check < 20; check++) {
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
      p.clock.t += 10 * 60_000;
    }
    p.rental.instances.set(1000, { ...listed, gpuUtil: 0 });
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "idle", idleMinutes: 10 } });
    p.clock.t += 45 * 60_000;
    expect(await uc.idleCheck()).toEqual({
      ok: true,
      value: { action: "destroyed", idleMinutes: 55 },
    });
    expect(p.rental.instances.size).toBe(0);
  });
  test("--vm asks the market for a VM-capable host and brings it up from the KVM image, which --image overrides", async () => {
    const { p, uc } = await setup();
    p.rental.offers = [rtx5090];
    const r = await uc.lab({ gpu: "RTX_5090", vm: true, dryRun: true });
    expect(r.ok && r.value.kind === "dry-run" && r.value.query).toContain("vms_enabled=true");
    // a container box is what rig rents by default, and its query must not carry the filter
    const plain = await uc.lab({ gpu: "RTX_5090", dryRun: true });
    expect(plain.ok && plain.value.kind === "dry-run" && plain.value.query).not.toContain(
      "vms_enabled",
    );

    const vm = await setup();
    vm.p.rental.offers = [rtx5090];
    expect((await vm.uc.lab({ gpu: "RTX_5090", vm: true })).ok).toBe(true);
    // a KVM box's sshd refuses the key vast writes (vast-cli#336: authorized_keys owned by a uid the VM has
    // no user for, StrictModes says no), so a VM is created with the on-start that repairs it
    expect(vm.p.rental.ops.at(-1)).toBe(
      `create 50263001 docker.io/vastai/kvm:ubuntu_terminal 40 onstart ${VM_ONSTART}`,
    );
    expect(VM_ONSTART).toContain("chown root:root /root/.ssh/authorized_keys");
    // and it is reached at the host's own address, where box 53930876's sshd answered while vast's proxy refused
    const recorded = async (t: typeof vm) =>
      JSON.parse(await t.p.fs.readText("/r/local/rented-box/boxes/1000/instance.json")) as {
        sshHost: string;
        sshPort: number;
      };
    expect(await recorded(vm)).toMatchObject({ sshHost: "203.0.113.7", sshPort: 40174 });
    const container = await setup();
    container.p.rental.offers = [rtx5090];
    expect((await container.uc.lab({ gpu: "RTX_5090" })).ok).toBe(true);
    expect(container.p.rental.ops.at(-1)).not.toContain("onstart");
    // a container keeps vast's proxy
    expect(await recorded(container)).toMatchObject({ sshHost: "ssh5.vast.ai", sshPort: 12345 });
    const named = await setup();
    named.p.rental.offers = [rtx5090];
    expect(
      (await named.uc.lab({ gpu: "RTX_5090", vm: true, image: "docker.io/vastai/kvm:ubuntu_22" }))
        .ok,
    ).toBe(true);
    expect(named.p.rental.ops.at(-1)).toContain("docker.io/vastai/kvm:ubuntu_22");
  });
  // A VM boots slower than a container (vast's own docs say so, with no figure), and the first one rig rented sat at
  // "Connection refused" past the five minutes a container is given while vast already listed it running. A VM gets
  // the longer wait; a container still gets the short one, so a container box that is dead is not billed for twenty.
  test("a VM that refuses ssh for longer than a container is given is waited for; a container is not", async () => {
    const refused = {
      code: 255,
      stdout: "",
      stderr: "ssh: connect to host h port 1: Connection refused",
    };
    const answersAfter = (p: Awaited<ReturnType<typeof setup>>["p"], n: number) => {
      let tries = 0;
      p.ssh.on(/^true$/, () => (++tries > n ? { code: 0, stdout: "", stderr: "" } : refused));
    };
    const vm = await setup();
    vm.p.rental.offers = [rtx5090];
    answersAfter(vm.p, 100);
    expect((await vm.uc.lab({ gpu: "RTX_5090", vm: true })).ok).toBe(true);
    expect(vm.p.rental.instances.size).toBe(1);

    const container = await setup();
    container.p.rental.offers = [rtx5090];
    answersAfter(container.p, 100);
    const r = await container.uc.lab({ gpu: "RTX_5090" });
    expect(!r.ok && r.message).toContain("ssh never answered");
    expect(!r.ok && r.message).toContain("Connection refused");
    expect(container.p.rental.instances.size).toBe(0);

    // and even a VM is not waited for for ever
    const dead = await setup();
    dead.p.rental.offers = [rtx5090];
    answersAfter(dead.p, 10_000);
    const gone = await dead.uc.lab({ gpu: "RTX_5090", vm: true });
    expect(!gone.ok && gone.message).toContain("ssh never answered");
    expect(dead.p.rental.instances.size).toBe(0);
  });
  // VM 54084522 (Oct 3, 7:40 PM CT): sshd answered rig's first probe, the VM's first boot then regenerated its host
  // keys, and the payload's scp was refused for a changed key; lab threw and left the VM billing.
  describe("a host key that changes", () => {
    const knownHosts = "/r/local/rented-box/boxes/1000/known_hosts";
    /** what OpenSSH 10 says, exit 255, to a host whose key is not the one its known_hosts holds (measured against a
     *  local sshd with rig's ssh options, Oct 3) */
    const refusal = (t: SshTarget) =>
      `Host key for [${t.host}]:${t.port} has changed and you have requested strict checking.\nHost key verification failed.`;
    /** a box's sshd as accept-new meets it through this box's known_hosts: the first answer records its key, and an
     *  answer with another key is refused. `onCommand` sees each command before it is answered; the box regenerating
     *  its keys is a test setting `key` there */
    class KeyedSsh extends FakeSsh {
      key = "key-1";
      constructor(
        private readonly fs: InMemoryFileSystem,
        private readonly onCommand: (cmd: string, ssh: KeyedSsh) => void,
      ) {
        super();
      }
      private async admits(t: SshTarget): Promise<boolean> {
        const known = (await this.fs.exists(t.knownHosts))
          ? (await this.fs.readText(t.knownHosts)).trim()
          : "";
        if (known === "") await this.fs.writeText(t.knownHosts, this.key);
        return known === "" || known === this.key;
      }
      override async run(t: SshTarget, cmd: string): Promise<SshResult> {
        this.onCommand(cmd, this);
        if (await this.admits(t)) return super.run(t, cmd);
        this.calls.push(`REFUSED ${cmd}`);
        return { code: 255, stdout: "", stderr: refusal(t), hostKeyChanged: true };
      }
      override async push(t: SshTarget, local: string, remote: string) {
        this.onCommand(`push ${remote}`, this);
        if (!(await this.admits(t))) throw new Error(`scp ${local} -> ${remote}: ${refusal(t)}`);
        return super.push(t, local, remote);
      }
    }
    /** a lab box whose sshd takes a new key at each of its first `rotations` commands that are not a probe: one is a
     *  VM's first boot, which lands after sshd first answered */
    async function boxRotating(rotations: number, at = (cmd: string) => cmd !== "true") {
      const t = await setup();
      t.p.rental.offers = [rtx5090];
      let rotated = 0;
      const ssh = new KeyedSsh(t.p.fs, (cmd, box) => {
        if (at(cmd) && rotated < rotations) box.key = `key-${++rotated + 1}`;
      });
      return { ...t, ssh, uc: new RentGpu({ ...t.deps, ssh }, t.layout, ok(engine)) };
    }

    test("a VM's key that changes once before its payload is relearned in this box's known_hosts alone, and the payload lands", async () => {
      // the key changing before the payload's first command, and between it and the copy (VM 54084522's scp)
      for (const at of [undefined, (cmd: string) => cmd.startsWith("push ")]) {
        const { p, ssh, uc } = await boxRotating(1, at);
        p.fs.put("/home/u/.ssh/known_hosts", "github.com ssh-ed25519 AAAAOTHER\n");
        const r = await uc.lab({ gpu: "RTX_5090", vm: true });
        expect(r.ok && r.value.kind).toBe("lab");
        expect(ssh.pushed).toEqual([
          ["/r/local/rented-box/payload.tar.gz", "/workspace/rig/payload.tar.gz"],
        ]);
        expect(ssh.calls.filter((c) => c.startsWith("tar -C /workspace/rig "))).toHaveLength(1);
        expect(p.fs.text(knownHosts)).toBe("key-2");
        expect(p.fs.text("/home/u/.ssh/known_hosts")).toBe("github.com ssh-ed25519 AAAAOTHER\n");
        expect(p.rental.ops).not.toContain("destroy 1000");
        expect(p.log.lines.join("\n")).toContain("box 1000's host key changed");
      }
    });
    // VM 54112003 (Oct 4, 1:23 AM CT): sshd answered a probe that did not log in, which recorded its key, the first
    // boot then regenerated the keys, and every probe after it read "Host key verification failed." until the wait ran
    // out and the VM was destroyed. Before ssh has let rig in once, nothing was trusted yet: a VM's key is learned again
    // from its next answer. A container's is not, as after the first answer.
    async function boxAnsweringBeforeLogin(vm: boolean) {
      const t = await setup();
      t.p.rental.offers = [rtx5090];
      let probes = 0;
      const ssh = new (class extends KeyedSsh {
        override async run(target: SshTarget, cmd: string): Promise<SshResult> {
          // the first probe meets sshd before the image's keys are in place: its key recorded, the login refused
          if (cmd === "true" && ++probes === 1) {
            const answered = await super.run(target, cmd);
            this.key = "key-2";
            return answered.code === 0
              ? { code: 255, stdout: "", stderr: "root@box: Permission denied (publickey)." }
              : answered;
          }
          return super.run(target, cmd);
        }
      })(t.p.fs, () => {});
      const uc = new RentGpu({ ...t.deps, ssh }, t.layout, ok(engine));
      return { ...t, ssh, uc, lab: () => uc.lab({ gpu: "RTX_5090", vm }) };
    }
    test("a VM's key that changes before ssh first let rig in is learned again from its next answer, and the payload lands", async () => {
      const { p, ssh, lab } = await boxAnsweringBeforeLogin(true);
      const r = await lab();
      expect(r.ok && r.value.kind).toBe("lab");
      expect(p.fs.text(knownHosts)).toBe("key-2");
      expect(ssh.calls).toContain("REFUSED true");
      expect(ssh.pushed).toHaveLength(1);
      expect(p.log.lines.join("\n")).toContain("before ssh let rig in");
    });
    test("a container's key that changes before ssh first let rig in is refused until the wait ends, and the box destroyed", async () => {
      const { p, ssh, lab } = await boxAnsweringBeforeLogin(false);
      const r = await lab();
      expect(!r.ok && r.message).toContain("ssh never answered");
      expect(!r.ok && r.message).toContain("Host key verification failed.");
      expect(ssh.pushed).toEqual([]);
      expect(p.rental.instances.size).toBe(0);
    });
    test("a VM's key that changes again after rig relearned it is refused, and the VM destroyed", async () => {
      const { p, ssh, uc } = await boxRotating(2);
      const r = await uc.lab({ gpu: "RTX_5090", vm: true });
      expect(!r.ok && r.message).toContain("changed again after rig relearned it");
      expect(!r.ok && r.message).toContain("destroyed 1000");
      expect(ssh.pushed).toEqual([]);
      expect(p.rental.instances.size).toBe(0);
    });
    test("a container's key is never relearned: a change before its payload is refused, and the box destroyed", async () => {
      const { p, ssh, uc } = await boxRotating(1);
      const r = await uc.lab({ gpu: "RTX_5090" });
      expect(!r.ok && r.message).toContain(
        "answered with a host key other than the one ssh recorded",
      );
      expect(!r.ok && r.message).toContain("destroyed 1000");
      expect(ssh.pushed).toEqual([]);
      expect(p.rental.instances.size).toBe(0);
    });
    test("once the payload has landed a changed key is refused: the idle check reads the box unreachable and relearns nothing", async () => {
      const { p, ssh, uc } = await boxRotating(1);
      expect((await uc.lab({ gpu: "RTX_5090", vm: true })).ok).toBe(true);
      ssh.key = "key-3";
      // vast lists no card sample for a VM, so the box's own reading is the evidence
      const { gpuUtil: _, ...listed } = p.rental.instances.get(1000)!;
      p.rental.instances.set(1000, listed);
      expect((await uc.idleCheck()).ok).toBe(true);
      expect(p.log.lines.join("\n")).toContain("box unreachable over ssh");
      expect(ssh.calls.at(-1)).toStartWith("REFUSED ");
      expect(p.fs.text(knownHosts)).toBe("key-2");
    });
    test("a failure after create destroys the box the way down --box does, and reports both", async () => {
      const copyFails = class extends FakeSsh {
        override async push(): Promise<void> {
          throw new Error("scp: Connection closed");
        }
      };
      const t = await setup();
      t.p.rental.offers = [rtx5090];
      const uc = new RentGpu({ ...t.deps, ssh: new copyFails() }, t.layout, ok(engine));
      const r = await uc.lab({ gpu: "RTX_5090", vm: true });
      expect(!r.ok && r.message).toContain("scp: Connection closed");
      expect(!r.ok && r.message).toContain("destroyed 1000");
      expect(t.p.rental.instances.size).toBe(0);
      expect(await t.p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(false);
      expect(t.p.systemd.ops).toContain("disable rig-vast-idle-1000.timer");

      // a destroy vast does not carry out is told beside the failure, and the box's cost control stays armed
      const kept = await setup();
      kept.p.rental.offers = [rtx5090];
      kept.p.rental.destroy = async (id: number) => {
        kept.p.rental.ops.push(`destroy ${id}`);
      };
      const stuck = new RentGpu({ ...kept.deps, ssh: new copyFails() }, kept.layout, ok(engine));
      const left = await stuck.lab({ gpu: "RTX_5090" });
      expect(!left.ok && left.message).toContain("scp: Connection closed");
      expect(!left.ok && left.message).toContain("box 1000 is STILL listed after destroy");
      expect(await kept.p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(true);

      // and a throw while the box is still coming up, before rig has reached it
      const early = await setup();
      early.p.rental.offers = [rtx5090];
      early.p.rental.show = async () => {
        throw new Error("vast answered 429");
      };
      const r429 = await early.uc.lab({ gpu: "RTX_5090", vm: true });
      expect(!r429.ok && r429.message).toContain("vast answered 429");
      expect(!r429.ok && r429.message).toContain("destroyed 1000");
      expect(early.p.rental.instances.size).toBe(0);
    });
  });
  test("an unmeasured card is refused unless --allow-arch, none beside a one-box rig's box, and a box that never answers is destroyed", async () => {
    const { p, uc } = await setup();
    p.rental.offers = [{ ...rtx5090, gpu: "RTX 4090", computeCap: "89" }];
    const refused = await uc.lab({ gpu: "RTX_4090", dryRun: true });
    expect(!refused.ok && refused.code).toBe(3);
    p.rental.offers = [rtx5090];
    p.fs.put("/r/local/rented-box/instance.json", JSON.stringify({ instanceId: 7 }));
    const second = await uc.lab({ gpu: "RTX_5090" });
    expect(!second.ok && second.message).toContain("a rig that held one box");
    await p.fs.remove("/r/local/rented-box/instance.json");
    p.ssh.on(/^true$/, { code: 255, stdout: "", stderr: "refused" });
    const dead = await uc.lab({ gpu: "RTX_5090" });
    expect(!dead.ok && dead.message).toContain("ssh never answered");
    // and what it last said, so a box that refuses the key reads differently from one that never booted
    expect(!dead.ok && dead.message).toContain("last ssh said: refused");
    expect(p.rental.instances.size).toBe(0);
  });
});

// Rentals 2b and 2c ran their KL legs one box after the other because rig held one (2026-10-03): each box gets its own
// state, reaper, hard stop and tunnel port, and every command that acts on a box acts on the one it is named.
describe("more than one box", () => {
  const units = "/home/u/.config/systemd/user";
  const lab5090: Offer = { ...h100, id: 50263001, gpu: "RTX 5090", computeCap: "120", dph: 0.41 };
  /** a serving box (1000, its tunnel at 8100), a lab box (1001) and a second serving box (1002, at 8101) */
  async function three() {
    const t = await setup();
    t.p.http.json(/8101\/health$/, { status: "ok" });
    t.p.http.json(/8101\/props$/, {
      model_path: `/workspace/rig/local/packs/bonsai-2-27b/${t.head.public?.file}`,
      total_slots: 16,
    });
    expect((await t.uc.up(t.head, { gpu: "H100_SXM" })).ok).toBe(true);
    t.p.rental.offers = [lab5090];
    expect((await t.uc.lab({ gpu: "RTX_5090", maxHours: 3 })).ok).toBe(true);
    t.p.rental.offers = [h100];
    const second = await t.uc.up(t.head, { gpu: "H100_SXM" });
    expect(second.ok && second.value.kind === "up" && second.value.localUrl).toBe(
      "http://127.0.0.1:8101",
    );
    return t;
  }

  test("each box is held with its own state, reaper, hard stop and, serving, its own tunnel port", async () => {
    const { p } = await three();
    expect(p.rental.instances.size).toBe(3);
    const saved = (id: number) =>
      JSON.parse(p.fs.text(`/r/local/rented-box/boxes/${id}/instance.json`)!);
    expect(saved(1000)).toMatchObject({ localPort: 8100, head: "bonsai-2-27b" });
    expect(saved(1001).localPort).toBeUndefined();
    expect(saved(1002)).toMatchObject({ localPort: 8101, head: "bonsai-2-27b" });
    expect(p.fs.text(`${units}/rig-vast-tunnel-1002.service`)).toContain(
      "-L 127.0.0.1:8101:127.0.0.1:8099",
    );
    expect(await p.fs.exists(`${units}/rig-vast-tunnel-1001.service`)).toBe(false);
    for (const id of [1000, 1001, 1002])
      for (const timer of [`rig-vast-idle-${id}.timer`, `rig-vast-stop-${id}.timer`])
        expect(await p.systemd.isActive(timer)).toBe(true);
    // --max-hours is the lab box's own hard stop, over vast.toml's 12
    expect(saved(1001).stopAt - saved(1001).createdAt).toBe(3 * 3_600_000);
    expect(saved(1002).stopAt - saved(1002).createdAt).toBe(12 * 3_600_000);
  });
  test("down acts on the box it names and retires only that box's units; none named with several is refused", async () => {
    const { p, uc } = await three();
    const unnamed = await uc.down();
    expect(!unnamed.ok && unnamed.code).toBe(64);
    expect(!unnamed.ok && unnamed.message).toContain(
      "3 boxes held: name one with --box (1000, 1001, 1002)",
    );
    expect(p.rental.instances.size).toBe(3);
    const before = p.systemd.ops.length;
    const r = await uc.down({ box: 1001 });
    expect(r).toMatchObject({ ok: true, value: { destroyed: [1001] } });
    expect([...p.rental.instances.keys()]).toEqual([1000, 1002]);
    expect(await p.fs.exists("/r/local/rented-box/boxes/1001")).toBe(false);
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(true);
    const ops = p.systemd.ops.slice(before);
    expect(ops.some((op) => op.includes("1000") || op.includes("1002"))).toBe(false);
    // the box goes down from inside its own idle check or hard stop: their services are never stopped
    expect(
      ops.some((op) => op.startsWith("stop") && op.endsWith(".service") && !op.includes("tunnel")),
    ).toBe(false);
    expect(await p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(true);
    expect((await p.fs.list(units)).filter((unit) => unit.includes("1001"))).toEqual([]);
    const missing = await uc.down({ box: 7 });
    expect(!missing.ok && missing.message).toContain("no box 7 held here");
    const all = await uc.down({ all: true });
    expect(all.ok && all.value.destroyed).toEqual([1000, 1002]);
    expect(await p.fs.list(units)).toEqual([]);
  });
  test("the idle check reads the box it names: a lab box never reads another box's server through its port", async () => {
    const { p, uc } = await three();
    const unnamed = await uc.idleCheck();
    expect(!unnamed.ok && unnamed.code).toBe(64);
    p.http.on(/8100\/metrics$/, () => ({
      status: 200,
      text: "llamacpp:prompt_tokens_total 9\nllamacpp:tokens_predicted_total 9\nllamacpp:requests_processing 1\n",
    }));
    const lab = p.rental.instances.get(1001)!;
    p.rental.instances.set(1001, { ...lab, gpuUtil: 0 });
    const before = p.http.requests.length;
    expect(await uc.idleCheck({ box: 1001 })).toEqual({ ok: true, value: { action: "changed" } });
    expect(p.http.requests.slice(before)).toEqual([]);
    expect(p.log.lines.at(-1)).toContain("idle-check box 1001: server unreachable");
    // the serving box reads its own server, busy
    expect(await uc.idleCheck({ box: 1000 })).toEqual({ ok: true, value: { action: "active" } });
    // the lab box idles out alone, and the others stay
    p.clock.t += 45 * 60_000;
    expect(await uc.idleCheck({ box: 1001 })).toMatchObject({
      ok: true,
      value: { action: "destroyed" },
    });
    expect([...p.rental.instances.keys()]).toEqual([1000, 1002]);
    expect(await p.systemd.isActive("rig-vast-idle-1002.timer")).toBe(true);
  });
  test("status lists every box, the oldest first, or the one named", async () => {
    const { p, uc } = await three();
    const all = await uc.status();
    expect(all.ok && all.value.boxes.map((each) => each.box.instanceId)).toEqual([
      1000, 1001, 1002,
    ]);
    expect(all.ok && all.value.boxes.map((each) => each.healthy)).toEqual([true, false, true]);
    const one = await uc.status({ box: 1002 });
    expect(one.ok && one.value.boxes.map((each) => each.box.localPort)).toEqual([8101]);
    // a port freed by a box going down is taken by the next serving box
    await uc.down({ box: 1000 });
    p.rental.offers = [h100];
    const next = await uc.up((await setup()).head, { gpu: "H100_SXM" });
    expect(next.ok && next.value.kind === "up" && next.value.localUrl).toBe(
      "http://127.0.0.1:8100",
    );
  });
  test("two ups at once take two ports: the port is claimed before the create, not read from what is saved", async () => {
    const { p, head, uc } = await setup();
    p.http.json(/8101\/health$/, { status: "ok" });
    p.http.json(/8101\/props$/, {
      model_path: `/workspace/rig/local/packs/bonsai-2-27b/${head.public?.file}`,
      total_slots: 16,
    });
    const both = await Promise.all([
      uc.up(head, { gpu: "H100_SXM" }),
      uc.up(head, { gpu: "H100_SXM" }),
    ]);
    expect(both.every((r) => r.ok)).toBe(true);
    const saved = (id: number) =>
      JSON.parse(p.fs.text(`/r/local/rented-box/boxes/${id}/instance.json`)!).localPort;
    expect([saved(1000), saved(1001)].sort()).toEqual([8100, 8101]);
  });
  test("a box record that does not parse is named and skipped; every other box is still read and reaped", async () => {
    const { p, uc } = await three();
    // a record cut short (a full disk, a hand edit): records are written by rename, so a reader never sees one mid-write
    expect(p.fs.replaced).toContain("/r/local/rented-box/boxes/1001/instance.json");
    p.fs.put("/r/local/rented-box/boxes/1001/instance.json", '{"instanceId": 10');
    expect((await uc.idleCheck({ box: 1000 })).ok).toBe(true);
    const status = await uc.status();
    expect(status.ok && status.value.boxes.map((each) => each.box.instanceId)).toEqual([
      1000, 1002,
    ]);
    expect(p.log.lines.join("\n")).toContain(
      "/r/local/rented-box/boxes/1001/instance.json does not parse",
    );
  });
  test("down --all tries every box though vast refuses one, and --all with --box is refused", async () => {
    const { p, uc } = await three();
    const both = await uc.down({ all: true, box: 1001 });
    expect(!both.ok && both.code).toBe(64);
    expect(p.rental.instances.size).toBe(3);
    const destroy = p.rental.destroy.bind(p.rental);
    p.rental.destroy = async (id) => {
      if (id === 1000) throw new Error("vastai destroy instance 1000: 502 Bad Gateway");
      await destroy(id);
    };
    const all = await uc.down({ all: true });
    expect(!all.ok && all.message).toContain("box 1000 is STILL listed after destroy");
    expect([...p.rental.instances.keys()]).toEqual([1000]);
    expect(await p.systemd.isActive("rig-vast-stop-1000.timer")).toBe(true);
    expect(await p.fs.exists("/r/local/rented-box/boxes/1002")).toBe(false);
  });
  test("a hard stop outside a quarter hour to a week is refused before anything is rented", async () => {
    const { p, uc } = await setup();
    p.rental.offers = [lab5090];
    for (const maxHours of [0.1, 169, 99_999_999_999]) {
      const r = await uc.lab({ gpu: "RTX_5090", maxHours });
      expect(!r.ok && r.code).toBe(64);
      expect(!r.ok && r.message).toContain("hard stop takes 0.25 to 168 hours");
    }
    expect(p.rental.ops.some((op) => op.startsWith("create"))).toBe(false);
    expect((await uc.lab({ gpu: "RTX_5090", maxHours: 168 })).ok).toBe(true);
  });
});

// A rig that held one box kept it at local/rented-box/instance.json with units named for none: a box it rented before an
// upgrade is checked, reaped and downed where it is, so the upgrade never leaves it billing without its reaper.
describe("a box rented by a rig that held one", () => {
  const legacy = {
    instanceId: 1000,
    offerId: 1,
    gpu: "RTX 5090",
    cap: "120",
    dph: 0.41,
    geo: "Texas, US",
    createdAt: 0,
    sshHost: "ssh5.vast.ai",
    sshPort: 12345,
  };
  async function held() {
    const t = await setup();
    t.p.rental.instances.set(1000, {
      id: 1000,
      status: "running",
      label: "rig",
      dph: 0.41,
      gpuUtil: 0,
    });
    t.p.fs.put("/r/local/rented-box/instance.json", JSON.stringify(legacy));
    for (const unit of ["rig-vast-idle.service", "rig-vast-idle.timer"])
      t.p.fs.put(`/home/u/.config/systemd/user/${unit}`, "[Unit]\n");
    await t.p.systemd.enable("rig-vast-idle.timer");
    await t.p.systemd.restart("rig-vast-idle.timer");
    return t;
  }
  test("its old timer's idle check, which names no box, reaps it where it is", async () => {
    const { p, uc } = await held();
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "changed" } });
    expect(p.fs.text("/r/local/rented-box/idle.json")).toContain('"key":"unreachable"');
    p.clock.t += 45 * 60_000;
    expect(await uc.idleCheck()).toMatchObject({ ok: true, value: { action: "destroyed" } });
    expect(p.rental.instances.size).toBe(0);
    expect(await p.fs.exists("/r/local/rented-box/instance.json")).toBe(false);
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(false);
    expect(await p.fs.list("/home/u/.config/systemd/user")).toEqual([]);
  });
  test("status reads it with no hard stop and re-arms its old timer; no box is rented beside it", async () => {
    const { p, head, uc } = await held();
    p.systemd.active.delete("rig-vast-idle.timer");
    const r = await uc.status();
    expect(r).toMatchObject({
      ok: true,
      value: {
        boxes: [{ box: { instanceId: 1000, legacy: true }, idleTimer: "re-armed", hardStop: null }],
      },
    });
    expect(await p.systemd.isActive("rig-vast-idle.timer")).toBe(true);
    const beside = await uc.up(head, { gpu: "H100_SXM" });
    expect(!beside.ok && beside.message).toContain("run: rig vast down --box 1000");
    expect(p.rental.instances.size).toBe(1);
    expect((await uc.down({ box: 1000 })).ok).toBe(true);
    expect((await uc.up(head, { gpu: "H100_SXM" })).ok).toBe(true);
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
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000")).toBe(false);
    // its units retired: the timers disabled and stopped, the tunnel stopped, their files gone. Never a service: a
    // box goes down from inside its own idle check or hard stop, which stopping would kill.
    expect(p.systemd.ops.slice(-6)).toEqual([
      "disable rig-vast-idle-1000.timer",
      "disable rig-vast-stop-1000.timer",
      "stop rig-vast-idle-1000.timer",
      "stop rig-vast-stop-1000.timer",
      "stop rig-vast-tunnel-1000.service",
      "daemon-reload",
    ]);
    expect(await p.fs.list("/home/u/.config/systemd/user")).toEqual([]);
  });
  test("down refuses to believe a destroy the listing contradicts, and leaves the box's cost control running", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    p.rental.destroy = async () => {}; // vast says nothing, the box stays listed
    const before = p.systemd.ops.length;
    const r = await uc.down();
    expect(!r.ok && r.message).toContain("STILL listed");
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(true);
    expect(p.systemd.ops.slice(before)).toEqual([]);
    expect(await p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(true);
    expect(await p.systemd.isActive("rig-vast-stop-1000.timer")).toBe(true);
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
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(true);
    expect(p.systemd.ops.slice(before)).toEqual([]);
    expect(await p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(true);
    // destroyed from vast's console meanwhile: the next down's call refuses the gone id
    p.rental.instances.delete(1000);
    const gone = await uc.down();
    expect(gone.ok && gone.value.destroyed).toEqual([1000]);
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(false);
    expect(await p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(false);
    expect(await p.systemd.isActive("rig-vast-stop-1000.timer")).toBe(false);
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
    expect(await stale.status()).toMatchObject({ ok: true, value: { boxes: [{ listed: true }] } });
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
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(false);
    expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "no-box" } });
    // the box's own unit fired after the box went is no box, never a failure
    expect(await uc.idleCheck({ box: 1000 })).toEqual({ ok: true, value: { action: "no-box" } });
  });
  test("--disk-gb sizes the query and the box; --idle-minutes is this box's own budget, read by every idle check", async () => {
    const { p, head, uc } = await setup();
    const r = await uc.up(head, { gpu: "H100_SXM", diskGb: 300, idleMinutes: 720 });
    expect(r.ok).toBe(true);
    expect(p.rental.ops.find((op) => op.startsWith("search "))).toContain("disk_space>=300");
    expect(p.rental.ops.at(-1)).toBe("create 50262229 nvidia/cuda:13.0.3-devel-ubuntu24.04 300");
    expect(JSON.parse(p.fs.text("/r/local/rented-box/boxes/1000/instance.json")!)).toMatchObject({
      idleMinutes: 720,
    });
    expect(p.fs.text("/home/u/.config/systemd/user/rig-vast-idle-1000.service")).toContain(
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
  // vast lists no gpu_util for a VM (boxes 53930876 and 53933738), so the reaper, which refuses to count a card it did
  // not read, never destroyed one: nothing stopped a forgotten VM billing. The card is then read on the box itself, over
  // ssh, at the address rig recorded for it, and it decides the way vast's own reading does.
  describe("idle-check on a box vast lists no GPU reading for (a VM)", () => {
    const smi = /nvidia-smi --query-gpu=utilization\.gpu/;
    const reads = (util: string, rc = 0) => ({
      code: 0,
      stdout: `${util}\nrc=${rc}\n`,
      stderr: "",
    });
    /** what the check prints on a box whose sampler has been writing: the card now, its peak over the window, and the
     *  average KB/s the box received over it (absent from a box whose sampler predates the download column) */
    const sampled = (now: string, peak: number, download?: number, pull?: [number, number]) => ({
      code: 0,
      stdout: `${now}\nrc=0\nwindow=${peak}\n${download === undefined ? "" : `download=${download}\n`}${pull ? `pull=${pull[0]} ${pull[1]}\n` : ""}`,
      stderr: "",
    });
    const unreachable = {
      code: 255,
      stdout: "",
      stderr: "ssh: connect to host h port 1: Connection timed out",
    };
    const rtx: Offer = { ...h100, id: 50263001, gpu: "RTX 5090", computeCap: "120", dph: 0.41 };
    async function vm() {
      const t = await setup();
      t.p.rental.offers = [rtx];
      expect((await t.uc.lab({ gpu: "RTX_5090", vm: true })).ok).toBe(true);
      // the box answers ssh for rig's own steps; the market lists the VM running with no sample
      const { gpuUtil: _, ...unsampled } = t.p.rental.instances.get(1000)!;
      t.p.rental.instances.set(1000, unsampled);
      return t;
    }
    test("the card read over ssh decides: busy resets the clock, idle for the budget destroys", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, reads("57"));
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
      p.ssh.on(smi, reads("3"));
      p.clock.t += 20 * 60_000;
      expect(await uc.idleCheck()).toEqual({
        ok: true,
        value: { action: "idle", idleMinutes: 20 },
      });
      p.ssh.on(smi, reads("41\n0")); // the busiest of several cards
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
      p.ssh.on(smi, reads("0"));
      p.clock.t += 44 * 60_000;
      expect(await uc.idleCheck()).toEqual({
        ok: true,
        value: { action: "idle", idleMinutes: 44 },
      });
      p.clock.t += 60_000;
      expect(await uc.idleCheck()).toEqual({
        ok: true,
        value: { action: "destroyed", idleMinutes: 45 },
      });
      expect(p.rental.instances.size).toBe(0);
      expect(p.log.lines.join("\n")).toContain("GPU 0 % read on the box");
    });
    // 2026-10-02: a lab VM held for hours was destroyed with "idle for 51 min (GPU 0 % read on the box)" and the six
    // checks before it left no trace, so nothing could say whether the card had been busy between samples or the
    // reads had failed. Every check now says what it saw.
    test("every check says what it saw and where it read it, so a destroy can be traced to the readings behind it", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, reads("57"));
      await uc.idleCheck();
      p.ssh.on(smi, reads("3"));
      p.clock.t += 20 * 60_000;
      await uc.idleCheck();
      p.ssh.on(smi, unreachable);
      p.clock.t += 5 * 60_000;
      await uc.idleCheck();
      const seen = p.log.lines.filter((line) => line.includes("idle-check box 1000"));
      expect(seen).toHaveLength(3);
      expect(seen[0]).toMatch(/GPU 57 % read on the box: active$/);
      expect(seen[1]).toMatch(/GPU 3 % read on the box: idle 20 of 45 min$/);
      expect(seen[2]).toMatch(/box unreachable over ssh: idle 25 of 45 min$/);
    });
    // The orchestrator's acceptance for the reaper (2026-10-02, after VM 4): a trace in which every check reads under
    // 10 % at the instant while work runs between the checks must not destroy the box, and a box with nothing running
    // for the whole budget must still be destroyed. A card read at one instant cannot tell the first from the second.
    test("work that ran between two reads is seen: six checks that each read under 10 % at the instant, with the sampler's peak in the window, keep the box", async () => {
      const { p, uc } = await vm();
      // 10 minutes apart, 60 minutes: VM 4's shape, against a 45-minute budget
      const trace: Array<[string, number]> = [
        ["1", 0],
        ["2", 64],
        ["0", 0],
        ["3", 41],
        ["0", 0],
        ["1", 88],
      ];
      for (const [now, peak] of trace) {
        p.ssh.on(smi, sampled(now, peak));
        const r = await uc.idleCheck();
        expect(r.ok && r.value.action).not.toBe("destroyed");
        p.clock.t += 10 * 60_000;
      }
      expect(p.rental.instances.size).toBe(1);
      // the same six reads without a sampler's window are VM 4's trace, and it is destroyed: the replay can fail
      const bare = await vm();
      let destroyed = false;
      for (const [now] of trace) {
        bare.p.ssh.on(smi, reads(now));
        const r = await bare.uc.idleCheck();
        destroyed ||= r.ok && r.value.action === "destroyed";
        bare.p.clock.t += 10 * 60_000;
      }
      expect(destroyed).toBe(true);
      expect(bare.p.rental.instances.size).toBe(0);
      // and what each check saw is on the log, the window's peak beside the instant
      expect(p.log.lines.join("\n")).toMatch(
        /GPU 3 % now, peak 41 % in the last 6 min, read on the box: active/,
      );
    });
    test("a box with nothing running for the whole budget is destroyed, the sampler's quiet window included", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, sampled("0", 0));
      expect((await uc.idleCheck()).ok).toBe(true); // the clock starts here
      for (let minute = 3; minute < 45; minute += 3) {
        p.clock.t += 3 * 60_000;
        const r = await uc.idleCheck();
        expect(r.ok && r.value.action).toBe("idle");
      }
      p.clock.t += 3 * 60_000;
      expect(await uc.idleCheck()).toMatchObject({ ok: true, value: { action: "destroyed" } });
      expect(p.rental.instances.size).toBe(0);
      expect(p.log.lines.join("\n")).toContain("idle-check box 1000");
    });
    // 2026-10-02: the 2a box was rented with a 60-minute budget for a 134 GB pack pulled at 251 Mb/s, about 70 minutes
    // with the card at 0 % and no server: the reaper would have destroyed it mid-pull. A box receiving data is working.
    test("a box receiving at a steady rate is kept past the whole budget, its card and server idle throughout", async () => {
      const { p, uc } = await vm();
      for (let minute = 0; minute <= 3 * 45; minute += 3) {
        p.ssh.on(smi, sampled("0", 0, 31_000));
        const r = await uc.idleCheck();
        expect(r.ok && r.value.action).toBe("active");
        p.clock.t += 3 * 60_000;
      }
      expect(p.rental.instances.size).toBe(1);
      expect(p.log.lines.filter((line) => line.includes("idle-check box 1000")).at(-1)).toMatch(
        /GPU 0 % read on the box, download 30\.3 MB\/s: active$/,
      );
    });
    test("a box with no traffic and no GPU work is destroyed after its budget, and the check says it saw none", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, sampled("0", 0, 0));
      expect((await uc.idleCheck()).ok).toBe(true);
      p.clock.t += 44 * 60_000;
      expect(await uc.idleCheck()).toMatchObject({ ok: true, value: { action: "idle" } });
      p.clock.t += 60_000;
      expect(await uc.idleCheck()).toMatchObject({ ok: true, value: { action: "destroyed" } });
      expect(p.rental.instances.size).toBe(0);
      expect(p.log.lines.join("\n")).toMatch(
        /GPU 0 % read on the box, download 0 KB\/s: idle 44 of 45 min/,
      );
    });
    test("the budget runs from where the traffic stopped, and a trickle under the floor is not traffic", async () => {
      const { p, uc } = await vm();
      for (let check = 0; check < 20; check++) {
        p.ssh.on(smi, sampled("0", 0, 25_000));
        expect((await uc.idleCheck()).ok).toBe(true);
        p.clock.t += 3 * 60_000;
      }
      // the pull ended at the last of those checks: 200 KB/s is an ssh session and a log tail, not a download
      p.ssh.on(smi, sampled("0", 0, 200));
      for (let minute = 3; minute < 45; minute += 3) {
        expect(await uc.idleCheck()).toEqual({
          ok: true,
          value: { action: "idle", idleMinutes: minute },
        });
        p.clock.t += 3 * 60_000;
      }
      expect(await uc.idleCheck()).toMatchObject({ ok: true, value: { action: "destroyed" } });
    });
    // what the next rental ranks on. 2026-10-03 the sampler read 51-64 MB/s through box 54020392's pull (Japan) and 437.5
    // through 53980196's (Texas), each within a fifth of the rate measured by hand from its fetch.
    test("a box's pull is recorded by host and region, its best window kept and kept past the box; rig's payload push is never one", async () => {
      const { p, uc } = await vm();
      const RATES = "/r/local/rented-box/download-rates.json";
      p.fs.put(
        RATES,
        JSON.stringify([{ instanceId: 7, machineId: 1, geo: "Japan, JP", mbps: 447, at: 0 }]),
      );
      // rig's payload pushed from here: 20 seconds at 30 MB/s, 600 MB
      p.ssh.on(smi, sampled("0", 0, 1_700, [30_000, 4]));
      await uc.idleCheck();
      // the engine's runtime from NVIDIA's CDN: 2.1 GiB in 15 seconds, too short a pull to measure a host by
      p.clock.t += 3 * 60_000;
      p.ssh.on(smi, sampled("0", 0, 6_250, [150_000, 3]));
      await uc.idleCheck();
      expect(JSON.parse(p.fs.text(RATES)!)).toHaveLength(1);
      // the pack: 437.5 MiB/s for the three minutes of the window it ran in, half the window's average
      p.ssh.on(smi, sampled("0", 0, 224_000, [448_000, 36]));
      p.clock.t += 3 * 60_000;
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
      p.ssh.on(smi, sampled("0", 0, 100_000, [200_000, 30])); // the window the pull ended in
      p.clock.t += 3 * 60_000;
      await uc.idleCheck();
      const rows = JSON.parse(p.fs.text(RATES)!);
      expect(rows).toEqual([
        { instanceId: 7, machineId: 1, geo: "Japan, JP", mbps: 447, at: 0 },
        {
          instanceId: 1000,
          machineId: 41200,
          geo: "Germany, DE",
          mbps: 3670,
          at: expect.any(Number),
        },
      ]);
      expect(p.log.lines.join("\n")).toContain("measured 3670 Mb/s");
      expect((await uc.down()).ok).toBe(true);
      expect(JSON.parse(p.fs.text(RATES)!)).toHaveLength(2);
    });
    test("a box whose sampler has no download column yet is read as before: the card decides alone", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, sampled("0", 0));
      expect((await uc.idleCheck()).ok).toBe(true);
      expect(p.log.lines.join("\n")).not.toContain("download");
    });
    test("the check's command reads the box's received bytes and retires the sampler that predates the column", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, sampled("0", 0, 0));
      await uc.idleCheck();
      const command = p.ssh.calls.filter((call) => smi.test(call)).at(-1) ?? "";
      expect(command).toContain("/proc/net/dev");
      expect(command).toContain("rig-card-sampler.v2.pid");
      expect(command).toContain("kill"); // the first-generation sampler, found by its own pid file
    });
    test("the check keeps the box's sampler running and asks for the last two checks' worth of it", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, sampled("0", 0));
      await uc.idleCheck();
      const command = p.ssh.calls.filter((call) => smi.test(call)).at(-1) ?? "";
      expect(command).toContain("rig-card-sampler"); // started when it is not running, so a rebooted box has one again
      expect(command).toContain("kill -0");
      expect(command).toContain("360"); // the window, in seconds: two checks of 3 minutes
      expect(command).toContain("sleep 5");
    });
    test("a box that does not answer ssh counts as idle, so a VM that lost its network does not bill for ever; one that answers busy resets it", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, unreachable);
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "changed" } });
      p.clock.t += 30 * 60_000;
      expect(await uc.idleCheck()).toEqual({
        ok: true,
        value: { action: "idle", idleMinutes: 30 },
      });
      p.ssh.on(smi, reads("88"));
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
      p.ssh.on(smi, unreachable);
      p.clock.t += 44 * 60_000;
      expect(await uc.idleCheck()).toEqual({
        ok: true,
        value: { action: "idle", idleMinutes: 44 },
      });
      p.clock.t += 60_000;
      expect(await uc.idleCheck()).toMatchObject({ ok: true, value: { action: "destroyed" } });
      expect(p.rental.instances.size).toBe(0);
      expect(p.log.lines.join("\n")).toContain("unreachable over ssh");
    });
    test("a box that answers but whose nvidia-smi gives no number is still a card nobody read: nothing is counted", async () => {
      const { p, uc } = await vm();
      p.ssh.on(smi, { code: 0, stdout: "rc=9\n", stderr: "" });
      for (let check = 0; check < 80; check++) {
        const r = await uc.idleCheck();
        expect(!r.ok && r.message).toBe(
          "the server did not answer and vast listed no GPU reading: box 1000 not counted, its idle clock unchanged",
        );
        p.clock.t += 10 * 60_000;
      }
      expect(p.rental.instances.size).toBe(1);
    });
    // vast's sample for a container is one instant too, so a box it does sample is read on the box as well, and the
    // higher of the two decides; an ssh that fails leaves vast's reading standing rather than reading as an idle card
    test("a box vast does sample is read on the box too: the higher reading decides, and an ssh that fails leaves vast's", async () => {
      const { p, head, uc } = await setup();
      await uc.up(head, { gpu: "H100_SXM" });
      const instance = p.rental.instances.get(1000)!;
      p.rental.instances.set(1000, { ...instance, gpuUtil: 2 });
      p.ssh.on(smi, { code: 0, stdout: "1\nrc=0\nwindow=77\n", stderr: "" });
      expect(await uc.idleCheck()).toEqual({ ok: true, value: { action: "active" } });
      expect(p.ssh.calls.filter((c) => smi.test(c))).toHaveLength(1);
      p.ssh.on(smi, { code: 255, stdout: "", stderr: "ssh: connect to host h port 1: timed out" });
      p.log.lines.length = 0;
      await uc.idleCheck();
      const said = p.log.lines.filter((line) => line.includes("idle-check box 1000")).at(-1) ?? "";
      expect(said).toContain("GPU 2 %");
      expect(said).not.toContain("unreachable over ssh");
    });
  });
  test("status reads the state, the market, the tunnel and the hard stop", async () => {
    const { p, head, uc } = await setup();
    expect(await uc.status()).toEqual({ ok: true, value: { boxes: [] } });
    await uc.up(head, { gpu: "H100_SXM" });
    p.clock.t += 3_600_000;
    const r = await uc.status();
    expect(r).toMatchObject({
      ok: true,
      value: {
        boxes: [{ listed: true, hours: 1, cost: 1.98, healthy: true, tunnelActive: true }],
      },
    });
    const box = r.ok ? r.value.boxes[0]! : undefined;
    expect(box?.hardStop).toEqual({ at: box!.box.createdAt + 12 * 3_600_000, timer: "active" });
  });
  test("status re-arms the idle timer of a box that bills without it, and says so; a box gone from the listing is left alone", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ idleTimer: "active" }] },
    });
    p.systemd.active.delete("rig-vast-idle-1000.timer"); // 2026-09-24: dead at 09:08, no stop logged
    const before = p.systemd.ops.length;
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ idleTimer: "re-armed" }] },
    });
    expect(p.systemd.ops.slice(before)).toEqual([
      "enable rig-vast-idle-1000.timer",
      "restart rig-vast-idle-1000.timer",
    ]);
    expect(await p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(true);
    expect(p.log.lines.join("\n")).toContain("rig-vast-idle-1000.timer was not running: re-armed");
    // the hard stop's timer the same
    p.systemd.active.delete("rig-vast-stop-1000.timer");
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ hardStop: { timer: "re-armed" } }] },
    });
    expect(await p.systemd.isActive("rig-vast-stop-1000.timer")).toBe(true);
    // destroyed elsewhere: nothing bills, so nothing is armed
    p.systemd.active.delete("rig-vast-idle-1000.timer");
    p.systemd.active.delete("rig-vast-stop-1000.timer");
    p.rental.instances.delete(1000);
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ listed: false, idleTimer: "inactive", hardStop: { timer: "inactive" } }] },
    });
    expect(await p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(false);
  });
  test("status: a market that cannot be read is no evidence the box is gone, so a dead timer is re-armed", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    p.systemd.active.delete("rig-vast-idle-1000.timer");
    p.rental.show = async () => {
      throw new Error("vastai show instance 1000: 401 key expired");
    };
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ listed: "unread", idleTimer: "re-armed" }] },
    });
    expect(await p.systemd.isActive("rig-vast-idle-1000.timer")).toBe(true);
    expect(p.log.lines.join("\n")).toContain(
      "box 1000 may be billing and rig-vast-idle-1000.timer was not running: re-armed",
    );
  });
  test("status reports an active timer whose last check failed: a check that exits 1 controls nothing", async () => {
    const { p, head, uc } = await setup();
    await uc.up(head, { gpu: "H100_SXM" });
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ idleTimer: "active", idleCheck: "ok" }] },
    });
    p.systemd.results.set("rig-vast-idle-1000.service", "exit-code");
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ idleTimer: "active", idleCheck: "failed" }] },
    });
    expect(p.log.lines.join("\n")).toContain(
      "rig-vast-idle-1000.service's last run ended exit-code: box 1000's cost control is not running",
    );
    p.systemd.results.set("rig-vast-idle-1000.service", null);
    expect(await uc.status()).toMatchObject({
      ok: true,
      value: { boxes: [{ idleCheck: "unread" }] },
    });
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
