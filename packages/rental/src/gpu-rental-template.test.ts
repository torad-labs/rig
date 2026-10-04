import { describe, expect, test } from "bun:test";
import { basename } from "node:path";
import type { Offer } from "@rig/core";
import { layoutAt, ok } from "@rig/core";
import { loadEngine } from "@rig/engine";
import { loadHead } from "@rig/head";
import { fakePorts, repoRoot } from "@rig/testing";
import { type LiveGate, RentGpu } from "./gpu-rental.service.ts";
import { estimateOffer, rankOffers, templateQuery } from "./template-offers.ts";

const IMAGE = "registry.torad.ai/rig:glm-5.3-flash-sm120-abc1234-0e11d35a";
const RECORD = "/r/local/images/glm-5.3-flash-sm120";
const GB = 1e9;

/** two 2 x PRO 6000 hosts as the market listed them on 2026-09-29: the cheaper by the hour bills 17 times more a GB */
const pro6000 = (o: Partial<Offer>): Offer => ({
  id: 1,
  gpu: "RTX PRO 6000 WS",
  gpus: 2,
  gpuRamMiB: 97887,
  computeCap: "120",
  dph: 3.2,
  geo: "Norway, NO",
  cpu: "Xeon 6767P",
  ramGiB: 251,
  bandwidth: 1600,
  cudaMaxGood: 13.3,
  reliability: 0.99,
  downMbps: 900,
  downCostPerGb: 0.003,
  storagePerHour: 0.02,
  machineId: 1,
  ...o,
});
const dearDownload = pro6000({
  id: 53168569,
  dph: 3.2,
  downCostPerGb: 0.051,
  downMbps: 900,
  machineId: 53168,
  geo: "Sweden, SE",
});
const cheapDownload = pro6000({
  id: 33665866,
  dph: 3.47,
  downCostPerGb: 0.003,
  downMbps: 6724,
  machineId: 33665,
});

async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  for (const file of ["head.toml", "assets/chat-template.jinja"])
    p.fs.put(
      `/r/heads/glm-5.3-flash/${file}`,
      await Bun.file(`${repoRoot}/heads/glm-5.3-flash/${file}`).text(),
    );
  p.fs.put("/r/engine/engine.toml", await Bun.file(`${repoRoot}/engine/engine.toml`).text());
  p.fs.put("/r/vast.toml", await Bun.file(`${repoRoot}/vast.toml`).text());
  p.fs.put("/home/u/.ssh/id_ed25519.pub", "ssh-ed25519 AAAAKEY marcos");
  p.fs.put(`${RECORD}/image.json`, JSON.stringify({ image: IMAGE, cap: "120", pushed: true }));
  p.fs.put(
    `${RECORD}/template.json`,
    JSON.stringify({ id: 743917, hashId: "a79f7a77", image: IMAGE }),
  );
  const head = await loadHead(p.fs, layout, "glm-5.3-flash");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  p.rental.offers = [dearDownload, cheapDownload];
  p.http.json(/8100\/props$/, {
    model_path: `/opt/rig/local/packs/glm-5.3-flash/${basename(head.value.servedPath)}`,
    total_slots: 1,
  });
  const gate: LiveGate = { run: async () => ok({ dir: "", pass: true }) };
  const uc = new RentGpu(
    { ...p, gate, self: ["/r/dist/rig"], vastai: "/home/u/.local/bin/vastai", home: "/home/u" },
    layout,
    engine,
  );
  return { p, head: head.value, engine: engine.value, uc };
}

/** the server answers /health only from the `after`-th poll on: the box's own bring-up until then */
function servesAfter(p: ReturnType<typeof fakePorts>, after: number) {
  let polls = 0;
  p.http.on(/8100\/health$/, () =>
    ++polls >= after
      ? { status: 200, text: '{"status":"ok"}' }
      : { status: 503, text: '{"error":"loading model"}' },
  );
}

describe("the template's offers", () => {
  test("the market is asked with the template's own filters, a reliable host and a download floor", async () => {
    const { head, engine } = await setup();
    expect(templateQuery(head, { cap: "120", cuda: engine.cuda?.version, diskGb: 160 })).toBe(
      `num_gpus=2 gpu_ram>=89 compute_cap=1200 cuda_max_good>=${engine.cuda?.version} disk_space>=160 gpu_mem_bw>=1300 reliability>0.97 inet_down>=800 rentable=true`,
    );
  });

  test("the cheapest box all in is not the cheapest by the hour: the pack's download is billed per GB", () => {
    const pack = 134.3 * GB;
    const [first, second] = rankOffers([dearDownload, cheapDownload], pack, 1);
    expect(first!.offer.id).toBe(cheapDownload.id);
    // 134.3 GB at 0.051 $/GB is $6.85 before the box serves anything
    expect(second!.dollars - first!.dollars).toBeGreaterThan(5);
    // the hours at dph, which carries the disk already, the boot billed too, and the GB at the host's price
    const boot = estimateOffer(cheapDownload, pack, 1).minutesToServe / 60;
    expect(first!.dollars).toBeCloseTo(3.47 * (1 + boot) + 0.003 * 134.3, 6);
    // the download at the host's own speed is most of the time to serve on a slow link
    expect(estimateOffer(dearDownload, pack, 1).minutesToServe).toBeCloseTo(3 + 19.9 + 0.75 + 1, 0);
    expect(estimateOffer(cheapDownload, pack, 1).minutesToServe).toBeCloseTo(
      3 + 2.66 + 0.75 + 1,
      0,
    );
    // a host whose RAM cannot hold the pack reads it from disk again to load it: 134.3 GB at 2 GB/s
    const fits = estimateOffer(cheapDownload, pack, 1).minutesToServe;
    const small = estimateOffer({ ...cheapDownload, ramGiB: 126 }, pack, 1).minutesToServe;
    expect(small - fits).toBeCloseTo(1.12, 1);
  });
});

describe("vast up --template", () => {
  test("a dry run ranks the offers all in and creates nothing", async () => {
    const { p, head, uc } = await setup();
    const r = await uc.upFromTemplate(head, { hours: 1, dryRun: true });
    expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(cheapDownload.id);
    expect(
      r.ok && r.value.kind === "dry-run" && r.value.ranked?.map((each) => each.offer.id),
    ).toEqual([cheapDownload.id, dearDownload.id]);
    expect(p.rental.instances.size).toBe(0);
  });

  // 2026-10-03: box 54020392 declared 7,754 Mb/s and pulled the pack at about 447 Mb/s by its own sampler
  test("a host measured slower than it declares is ranked on what it pulled at", async () => {
    const { p, head, uc } = await setup();
    p.fs.put(
      "/r/local/rented-box/download-rates.json",
      JSON.stringify([{ instanceId: 1, machineId: 33665, geo: "Norway, NO", mbps: 100, at: 0 }]),
    );
    const r = await uc.upFromTemplate(head, { hours: 1, dryRun: true });
    expect(r.ok && r.value.kind === "dry-run" && r.value.pick.id).toBe(dearDownload.id);
    if (!r.ok || r.value.kind !== "dry-run") return;
    const measured = r.value.ranked?.find((each) => each.offer.id === cheapDownload.id);
    expect(measured?.rate).toEqual({ mbps: 100, source: "host" });
    // 134 GB at 100 Mb/s is about 179 minutes, not the 2.7 the host's 6,724 would give: billed, it outweighs the
    // other host's 0.051 $/GB
    expect(measured?.minutesToServe).toBeGreaterThan(179);
  });

  test("rents from the template with rig's label, follows the box's own bring-up, then the tunnel, the timer and READY", async () => {
    const { p, head, uc } = await setup();
    servesAfter(p, 3);
    let polls = 0;
    p.ssh.on(/test -e \/var\/log\/rig\/FAILED/, () => ({
      code: 0,
      stdout: `${["2026-10-01T03:01:00Z rig: == fetch", "2026-10-01T03:04:00Z rig: == serve"][Math.min(polls++, 1)]}\n`,
      stderr: "",
    }));
    const r = await uc.upFromTemplate(head, { hours: 2 });
    expect(r.ok).toBe(true);
    if (!r.ok || r.value.kind !== "up") return;
    expect(p.rental.ops.at(-1)).toBe(`create ${cheapDownload.id} template a79f7a77 160 label rig`);
    expect(r.value).toMatchObject({
      instanceId: 1000,
      localUrl: "http://127.0.0.1:8100",
      serving: { slots: 1 },
    });
    expect(r.value.estimate?.hours).toBe(2);
    // nothing is shipped or built from here: the image and its on-start do it
    expect(p.ssh.pushed).toEqual([]);
    expect(
      p.ssh.calls.some((call) => call.includes("rig build") || call.includes("rig serve")),
    ).toBe(false);
    const log = p.log.lines.join("\n");
    expect(log).toContain("box 1000: 2026-10-01T03:01:00Z rig: == fetch");
    expect(log).toContain("box 1000: 2026-10-01T03:04:00Z rig: == serve");
    expect(log).toContain("READY: 2× RTX PRO 6000 WS box 1000");
    expect(log).toContain("ANTHROPIC_BASE_URL=http://127.0.0.1:8100");
    expect(p.systemd.ops).toContain("enable rig-vast-idle-1000.timer");
    expect(p.systemd.ops).toContain("enable rig-vast-stop-1000.timer");
    expect(p.fs.text("/r/local/rented-box/boxes/1000/instance.json")).toContain(
      '"instanceId": 1000',
    );
  });

  test("--disk-gb grows the box's disk past the head's own, in the market's pricing and the rental; never below it", async () => {
    const { p, head, uc } = await setup();
    const small = await uc.upFromTemplate(head, { hours: 1, diskGb: 100 });
    expect(!small.ok && small.message).toContain(
      "--disk-gb 100 is smaller than glm-5.3-flash's own 160 GB",
    );
    expect(p.rental.instances.size).toBe(0);
    servesAfter(p, 1);
    p.ssh.on(/test -e \/var\/log\/rig\/FAILED/, () => ({
      code: 0,
      stdout: "2026-10-01T03:04:00Z rig: == serve\n",
      stderr: "",
    }));
    const priced: number[] = [];
    const search = p.rental.searchOffers.bind(p.rental);
    p.rental.searchOffers = async (query: string, diskGb: number) => {
      priced.push(diskGb); // vast prices an offer's storage at the disk asked for
      return search(query, diskGb);
    };
    const r = await uc.upFromTemplate(head, { hours: 1, diskGb: 450 });
    expect(r.ok).toBe(true);
    expect(priced).toEqual([450]);
    expect(p.rental.ops.at(-1)).toBe(`create ${cheapDownload.id} template a79f7a77 450 label rig`);
  });

  test("a box whose supervisor gave up has its log copied here and is destroyed", async () => {
    const { p, head, uc } = await setup();
    servesAfter(p, Number.POSITIVE_INFINITY);
    p.ssh.on(/test -e \/var\/log\/rig\/FAILED/, {
      code: 0,
      stdout: "FAILED\nrig: rig up exited, try 5 of 5 in 120 s\n",
      stderr: "",
    });
    const r = await uc.upFromTemplate(head, { hours: 1 });
    expect(!r.ok && r.message).toContain("box 1000: its supervisor gave up on rig up");
    expect(!r.ok && r.message).toContain("destroyed");
    expect(p.ssh.pulled).toEqual([["/var/log/rig/up.log", "/r/local/rented-box/box-1000-up.log"]]);
    expect(p.rental.instances.size).toBe(0);
    expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(false);
  });

  test("a box that never serves within the budget is destroyed, its log kept", async () => {
    const { p, head, uc } = await setup();
    servesAfter(p, Number.POSITIVE_INFINITY);
    const r = await uc.upFromTemplate(head, { hours: 1 });
    expect(!r.ok && r.message).toContain("it did not serve within 45 min");
    expect(p.rental.instances.size).toBe(0);
  });

  test("a budget rents the cheapest offer within it all in, and none when every offer is over it", async () => {
    const { p, head, uc } = await setup();
    const cheapest = (await uc.upFromTemplate(head, { hours: 2, dryRun: true })) as {
      ok: true;
      value: { ranked: { dollars: number }[] };
    };
    const [first, second] = cheapest.value.ranked.map((each) => each.dollars);
    const over = await uc.upFromTemplate(head, { hours: 2, budget: first! - 0.01 });
    expect(!over.ok && over.message).toContain(
      `the cheapest is ~$${first!.toFixed(2)} (offer ${cheapDownload.id}); nothing rented`,
    );
    expect(p.rental.instances.size).toBe(0);
    expect(second).toBeGreaterThan(first!);
    const within = await uc.upFromTemplate(head, { hours: 2, budget: first!, dryRun: true });
    expect(within.ok && within.value.kind === "dry-run" && within.value.pick.id).toBe(
      cheapDownload.id,
    );
  });

  test("funds that do not cover the pick all in are refused before anything is rented", async () => {
    const { p, head, uc } = await setup();
    p.rental.balance = 7.19; // the credit box 53422109 drained to $0 on its download
    const r = await uc.upFromTemplate(head, { hours: 2 });
    expect(!r.ok && r.message).toMatch(
      /funds \$7\.19 cover less than the pick's 2 h all in, ~\$\d+\.\d\d/,
    );
    expect(p.rental.instances.size).toBe(0);
  });

  test("a template older than the last pushed image is refused: re-render it first", async () => {
    const { p, head, uc } = await setup();
    p.fs.put(
      `${RECORD}/template.json`,
      JSON.stringify({ hashId: "old", image: "registry.torad.ai/rig:old" }),
    );
    const r = await uc.upFromTemplate(head, { hours: 1 });
    expect(!r.ok && r.message).toContain("run rig vast template glm-5.3-flash");
  });
});
