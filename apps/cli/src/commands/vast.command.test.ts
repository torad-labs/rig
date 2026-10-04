import { describe, expect, test } from "bun:test";
import { layoutAt, ok } from "@rig/core";
import { loadEngine } from "@rig/engine";
import { loadHead } from "@rig/head";
import { GuardBox, RentGpu, SweepStopped } from "@rig/rental";
import { fakePorts, putHead, repoRoot } from "@rig/testing";
import { UsageError } from "../cli/args.ts";
import { gpuRentalCommand, SUBCOMMAND_FLAGS } from "./vast.command.ts";

async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  putHead(p.fs, "/r", await Bun.file(`${repoRoot}/heads/bonsai-2-27b/head.toml`).text());
  p.fs.put("/r/engine/engine.toml", await Bun.file(`${repoRoot}/engine/engine.toml`).text());
  p.fs.put("/r/vast.toml", await Bun.file(`${repoRoot}/vast.toml`).text());
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  const gate = { run: async () => ok({ dir: "/r/local/gate-runs/bonsai-2-27b/live", pass: true }) };
  const uc = new RentGpu(
    { ...p, gate, self: ["/r/dist/rig"], vastai: "/home/u/.local/bin/vastai", home: "/home/u" },
    layout,
    engine,
  );
  return { p, head: head.value, uc };
}

describe("vast command", () => {
  test("a subcommand refuses another subcommand's flag before it acts: down --dry-run destroys nothing", async () => {
    const { p, head, uc } = await setup();
    p.fs.put("/r/local/rented-box/instance.json", JSON.stringify({ instanceId: 7 })); // a box to destroy
    const guard = new GuardBox({ ...p, box: {} });
    const sweep = new SweepStopped({ ...p, self: ["/r/dist/rig"] }, layoutAt("/r"));
    const vast = gpuRentalCommand(uc, null, guard, sweep, async () => ok(head), p.log);
    const code = await vast.run({ positionals: ["down"], flags: { "dry-run": true }, dashed: [] });
    expect(code).toBe(64);
    expect(p.log.lines.join("\n")).toContain("vast down does not take --dry-run");
    expect(p.rental.ops.some((op) => op.startsWith("destroy"))).toBe(false);
  });
  test("lab needs a card class and takes no head's flags; either refusal rents nothing", async () => {
    const { p, head, uc } = await setup();
    const guard = new GuardBox({ ...p, box: {} });
    const sweep = new SweepStopped({ ...p, self: ["/r/dist/rig"] }, layoutAt("/r"));
    const vast = gpuRentalCommand(uc, null, guard, sweep, async () => ok(head), p.log);
    expect(await vast.run({ positionals: ["lab"], flags: {}, dashed: [] })).toBe(64);
    expect(p.log.lines.join("\n")).toContain("usage: rig vast lab --gpu CLASS");
    const private_ = {
      positionals: ["lab"],
      flags: { gpu: "RTX_5090", private: true },
      dashed: [],
    };
    expect(await vast.run(private_)).toBe(64);
    expect(p.log.lines.join("\n")).toContain("vast lab does not take --private");
    expect(p.rental.ops).toEqual([]);
  });
  describe("--min-down-mbps", () => {
    async function command() {
      const t = await setup();
      t.p.fs.put("/home/u/.ssh/id_ed25519.pub", "ssh-ed25519 AAAAKEY marcos");
      const guard = new GuardBox({ ...t.p, box: {} });
      const sweep = new SweepStopped({ ...t.p, self: ["/r/dist/rig"] }, layoutAt("/r"));
      const vast = gpuRentalCommand(t.uc, null, guard, sweep, async () => ok(t.head), t.p.log);
      return { ...t, vast };
    }
    const card = {
      id: 50263001,
      gpu: "RTX 5090",
      gpus: 1,
      gpuRamMiB: 32607,
      computeCap: "120",
      dph: 0.41,
      geo: "Texas, US",
      cpu: "AMD EPYC",
      ramGiB: 92,
      bandwidth: 1792,
      cudaMaxGood: 13.2,
      reliability: 0.99,
      downMbps: 251,
      downCostPerGb: 0.003,
      storagePerHour: 0.01,
      machineId: 41200,
    };
    const run = (
      vast: { run(args: never): Promise<number> },
      flags: Record<string, string | boolean>,
    ) =>
      vast.run({
        positionals: ["lab"],
        flags: { gpu: "RTX_5090", "dry-run": true, ...flags },
        dashed: [],
      } as never);
    test("reaches the offer pick: a slow market fails the command, a fast offer passes it", async () => {
      const { p, vast } = await command();
      p.rental.offers = [card];
      expect(await run(vast, { "min-down-mbps": "2000" })).toBe(1);
      expect(p.log.lines.join("\n")).toContain("no offer downloads at 2000 Mb/s or more");
      p.rental.offers = [card, { ...card, id: 50263002, dph: 0.6, downMbps: 5436 }];
      expect(await run(vast, { "min-down-mbps": "2000" })).toBe(0);
      expect(await run(vast, {})).toBe(0);
    });
    test("takes a whole number above zero; anything else is refused before the market is read", async () => {
      const { p, vast } = await command();
      p.rental.offers = [card];
      for (const bad of ["0", "-5"]) expect(await run(vast, { "min-down-mbps": bad })).toBe(64);
      expect(p.log.lines.join("\n")).toContain("--min-down-mbps take a whole number above zero");
      for (const bad of ["fast", "2.5", true])
        await expect(run(vast, { "min-down-mbps": bad })).rejects.toBeInstanceOf(UsageError);
      expect(p.rental.ops).toEqual([]);
    });
    test("lab --pack names the head whose pack is priced, --hours its session; --hours alone is refused", async () => {
      const { p, vast } = await command();
      const loaded: string[] = [];
      const t = await setup();
      const guard = new GuardBox({ ...p, box: {} });
      const sweep = new SweepStopped({ ...p, self: ["/r/dist/rig"] }, layoutAt("/r"));
      const lab = gpuRentalCommand(
        t.uc,
        null,
        guard,
        sweep,
        async (name) => {
          loaded.push(name);
          return ok(t.head);
        },
        t.p.log,
      );
      t.p.fs.put("/home/u/.ssh/id_ed25519.pub", "ssh-ed25519 AAAAKEY marcos");
      t.p.rental.offers = [card];
      expect(await run(lab, { pack: "bonsai-2-27b", hours: "2", json: true })).toBe(0);
      expect(loaded).toEqual(["bonsai-2-27b"]);
      expect(t.p.log.lines.join("\n")).toContain("all in");
      expect(await run(vast, { hours: "2" })).toBe(64);
      expect(p.log.lines.join("\n")).toContain("--hours prices the session --pack names");
      expect(await run(lab, { pack: "bonsai-2-27b", hours: "0" })).toBe(64);
    });
    test("up --template refuses it: that form ranks by the download already and floors at 800 Mb/s", async () => {
      const { p, head, vast } = await command();
      const code = await vast.run({
        positionals: ["up", head.name],
        flags: { template: true, "min-down-mbps": "2000" },
        dashed: [],
      });
      expect(code).toBe(64);
      expect(p.log.lines.join("\n")).toContain("--min-down-mbps");
      expect(p.rental.ops).toEqual([]);
    });
  });
  describe("more than one box", () => {
    /** boxes 1000 and 1001 held, the older first, both listed as running */
    async function held() {
      const t = await setup();
      const guard = new GuardBox({ ...t.p, box: {} });
      const sweep = new SweepStopped({ ...t.p, self: ["/r/dist/rig"] }, layoutAt("/r"));
      const vast = gpuRentalCommand(t.uc, null, guard, sweep, async () => ok(t.head), t.p.log);
      for (const [age, id] of [
        [2, 1000],
        [1, 1001],
      ] as const) {
        t.p.rental.instances.set(id, { id, status: "running", label: "rig", dph: 0.5 });
        t.p.fs.put(
          `/r/local/rented-box/boxes/${id}/instance.json`,
          JSON.stringify({
            instanceId: id,
            offerId: 1,
            gpu: "RTX 5090",
            cap: "12.0",
            dph: 0.5,
            geo: "Texas, US",
            createdAt: Date.now() - age * 3_600_000,
          }),
        );
      }
      return { ...t, vast };
    }
    const args = (subcommand: string, flags: Record<string, string | boolean> = {}) => ({
      positionals: [subcommand],
      flags,
      dashed: [],
    });
    test("--box names the box down, status, idle-check and bench act on; none named with two held is refused", async () => {
      const { p, vast } = await held();
      for (const subcommand of ["down", "idle-check", "bench"]) {
        const at = p.log.lines.length;
        const run =
          subcommand === "bench"
            ? { ...args(subcommand), positionals: ["bench", "bonsai-2-27b"] }
            : args(subcommand);
        expect(await vast.run(run)).toBe(64);
        expect(p.log.lines.slice(at).join("\n")).toContain(
          "2 boxes held: name one with --box (1000, 1001)",
        );
      }
      let at = p.log.lines.length;
      expect(await vast.run(args("status", { box: "1001" }))).toBe(0);
      const one = p.log.lines.slice(at).join("\n");
      expect(one).toContain("1001");
      expect(one).not.toContain("1000");
      at = p.log.lines.length;
      expect(await vast.run(args("status"))).toBe(0);
      const both = p.log.lines.slice(at).join("\n");
      expect(both.indexOf("1000")).toBeGreaterThan(-1);
      expect(both.indexOf("1000")).toBeLessThan(both.indexOf("1001"));
      at = p.log.lines.length;
      expect(
        await vast.run({ ...args("bench", { box: "7" }), positionals: ["bench", "bonsai-2-27b"] }),
      ).not.toBe(0);
      expect(p.log.lines.slice(at).join("\n")).toContain("no box 7 held here");
      expect(p.rental.ops).toEqual([]);
      expect(await vast.run(args("down", { box: "1001" }))).toBe(0);
      expect(p.rental.ops).toEqual(["destroy 1001"]);
      expect(await p.fs.exists("/r/local/rented-box/boxes/1000/instance.json")).toBe(true);
      // with one box left, naming none acts on it
      expect(await vast.run(args("down"))).toBe(0);
      expect(p.rental.ops).toEqual(["destroy 1001", "destroy 1000"]);
    });
    test("--box takes an instance id and --max-hours a quarter hour to a week; anything else acts on nothing", async () => {
      const { p, vast } = await held();
      for (const subcommand of ["down", "status", "idle-check"]) {
        expect(await vast.run(args(subcommand, { box: "0" }))).toBe(64);
        await expect(vast.run(args(subcommand, { box: "box7" }))).rejects.toBeInstanceOf(
          UsageError,
        );
      }
      expect(p.log.lines.join("\n")).toContain("usage: --box takes a box's instance id");
      const lab = (hours: string) => vast.run(args("lab", { gpu: "RTX_5090", "max-hours": hours }));
      const up = (hours: string) =>
        vast.run({
          ...args("up", { gpu: "RTX_5090", "max-hours": hours }),
          positionals: ["up", "bonsai-2-27b"],
        });
      for (const rent of [lab, up]) {
        for (const out of ["0", "0.1", "169"]) expect(await rent(out)).toBe(64);
        for (const bad of ["-2", "soon"])
          await expect(rent(bad)).rejects.toBeInstanceOf(UsageError);
      }
      expect(p.log.lines.join("\n")).toContain("usage: --max-hours takes 0.25 to 168 hours");
      expect(p.rental.ops).toEqual([]);
      expect(p.rental.instances.size).toBe(2);
    });
  });
  test("the subcommands' flags are exactly the ones the usage line names", async () => {
    const source = await Bun.file(`${import.meta.dir}/vast.command.ts`).text();
    const usage = /const USAGE =\s*"([^"]*)"/.exec(source)![1]!;
    const named = new Set([...usage.matchAll(/--([a-z][a-z-]*)/g)].map((m) => m[1]!));
    expect(new Set(Object.values(SUBCOMMAND_FLAGS).flat())).toEqual(named);
  });
});
