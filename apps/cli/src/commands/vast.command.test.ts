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
  const uc = new RentGpu({ ...p, gate, self: ["/r/dist/rig"], home: "/home/u" }, layout, engine);
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
  test("the subcommands' flags are exactly the ones the usage line names", async () => {
    const source = await Bun.file(`${import.meta.dir}/vast.command.ts`).text();
    const usage = /const USAGE =\s*"([^"]*)"/.exec(source)![1]!;
    const named = new Set([...usage.matchAll(/--([a-z][a-z-]*)/g)].map((m) => m[1]!));
    expect(new Set(Object.values(SUBCOMMAND_FLAGS).flat())).toEqual(named);
  });
});
