import { describe, expect, test } from "bun:test";
import { layoutAt, ok } from "@rig/core";
import { loadEngine } from "@rig/engine";
import { loadHead } from "@rig/head";
import { GuardBox, RentGpu, SweepStopped } from "@rig/rental";
import { fakePorts, putHead, repoRoot } from "@rig/testing";
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
  test("the subcommands' flags are exactly the ones the usage line names", async () => {
    const source = await Bun.file(`${import.meta.dir}/vast.command.ts`).text();
    const usage = /const USAGE =\s*"([^"]*)"/.exec(source)![1]!;
    const named = new Set([...usage.matchAll(/--([a-z][a-z-]*)/g)].map((m) => m[1]!));
    expect(new Set(Object.values(SUBCOMMAND_FLAGS).flat())).toEqual(named);
  });
});
