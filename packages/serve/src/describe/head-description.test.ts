import { describe, expect, test } from "bun:test";
import type { Devices } from "@rig/core";
import { layoutAt } from "@rig/core";
import { loadEngine } from "@rig/engine";
import { loadHead } from "@rig/head";
import { fakePorts, repoRoot } from "@rig/testing";
import { DescribeHead } from "./head-description.service.ts";

const headToml = await Bun.file(`${repoRoot}/heads/bonsai-2-27b/head.toml`).text();
const engineToml = await Bun.file(`${repoRoot}/engine/engine.toml`).text();

async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  p.fs.put("/r/heads/bonsai-2-27b/head.toml", headToml);
  p.fs.put("/r/engine/engine.toml", engineToml);
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  const unit: { installed: boolean; active: boolean; devices?: Devices } = {
    installed: false,
    active: false,
  };
  const reader = { status: async () => unit, devices: async () => unit.devices };
  const uc = new DescribeHead({ ...p, unit: reader }, layout, engine.value);
  return { p, head: head.value, engine: engine.value, uc, unit };
}

describe("describe", () => {
  test("a bare machine: the head's facts, nothing built, nothing serving", async () => {
    const { head, engine, uc } = await setup();
    const d = await uc.run(head, [0]);
    expect(d).toMatchObject({
      name: "bonsai-2-27b",
      dialect: "openai-chat",
      port: 8099,
      base_url: "http://127.0.0.1:8099/v1",
      engine_commit: `${engine.sha7}`,
      model_ctx: 262144,
      advertise_ctx: 245760,
      supported_archs: ["sm_120", "sm_90"],
      this_gpu: 0,
      this_gpus: [0],
      this_arch: "sm_120",
      profile: "16000 MiB, 4 slots × 294912",
      supported: true,
      built: false,
      pack_verified: false,
      unit_installed: false,
      serving: false,
      server_facts: { rejects_reasoning_effort: true, slot_pinning: true, any_model_id: true },
      sampling: ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20"],
      speculative: { type: "draft-mtp", file: null, n_max: 3 },
      repo: "/r",
      head_dir: "/r/heads/bonsai-2-27b",
    });
  });
  test("a served machine: built, pinned pack, unit, health", async () => {
    const { p, head, engine, uc, unit } = await setup();
    p.fs.put(`${engine.binDir("120")}/BUILD`, "x");
    p.fs.put(`${engine.binDir("120")}/llama-server`, "x");
    p.fs.put(head.servedPath, "pack");
    p.hasher.pinned.set(head.servedPath, head.servedFiles[0]!.sha256);
    unit.installed = true;
    unit.active = true;
    p.http.json(/\/health$/, { status: "ok" });
    expect(await uc.run(head, [0])).toMatchObject({
      built: true,
      pack_verified: true,
      unit_installed: true,
      unit_active: true,
      serving: true,
    });
  });
  test("without --gpu, the cards the installed unit serves on, else the ones the head's profile takes of every card", async () => {
    const { p, head, uc, unit } = await setup();
    p.gpu.card(0, { usedMiB: 8000 }); // a desktop's share: the head fits no profile beside it
    p.gpu.card(1, { name: "NVIDIA GeForce RTX 5090", memoryMiB: 32607 });
    expect(await uc.run(head)).toMatchObject({
      this_gpu: 1,
      this_gpus: [1],
      profile: "30000 MiB, 8 slots × 786432",
    });
    unit.devices = [0];
    expect(await uc.run(head)).toMatchObject({ this_gpu: 0, this_gpus: [0], profile: null });
  });
  test("an unmeasured card or no card is reported, not refused", async () => {
    const { p, head, uc } = await setup();
    p.gpu.card(1, { computeCap: "89", name: "RTX 4090", memoryMiB: 24564 });
    expect(await uc.run(head, [1])).toMatchObject({
      this_arch: "sm_89",
      supported: false,
      built: false,
    });
    expect(await uc.run(head, [5])).toMatchObject({ this_arch: null, supported: false });
  });
});
