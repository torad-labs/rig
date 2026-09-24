import { describe, expect, test } from "bun:test";
import { fakePorts } from "../../../test/fakes/index.ts";
import { loadEngine } from "../../shared/engine/engine.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { DescribeHead } from "./head-description.service.ts";

const headToml = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();
const engineToml = await Bun.file(`${import.meta.dir}/../../../engine/engine.toml`).text();

async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  p.fs.put("/r/heads/bonsai-2-27b/head.toml", headToml);
  p.fs.put("/r/engine/engine.toml", engineToml);
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  const unit = { installed: false, active: false };
  const uc = new DescribeHead({ ...p, unit: { status: async () => unit } }, layout, engine.value);
  return { p, head: head.value, engine: engine.value, uc, unit };
}

describe("describe", () => {
  test("a bare machine: the head's facts, nothing built, nothing serving", async () => {
    const { head, engine, uc } = await setup();
    const d = await uc.run(head, 0);
    expect(d).toMatchObject({
      name: "bonsai-2-27b",
      dialect: "openai-chat",
      port: 8099,
      base_url: "http://127.0.0.1:8099/v1",
      engine_commit: `${engine.sha7}`,
      model_ctx: 262144,
      advertise_ctx: 245760,
      supported_archs: ["sm_120", "sm_90"],
      this_arch: "sm_120",
      supported: true,
      built: false,
      pack_verified: false,
      unit_installed: false,
      serving: false,
      server_facts: { rejects_reasoning_effort: true, slot_pinning: true, any_model_id: true },
      sampling: ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20"],
      speculative: { type: "draft-mtp", file: null, n_max: 2 },
      repo: "/r",
      head_dir: "/r/heads/bonsai-2-27b",
    });
  });
  test("a served machine: built, pinned pack, unit, health", async () => {
    const { p, head, engine, uc, unit } = await setup();
    p.fs.put(`${engine.binDir("120")}/BUILD`, "x");
    p.fs.put(`${engine.binDir("120")}/llama-server`, "x");
    p.fs.put(head.servedPath, "pack");
    p.hasher.pinned.set(head.servedPath, head.served.sha256);
    unit.installed = true;
    unit.active = true;
    p.http.json(/\/health$/, { status: "ok" });
    expect(await uc.run(head, 0)).toMatchObject({
      built: true,
      pack_verified: true,
      unit_installed: true,
      unit_active: true,
      serving: true,
    });
  });
  test("an unmeasured card or no card is reported, not refused", async () => {
    const { p, head, uc } = await setup();
    p.gpu.card(1, { computeCap: "89", name: "RTX 4090", memoryMiB: 24564 });
    expect(await uc.run(head, 1)).toMatchObject({
      this_arch: "sm_89",
      supported: false,
      built: false,
    });
    expect(await uc.run(head, 5)).toMatchObject({ this_arch: null, supported: false });
  });
});
