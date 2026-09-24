import { describe, expect, test } from "bun:test";
import { putHead } from "../../../test/fakes/head-fixtures.ts";
import { fakePorts } from "../../../test/fakes/index.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { ExitCode, fail, ok } from "../../shared/result.ts";
import { BringUpHead, type BringUpSteps } from "./head-bringup.service.ts";

const headToml = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();

async function setup(over: (calls: string[]) => Partial<BringUpSteps> = () => ({})) {
  const p = fakePorts();
  const layout = layoutAt("/r");
  putHead(p.fs, "/r", headToml);
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  if (!head.ok) throw new Error(head.message);
  const calls: string[] = [];
  const steps: BringUpSteps = {
    prepare: async () => {
      calls.push("prepare");
      return ok({});
    },
    fetch: async () => {
      calls.push("fetch");
      return ok({ state: "present" });
    },
    build: async () => {
      calls.push("build");
      return ok({ dir: "/r/local/engine-builds/x", alreadyBuilt: true });
    },
    derive: async () => {
      calls.push("derive");
      return ok({ state: "present" });
    },
    installUnit: async () => {
      calls.push("unit");
      return ok({ unit: "rig-bonsai-2-27b.service", state: "current" });
    },
    ...over(calls),
  };
  return { p, head: head.value, uc: new BringUpHead({ ...p, steps }), calls };
}

describe("up", () => {
  test("runs the steps in order and starts a head nobody is serving, waiting for health", async () => {
    const { p, head, uc, calls } = await setup();
    let polls = 0;
    p.http.on(/\/health$/, () => {
      polls++;
      if (polls < 3) throw new Error("ECONNREFUSED");
      return { status: polls < 5 ? 503 : 200, text: "" };
    });
    p.http.json(/\/props$/, {
      model_path: "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010.gguf",
      total_slots: 4,
    });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 4243);
    p.host.listeners.set(8099, 4243);
    const r = await uc.run(head, { gpu: 0 });
    expect(calls).toEqual(["prepare", "fetch", "build", "derive", "unit"]);
    expect(p.systemd.ops).toEqual(["restart rig-bonsai-2-27b.service"]);
    expect(r).toEqual({
      ok: true,
      value: {
        steps: { fetch: "present", build: "present", derive: "present", unit: "current" },
        start: "started",
        serving: { model: "Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010.gguf", slots: 4 },
      },
    });
    expect(p.log.lines.at(-1)).toContain("pid 4243");
    expect(p.clock.slept).toEqual([2000, 2000, 2000]);
  });
  test("an undrived head names why in the final line", async () => {
    const { p, uc } = await setup();
    const layout = layoutAt("/r");
    await p.fs.remove("/r/heads/bonsai-2-27b/assets/lora/bonsai-abliterate-lora.gguf");
    const undrived = await loadHead(p.fs, layout, "bonsai-2-27b");
    if (!undrived.ok) throw new Error(undrived.message);
    expect(undrived.value.undrived).toBeDefined();
    let polls = 0;
    p.http.on(/\/health$/, () => {
      polls++;
      if (polls < 3) throw new Error("ECONNREFUSED");
      return { status: polls < 5 ? 503 : 200, text: "" };
    });
    p.http.json(/\/props$/, { model_path: undrived.value.servedPath, total_slots: 4 });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 4243);
    p.host.listeners.set(8099, 4243);
    const r = await uc.run(undrived.value, { gpu: 0 });
    expect(r.ok).toBe(true);
    expect(p.log.lines.at(-1)).toContain(`UNDRIVED: ${undrived.value.undrived}`);
  });
  test("a failing step stops the sequence with that step's exit code", async () => {
    const { uc, head, calls } = await setup((calls) => ({
      build: async () => {
        calls.push("build");
        return fail(ExitCode.Unsupported, "sm_89");
      },
    }));
    const r = await uc.run(head, { gpu: 0 });
    expect(!r.ok && r.code).toBe(3);
    expect(calls).toEqual(["prepare", "fetch", "build"]);
  });
  test("a serving head is left running unless --restart", async () => {
    const { p, head, uc } = await setup();
    p.http.json(/\/health$/, { status: "ok" });
    const r = await uc.run(head, { gpu: 0 });
    expect(r.ok && r.value.start).toBe("left-running");
    expect(p.systemd.ops).toEqual([]);
  });
  const metrics = (deferred: number) =>
    `# HELP\nllamacpp:requests_processing 0\nllamacpp:requests_deferred ${deferred}\n`;
  test("--restart is refused with exit 2 while a slot is processing or a request is queued, and restarts when idle", async () => {
    const { p, head, uc } = await setup();
    p.http.json(/\/health$/, { status: "ok" });
    p.http.json(/\/slots$/, [
      { id: 0, is_processing: true },
      { id: 1, is_processing: false },
    ]);
    p.http.on(/\/metrics$/, () => ({ status: 200, text: metrics(0) }));
    let r = await uc.run(head, { gpu: 0, restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("1 request(s)");
    // slots idle between tokens, six requests queued behind them: still busy
    p.http.json(/\/slots$/, [{ id: 0, is_processing: false }]);
    p.http.on(/\/metrics$/, () => ({ status: 200, text: metrics(6) }));
    r = await uc.run(head, { gpu: 0, restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("6 request(s)");
    expect(p.systemd.ops).toEqual([]);
    p.http.on(/\/metrics$/, () => ({ status: 200, text: metrics(0) }));
    p.http.json(/\/props$/, {
      model_path: "a/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010.gguf",
      total_slots: 1,
    });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 7);
    p.host.listeners.set(8099, 7);
    r = await uc.run(head, { gpu: 0, restart: true });
    expect(r.ok && r.value.start).toBe("restarted");
    expect(p.systemd.ops).toEqual(["restart rig-bonsai-2-27b.service"]);
  });
  test("--restart never restarts blind: unreadable /slots or /metrics, a --no-slots error object, or a port that times out all refuse", async () => {
    const { p, head, uc } = await setup();
    p.http.json(/\/health$/, { status: "ok" });
    p.http.json(/\/slots$/, { error: { code: 501, message: "slots disabled" } }, 501);
    let r = await uc.run(head, { gpu: 0, restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("idle cannot be proved");
    p.http.json(/\/slots$/, [{ id: 0, is_processing: false }]); // /metrics still unanswered
    r = await uc.run(head, { gpu: 0, restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    p.http.on(/\/health$/, () => {
      throw new Error("The operation timed out");
    });
    r = await uc.run(head, { gpu: 0, restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("did not refuse the connection");
    expect(p.systemd.ops).toEqual([]);
  });
  test("a start whose answer is not the unit's own process serving the pinned pack is a failure, not a green line", async () => {
    const { p, head, uc } = await setup();
    let polls = 0;
    p.http.on(/\/health$/, () => {
      if (polls++ === 0) throw new Error("ECONNREFUSED");
      return { status: 200, text: "" };
    });
    p.http.json(/\/props$/, {
      model_path: "/elsewhere/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010.gguf",
      total_slots: 4,
    });
    let r = await uc.run(head, { gpu: 0 }); // no main pid: something else holds the port
    expect(!r.ok && r.message).toContain("no main process");
    p.systemd.pids.set("rig-bonsai-2-27b.service", 9);
    p.host.listeners.set(8099, 9);
    p.http.json(/\/props$/, { model_path: "/elsewhere/other.gguf", total_slots: 4 });
    polls = 0; // the port is free again, the unit starts, a foreign pack answers
    r = await uc.run(head, { gpu: 0 });
    expect(!r.ok && r.message).toContain(
      "serves other.gguf, not the pinned Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010.gguf",
    );
  });
  test("the port's owner must be the unit's main process: a leftover server answering 200 is named, not reported as the head", async () => {
    const { p, head, uc } = await setup();
    let polls = 0; // the port is free before each start; the answer after it is the question
    p.http.on(/\/health$/, () => {
      if (polls++ === 0) throw new Error("ECONNREFUSED");
      return { status: 200, text: "" };
    });
    p.http.json(/\/props$/, {
      model_path: "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010.gguf",
      total_slots: 4,
    });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 4243);
    p.host.listeners.set(8099, 1214085); // the old server, same pack, bound again first (04:31–04:39)
    let r = await uc.run(head, { gpu: 0 });
    expect(!r.ok && r.message).toContain(
      "held by pid 1214085, not rig-bonsai-2-27b.service's main process (pid 4243)",
    );
    p.host.listeners.delete(8099); // ss cannot name an owner: not proved either
    polls = 0;
    r = await uc.run(head, { gpu: 0 });
    expect(!r.ok && r.message).toContain("held by no process ss can name");
    p.host.listeners.set(8099, 4243);
    polls = 0;
    r = await uc.run(head, { gpu: 0 });
    expect(r.ok && r.value.start).toBe("started");
  });
  test("a unit that dies while the head is awaited ends the wait at once, naming the unit", async () => {
    const { p, head, uc } = await setup();
    p.http.on(/\/health$/, () => {
      throw new Error("ECONNREFUSED");
    });
    let polls = 0;
    p.systemd.isActive = async () => polls++ < 2; // restart activates it, ExecStartPre then fails
    const r = await uc.run(head, { gpu: 0, healthTimeoutMs: 60_000 });
    expect(!r.ok && r.message).toContain("is no longer active");
    expect(p.clock.slept.length).toBeLessThan(5);
  });
  test("a head that never comes healthy fails after the timeout", async () => {
    const { p, head, uc } = await setup();
    p.http.on(/\/health$/, () => {
      throw new Error("ECONNREFUSED");
    });
    const r = await uc.run(head, { gpu: 0, healthTimeoutMs: 5000 });
    expect(!r.ok && r.message).toContain("did not come up");
    expect(p.clock.slept.length).toBe(3);
  });
});
