import { describe, expect, test } from "bun:test";
import { ExitCode, fail, layoutAt, ok } from "@rig/core";
import { loadHead } from "@rig/head";
import { connectionRefused, fakePorts, putHead, repoRoot } from "@rig/testing";
import { BringUpHead, type BringUpSteps } from "./head-bringup.service.ts";

const headToml = await Bun.file(`${repoRoot}/heads/bonsai-2-27b/head.toml`).text();

async function setup(over: (calls: string[]) => Partial<BringUpSteps> = () => ({})) {
  const p = fakePorts();
  const layout = layoutAt("/r");
  putHead(p.fs, "/r", headToml);
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  if (!head.ok) throw new Error(head.message);
  const calls: string[] = [];
  const steps: BringUpSteps = {
    unitDevices: async () => undefined,
    room: async (_head, { devices }) => {
      calls.push("room");
      return ok({ gpus: devices === "auto" ? [0] : [...devices] }); // auto takes card 0, in this fake
    },
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
      const log = "/r/local/logs/bonsai-2-27b.log";
      return ok({ unit: "rig-bonsai-2-27b.service", log, state: "current", linger: true });
    },
    serve: async () => {
      calls.push("serve");
      return ok(0);
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
      if (polls < 3) throw connectionRefused();
      return { status: polls < 5 ? 503 : 200, text: "" };
    });
    p.http.json(/\/props$/, {
      model_path:
        "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      total_slots: 4,
    });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 4243);
    p.host.listeners.set(8099, 4243);
    const r = await uc.run(head, { devices: [0] });
    expect(calls).toEqual(["room", "prepare", "fetch", "build", "derive", "unit"]);
    expect(p.systemd.ops).toEqual(["restart rig-bonsai-2-27b.service"]);
    expect(r).toEqual({
      ok: true,
      value: {
        steps: { fetch: "present", build: "present", derive: "present", unit: "current" },
        linger: true,
        start: "started",
        serving: { model: "Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf", slots: 4 },
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
      if (polls < 3) throw connectionRefused();
      return { status: polls < 5 ? 503 : 200, text: "" };
    });
    p.http.json(/\/props$/, { model_path: undrived.value.servedPath, total_slots: 4 });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 4243);
    p.host.listeners.set(8099, 4243);
    const r = await uc.run(undrived.value, { devices: [0] });
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
    const r = await uc.run(head, { devices: [0] });
    expect(!r.ok && r.code).toBe(3);
    expect(calls).toEqual(["room", "prepare", "fetch", "build"]);
  });
  test("a card too small for the head or a disk too full for it stops before anything is installed or fetched", async () => {
    const { uc, head, calls } = await setup((calls) => ({
      room: async () => {
        calls.push("room");
        return fail(ExitCode.Unsupported, "12227 MiB of VRAM is below the smallest profile");
      },
    }));
    const r = await uc.run(head, { devices: [0] });
    expect(!r.ok && r.code).toBe(3);
    expect(calls).toEqual(["room"]);
  });
  test("the cards come from --gpu, else the installed unit, else auto; the ones the profile takes reach prepare, build and the unit", async () => {
    const seen: Record<string, unknown> = {};
    let unitHas: number[] | undefined;
    const { p, head, uc } = await setup(() => ({
      unitDevices: async () => unitHas,
      room: async (_head, { devices }) => {
        seen.room = devices;
        return ok({ gpus: devices === "auto" ? [2, 3] : [...devices] });
      },
      prepare: async (options) => {
        seen.prepare = options.gpu;
        return ok({});
      },
      build: async (options) => {
        seen.build = options.gpu;
        return ok({ dir: "/r/local/engine-builds/x", alreadyBuilt: true });
      },
      installUnit: async (_head, options) => {
        seen.unit = options.devices;
        return ok({ unit: "rig-bonsai-2-27b.service", log: "/l", state: "current", linger: true });
      },
    }));
    p.http.json(/\/health$/, { status: "ok" }); // serving: left running, the steps are what is under test
    await uc.run(head, {});
    expect(seen).toEqual({ room: "auto", prepare: 2, build: 2, unit: [2, 3] });
    unitHas = [1];
    await uc.run(head, {});
    expect(seen).toEqual({ room: [1], prepare: 1, build: 1, unit: [1] });
    await uc.run(head, { devices: [0] });
    expect(seen.room).toEqual([0]);
  });
  test("--foreground takes every step but the unit, then serves on the cards room took, its exit code the result", async () => {
    let served: unknown;
    const { p, head, uc, calls } = await setup((calls) => ({
      room: async () => {
        calls.push("room");
        return ok({ gpus: [0, 1] });
      },
      serve: async (_head, options) => {
        calls.push("serve");
        served = options;
        return ok(137); // the server killed: its code is up's
      },
    }));
    const r = await uc.foreground(head, { cacheRam: 196608 });
    expect(r).toEqual({ ok: true, value: 137 });
    expect(calls).toEqual(["room", "prepare", "fetch", "build", "derive", "serve"]);
    expect(served).toEqual({ devices: [0, 1], cacheRam: 196608, slots: undefined });
    expect(p.systemd.ops).toEqual([]); // no unit, no restart: a container has no systemd
    const failing = await setup((calls) => ({
      fetch: async () => {
        calls.push("fetch");
        return fail(ExitCode.Failure, "network");
      },
    }));
    const stopped = await failing.uc.foreground(failing.head, {});
    expect(!stopped.ok && stopped.message).toBe("network");
    expect(failing.calls).toEqual(["room", "prepare", "fetch"]);
  });
  test("a serving head is left running unless --restart", async () => {
    const { p, head, uc } = await setup();
    p.http.json(/\/health$/, { status: "ok" });
    const r = await uc.run(head, { devices: [0] });
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
    let r = await uc.run(head, { devices: [0], restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("1 request(s)");
    // slots idle between tokens, six requests queued behind them: still busy
    p.http.json(/\/slots$/, [{ id: 0, is_processing: false }]);
    p.http.on(/\/metrics$/, () => ({ status: 200, text: metrics(6) }));
    r = await uc.run(head, { devices: [0], restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("6 request(s)");
    expect(p.systemd.ops).toEqual([]);
    p.http.on(/\/metrics$/, () => ({ status: 200, text: metrics(0) }));
    p.http.json(/\/props$/, {
      model_path: "a/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      total_slots: 1,
    });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 7);
    p.host.listeners.set(8099, 7);
    r = await uc.run(head, { devices: [0], restart: true });
    expect(r.ok && r.value.start).toBe("restarted");
    expect(p.systemd.ops).toEqual(["restart rig-bonsai-2-27b.service"]);
  });
  test("--restart never restarts blind: unreadable /slots or /metrics, a --no-slots error object, or a port that times out all refuse", async () => {
    const { p, head, uc } = await setup();
    p.http.json(/\/health$/, { status: "ok" });
    p.http.json(/\/slots$/, { error: { code: 501, message: "slots disabled" } }, 501);
    let r = await uc.run(head, { devices: [0], restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("idle cannot be proved");
    p.http.json(/\/slots$/, [{ id: 0, is_processing: false }]); // /metrics still unanswered
    r = await uc.run(head, { devices: [0], restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    p.http.on(/\/health$/, () => {
      throw new Error("The operation timed out");
    });
    r = await uc.run(head, { devices: [0], restart: true });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("did not refuse the connection");
    expect(p.systemd.ops).toEqual([]);
  });
  test("a port that neither answers nor refuses is never reported as a head serving: without --restart it refuses too", async () => {
    const { p, head, uc } = await setup();
    p.http.on(/\/health$/, () => {
      throw new Error("The operation timed out");
    });
    const r = await uc.run(head, { devices: [0] });
    expect(!r.ok && r.code).toBe(ExitCode.Busy);
    expect(!r.ok && r.message).toContain("did not refuse the connection");
    expect(p.systemd.ops).toEqual([]);
  });
  test("a start whose answer is not the unit's own process serving the pinned pack is a failure, not a green line", async () => {
    const { p, head, uc } = await setup();
    let polls = 0;
    p.http.on(/\/health$/, () => {
      if (polls++ === 0) throw connectionRefused();
      return { status: 200, text: "" };
    });
    p.http.json(/\/props$/, {
      model_path: "/elsewhere/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      total_slots: 4,
    });
    let r = await uc.run(head, { devices: [0] }); // no main pid: something else holds the port
    expect(!r.ok && r.message).toContain("no main process");
    p.systemd.pids.set("rig-bonsai-2-27b.service", 9);
    p.host.listeners.set(8099, 9);
    p.http.json(/\/props$/, { model_path: "/elsewhere/other.gguf", total_slots: 4 });
    polls = 0; // the port is free again, the unit starts, a foreign pack answers
    r = await uc.run(head, { devices: [0] });
    expect(!r.ok && r.message).toContain(
      "serves other.gguf, not the pinned Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
    );
  });
  test("the port's owner must be the unit's main process: a leftover server answering 200 is named, not reported as the head", async () => {
    const { p, head, uc } = await setup();
    let polls = 0; // the port is free before each start; the answer after it is the question
    p.http.on(/\/health$/, () => {
      if (polls++ === 0) throw connectionRefused();
      return { status: 200, text: "" };
    });
    p.http.json(/\/props$/, {
      model_path:
        "/r/local/packs/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf",
      total_slots: 4,
    });
    p.systemd.pids.set("rig-bonsai-2-27b.service", 4243);
    p.host.listeners.set(8099, 1214085); // the old server, same pack, bound again first (04:31–04:39)
    let r = await uc.run(head, { devices: [0] });
    expect(!r.ok && r.message).toContain(
      "held by pid 1214085, not rig-bonsai-2-27b.service's main process (pid 4243)",
    );
    p.host.listeners.delete(8099); // no owner this user can see: not proved either
    polls = 0;
    r = await uc.run(head, { devices: [0] });
    expect(!r.ok && r.message).toContain("held by no process this user can see");
    p.host.listeners.set(8099, 4243);
    polls = 0;
    r = await uc.run(head, { devices: [0] });
    expect(r.ok && r.value.start).toBe("started");
  });
  test("a unit that dies while the head is awaited ends the wait at once, naming the unit", async () => {
    const { p, head, uc } = await setup();
    p.http.on(/\/health$/, () => {
      throw connectionRefused();
    });
    let polls = 0;
    p.systemd.isActive = async () => polls++ < 2; // restart activates it, ExecStartPre then fails
    const r = await uc.run(head, { devices: [0], healthTimeoutMs: 60_000 });
    expect(!r.ok && r.message).toContain("is no longer active");
    expect(p.clock.slept.length).toBeLessThan(5);
  });
  test("a head that never comes healthy fails after the timeout", async () => {
    const { p, head, uc } = await setup();
    p.http.on(/\/health$/, () => {
      throw connectionRefused();
    });
    const r = await uc.run(head, { devices: [0], healthTimeoutMs: 5000 });
    expect(!r.ok && r.message).toContain("did not come up");
    // the unit appends the server's output to its log file, not the journal
    expect(!r.ok && r.message).toContain("its log: tail -n 40 /r/local/logs/bonsai-2-27b.log");
    expect(p.clock.slept.length).toBe(3);
  });
});
