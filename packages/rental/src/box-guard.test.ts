import { describe, expect, test } from "bun:test";
import type { Head } from "@rig/head";
import { type FakePorts, fakePorts } from "@rig/testing";
import { GuardBox, type GuardOptions } from "./box-guard.ts";

const head = { name: "glm", port: 8102, packsDir: "/r/local/packs/glm" } as unknown as Head;
const STOP = /console\.vast\.ai\/api\/v0\/instances\/4242\//;
const MIN = 60_000;
const T0 = fakePorts().clock.now();
/** minutes since the fake clock's start */
const minute0 = (p: FakePorts) => (p.clock.now() - T0) / MIN;

/** a box whose server answers with fixed counters, its card idle, its pack whole, vast taking a stop */
function box(): FakePorts & { stops: () => number } {
  const p = fakePorts();
  p.fs.put("/r/local/packs/glm/pack.gguf", "whole");
  p.http.on(/127\.0\.0\.1:8102\/metrics/, () => ({
    status: 200,
    text: "llamacpp:prompt_tokens_total 10\nllamacpp:tokens_predicted_total 5\nllamacpp:requests_processing 0\n",
  }));
  p.http.on(STOP, () => ({ status: 200, text: '{"success":true}' }));
  return Object.assign(p, { stops: () => p.http.requests.filter((r) => STOP.test(r.url)).length });
}
const guard = (
  p: FakePorts,
  id: string | undefined = "4242",
  options: Partial<GuardOptions> = {},
) =>
  new GuardBox({ ...p, box: { id, apiKey: "instance-key" } }).run(head, {
    idleMinutes: 60,
    ...options,
  });

describe("vast guard", () => {
  test("an hour with nothing served, computed or fetched stops the box, with the box's own key", async () => {
    const p = box();
    const start = p.clock.now();
    const r = await guard(p);
    expect(r.ok && r.value.action).toBe("stopped");
    expect(p.clock.now() - start).toBe(60 * MIN);
    const stop = p.http.requests.find((request) => STOP.test(request.url))!;
    expect(stop).toMatchObject({
      method: "PUT",
      body: { state: "stopped" },
      headers: { authorization: "Bearer instance-key" },
    });
  });

  test("tokens moving, a request in flight, a busy CPU or a growing download each restart the hour", async () => {
    const start = fakePorts().clock.now();
    const minute = (p: FakePorts) => (p.clock.now() - start) / MIN;
    // tokens move for the first 30 minutes
    const tokens = box();
    tokens.http.on(/:8102\/metrics/, () => ({
      status: 200,
      text: `llamacpp:tokens_predicted_total ${Math.min(minute(tokens), 30)}\n`,
    }));
    await guard(tokens);
    expect(minute(tokens)).toBe(90);
    // a request is in flight until minute 20, the counters unchanged
    const busy = box();
    busy.http.on(/:8102\/metrics/, () => ({
      status: 200,
      text: `llamacpp:requests_processing ${minute(busy) < 20 ? 1 : 0}\n`,
    }));
    await guard(busy);
    expect(minute(busy)).toBe(79); // the last tick with a request in flight is minute 19
    // a bench over ssh holds a core until minute 45: the container's CPU moves a minute's worth each tick
    const card = box();
    card.host.cpuMicros = async () => Math.min(minute(card), 45) * 60e6;
    await guard(card);
    expect(minute(card)).toBe(105); // busy over the tick that ends at minute 45, idle from there
    // half a core is busy, a quarter is not
    for (const [share, stopped] of [
      [0.5, 105],
      [0.25, 60],
    ] as const) {
      const p = box();
      p.host.cpuMicros = async () => Math.min(minute(p), 45) * 60e6 * share;
      await guard(p);
      expect(minute(p), `${share} of a core`).toBe(stopped);
    }
    // the first start's download grows until minute 40, the server not up yet
    const fetching = box();
    fetching.http.on(/:8102\/metrics/, () => {
      throw Object.assign(new Error("refused"), { name: "ConnectionRefused" });
    });
    const sleep = fetching.clock.sleep.bind(fetching.clock);
    fetching.clock.sleep = async (ms) => {
      await sleep(ms);
      if (minute(fetching) <= 40)
        fetching.fs.put("/r/local/packs/glm/pack.gguf.part", "x".repeat(minute(fetching)));
    };
    await guard(fetching);
    expect(minute(fetching)).toBe(100);
    // aria2c's sparse .part: its size is the whole shard's from the start, only its writes move
    const sparse = box();
    sparse.http.on(/:8102\/metrics/, () => {
      throw Object.assign(new Error("refused"), { name: "ConnectionRefused" });
    });
    sparse.fs.put("/r/local/packs/glm/shard.gguf.part", "x".repeat(64));
    const tick = sparse.clock.sleep.bind(sparse.clock);
    sparse.clock.sleep = async (ms) => {
      await tick(ms);
      if (minute(sparse) <= 70)
        sparse.fs.mtimes.set("/r/local/packs/glm/shard.gguf.part", sparse.clock.now());
    };
    await guard(sparse);
    expect(minute(sparse)).toBe(130);
  });

  test("a card that reads 100 % with nothing on it does not keep the box up (the first PRO 6000 box read so)", async () => {
    const p = box();
    p.gpu.utilization = async () => [100, 100];
    const r = await guard(p, "4242", { maxHours: 3 }); // a guard that read the cards would stop at 180, for its lifetime
    expect(r.ok && r.value).toEqual({ action: "stopped", why: "idle", idleMinutes: 60 });
    expect(minute0(p)).toBe(60);
  });

  test("the supervisor's give-up file stops the box at its tick, busy or not", async () => {
    const p = box();
    p.host.cpuMicros = async () => minute0(p) * 60e6; // busy all along
    const sleep = p.clock.sleep.bind(p.clock);
    p.clock.sleep = async (ms) => {
      await sleep(ms);
      if (minute0(p) === 12) p.fs.put("/var/log/rig/FAILED", "");
    };
    const r = await guard(p, "4242", { stopWhen: "/var/log/rig/FAILED" });
    expect(r.ok && r.value.why).toBe("gave up");
    expect(minute0(p)).toBe(12);
    expect(p.log.lines.join("\n")).toContain("/var/log/rig/FAILED is there, rig up gave up");
  });

  test("the lifetime cap stops a box that never reads idle", async () => {
    const p = box();
    p.host.cpuMicros = async () => minute0(p) * 60e6;
    const r = await guard(p, "4242", { maxHours: 3 });
    expect(r.ok && r.value.why).toBe("lifetime");
    expect(minute0(p)).toBe(180);
  });

  test("a stop vast does not take is asked again at the next tick, the box still billing meanwhile", async () => {
    const p = box();
    let calls = 0;
    p.http.on(STOP, () =>
      ++calls === 1 ? { status: 503, text: "busy" } : { status: 200, text: "{}" },
    );
    const r = await guard(p);
    expect(r.ok).toBe(true);
    expect(p.stops()).toBe(2);
    expect(p.log.lines.join("\n")).toContain("vast did not take the stop (HTTP 503)");
  });

  test("off a vast box it refuses rather than guard nothing", async () => {
    const p = box();
    const r = await guard(p, "");
    expect(!r.ok && r.message).toContain("CONTAINER_ID");
    expect(p.stops()).toBe(0);
  });
});
