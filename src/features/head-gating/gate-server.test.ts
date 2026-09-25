import { describe, expect, test } from "bun:test";
import { withSidecarDraft } from "../../../test/fakes/head-fixtures.ts";
import { connectionRefused, fakePorts } from "../../../test/fakes/index.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { GateServer } from "./gate-server.ts";

const headToml = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();

async function setup(toml = headToml) {
  const p = fakePorts();
  p.fs.put("/r/heads/bonsai-2-27b/head.toml", toml);
  const head = await loadHead(p.fs, layoutAt("/r"), "bonsai-2-27b");
  if (!head.ok) throw new Error(head.message);
  return {
    p,
    head: head.value,
    server: new GateServer(
      p,
      head.value,
      "/r/local/engine-builds/60feea0-sm120",
      1,
      8098,
      "/r/local/gate-runs/run",
    ),
  };
}

describe("gate server", () => {
  test("a leg is the head's runtime args on the gate card and port, with the pack and geometry the probe chose", async () => {
    const { head, server } = await setup();
    const argv = server.argv({ label: "x", pack: "/packs/served.gguf", ctx: 8192, slots: 1 });
    expect(argv.slice(0, 10)).toEqual([
      "/r/local/engine-builds/60feea0-sm120/llama-server",
      "-m",
      "/packs/served.gguf",
      "-ngl",
      "99",
      "--jinja",
      "-fa",
      "on",
      "--cache-type-k",
      "q4_0",
    ]);
    expect(argv).toContain(head.path("assets/chat-template.jinja"));
    expect(argv.slice(-10)).toEqual([
      "--checkpoint-every",
      "16384",
      "-c",
      "8192",
      "-np",
      "1",
      "--host",
      "127.0.0.1",
      "--port",
      "8098",
    ]);
    expect(argv).not.toContain("--metrics"); // a gate leg is not a serving head: no unit flags, no sampling
    expect(argv).not.toContain("--temp");
  });
  test("draft: true adds the in-pack head's flags after the runtime args", async () => {
    const { server } = await setup();
    const argv = server.argv({
      label: "x",
      pack: "/packs/served.gguf",
      ctx: 8192,
      slots: 1,
      draft: true,
    });
    const i = argv.indexOf("--spec-type");
    expect(argv.slice(i, i + 10)).toEqual([
      "--spec-type",
      "draft-mtp",
      "--spec-draft-n-max",
      "2",
      "-ctkd",
      "q4_0",
      "-ctvd",
      "q4_0",
      "-c",
      "8192",
    ]);
    expect(
      server.argv({ label: "x", pack: "/packs/served.gguf", ctx: 8192, slots: 1 }),
    ).not.toContain("--spec-type");
  });
  test("gate legs ignore enabled live diagnostics and retain prompt-inclusive MTP", async () => {
    const { server } = await setup(headToml.replace("enabled = false", "enabled = true"));
    const argv = server.argv({
      label: "x",
      pack: "/packs/served.gguf",
      ctx: 8192,
      slots: 1,
      draft: true,
    });
    for (const flag of [
      "--lens-layers",
      "--lens-out",
      "--lens-top",
      "--lens-channels",
      "--pull-layers",
      "--pull-action",
      "-lv",
      "--spec-draft-mtp-decode-only",
      "--spec-draft-mtp-window",
    ])
      expect(argv).not.toContain(flag);
  });
  test("draft: true with a sidecar draft names its file, and the draft must be on disk", async () => {
    const { p, head, server } = await setup(withSidecarDraft(headToml));
    const argv = server.argv({
      label: "x",
      pack: "/packs/served.gguf",
      ctx: 8192,
      slots: 1,
      draft: true,
    });
    const i = argv.indexOf("--spec-type");
    expect(argv.slice(i, i + 10)).toEqual([
      "--spec-type",
      "draft-dflash",
      "-md",
      "/r/local/packs/bonsai-2-27b/Bonsai-2-27B-DFlash2-Q8_0.gguf",
      "--spec-draft-n-max",
      "3",
      "-ngld",
      "999",
      "-c",
      "8192",
    ]);
    p.http.json(/\/health$/, { status: "ok" });
    await expect(
      server.leg({ label: "d", pack: "/p.gguf", ctx: 8192, slots: 1, draft: true }, async () => 0),
    ).rejects.toThrow(
      "missing at /r/local/packs/bonsai-2-27b/Bonsai-2-27B-DFlash2-Q8_0.gguf (run: rig fetch)",
    );
    p.fs.put(head.draftPath!, "draft");
    expect(
      await server.leg(
        { label: "d", pack: "/p.gguf", ctx: 8192, slots: 1, draft: true },
        async () => 1,
      ),
    ).toBe(1);
  });
  test("start waits for health, logs to the run directory, and the leg stops the server whatever happens", async () => {
    const { p, server } = await setup();
    let polls = 0;
    p.http.on(/\/health$/, () => {
      if (polls++ < 3) throw connectionRefused();
      return { status: 200, text: "" };
    });
    const value = await server.leg(
      { label: "one", pack: "/p.gguf", ctx: 8192, slots: 1 },
      async (c) => {
        expect(await c.healthy()).toBe(true);
        return 42;
      },
    );
    expect(value).toBe(42);
    expect(p.shell.spawned[0]!.opts).toMatchObject({
      env: {
        CUDA_DEVICE_ORDER: "PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES: "1",
        LD_LIBRARY_PATH: "/r/local/engine-builds/60feea0-sm120",
      },
      stdoutPath: "/r/local/gate-runs/run/server-one.log",
    });
    await expect(
      server.leg({ label: "two", pack: "/p.gguf", ctx: 8192, slots: 1 }, async () => {
        throw new Error("probe blew up");
      }),
    ).rejects.toThrow("probe blew up");
    // both legs stopped: a third can start
    await server.leg({ label: "three", pack: "/p.gguf", ctx: 8192, slots: 1 }, async () => 0);
    expect(p.shell.spawned.map((s) => s.cmd[s.cmd.length - 1])).toEqual(["8098", "8098", "8098"]);
  });
  test("a gate leg volunteers to be the memory killer's first victim", async () => {
    const { p, server } = await setup();
    p.http.json(/\/health$/, { status: "ok" });
    await server.leg({ label: "one", pack: "/p.gguf", ctx: 8192, slots: 1 }, async () => 0);
    // raising is unprivileged and this process is the most disposable thing on the box: a gate
    // leg must be picked before someone's serving head or a paid run (scar 2026-09-20 03:28:38)
    expect(p.fs.text("/proc/4243/oom_score_adj")).toBe("800\n"); // FakeShell numbers its children from 4243
  });
  test("a server that never answers is given up on after 300 s, with the log named", async () => {
    const { p, server } = await setup();
    p.http.on(/\/health$/, () => {
      throw connectionRefused();
    });
    await expect(
      server.start({ label: "dead", pack: "/p.gguf", ctx: 8192, slots: 1 }),
    ).rejects.toThrow("did not come up in 300 s — /r/local/gate-runs/run/server-dead.log");
    expect(p.clock.slept.filter((ms) => ms === 1000).length).toBe(301); // 300 polls, then stop()'s wait
  });
});
