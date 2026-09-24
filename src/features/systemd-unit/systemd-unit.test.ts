import { describe, expect, test } from "bun:test";
import { fakePorts } from "../../../test/fakes/index.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { ok } from "../../shared/result.ts";
import { ManageUnit, type Planner } from "./systemd-unit.service.ts";
import { cacheRamOf, renderUnit } from "./unit-file.ts";

const headToml = await Bun.file(`${import.meta.dir}/../../../heads/bonsai-2-27b/head.toml`).text();

async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  p.fs.put("/r/heads/bonsai-2-27b/head.toml", headToml);
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  if (!head.ok) throw new Error(head.message);
  const plans: Array<{ gpu: number; cacheRam?: number | undefined }> = [];
  const planner: Planner = {
    plan: async (_h, o) => {
      plans.push(o);
      const cacheRam = o.cacheRam ?? 15704;
      return ok({
        argv: [
          "/r/local/engine-builds/da69dc5-sm120/llama-server",
          "-m",
          "/r/local/packs/x.gguf",
          "--cache-ram",
          String(cacheRam),
          "--chat-template-file",
          "/r/a b.jinja",
        ],
        env: {
          CUDA_VISIBLE_DEVICES: String(o.gpu),
          LD_LIBRARY_PATH: "/r/local/engine-builds/da69dc5-sm120",
        },
        cacheRam,
      });
    },
  };
  const uc = new ManageUnit({ ...p, planner, self: ["/r/dist/rig"] }, layout);
  return { p, head: head.value, uc, plans };
}

describe("unit", () => {
  test("renders the plan as ExecStart with verify as ExecStartPre, quoting what needs it", async () => {
    const { head, uc } = await setup();
    const r = await uc.render(head, { gpu: 1, cacheRam: 8192 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value.text).toContain(
      "ExecStartPre=/r/dist/rig verify bonsai-2-27b --gpu 1 --pack /r/local/packs/x.gguf\n",
    );
    expect(r.value.text).toContain(
      'ExecStart=/r/local/engine-builds/da69dc5-sm120/llama-server -m /r/local/packs/x.gguf --cache-ram 8192 --chat-template-file "/r/a b.jinja"\n',
    );
    expect(r.value.text).toContain(
      "Environment=CUDA_VISIBLE_DEVICES=1\nEnvironment=LD_LIBRARY_PATH=/r/local/engine-builds/da69dc5-sm120\n",
    );
    expect(r.value.text).toContain("StandardOutput=append:/r/local/logs/bonsai-2-27b.log\n");
    expect(r.value.text).toContain("WantedBy=default.target");
    expect(cacheRamOf(r.value.text)).toBe(8192);
  });
  test("install writes, reloads and enables; a re-run with nothing changed is current; a change is backed up with a date", async () => {
    const { p, head, uc } = await setup();
    let r = await uc.install(head, { gpu: 0, cacheRam: 8192 });
    expect(r.ok && r.value.state).toBe("installed");
    expect(p.systemd.ops).toEqual(["daemon-reload", "enable rig-bonsai-2-27b.service"]);
    expect(p.fs.text("/home/u/.config/systemd/user/rig-bonsai-2-27b.service")).toContain(
      "--cache-ram 8192",
    );
    r = await uc.install(head, { gpu: 0, cacheRam: 8192 });
    expect(r.ok && r.value.state).toBe("current");
    expect(p.systemd.ops.length).toBe(2);
    p.clock.t = Date.UTC(2026, 8, 20, 3, 4, 5);
    r = await uc.install(head, { gpu: 1, cacheRam: 8192 });
    expect(r.ok && r.value).toMatchObject({
      state: "updated",
      backup: "/home/u/.config/systemd/user/rig-bonsai-2-27b.service.20260920T030405Z.bak",
    });
    expect(
      p.fs.text("/home/u/.config/systemd/user/rig-bonsai-2-27b.service.20260920T030405Z.bak"),
    ).toContain("CUDA_VISIBLE_DEVICES=0");
  });
  test("without --cache-ram the installed unit's value is kept; only a fresh install takes the planner's rule", async () => {
    const { p, head, uc, plans } = await setup();
    await uc.install(head, { gpu: 0 });
    expect(plans.at(-1)).toEqual({ gpu: 0, cacheRam: undefined }); // fresh: the rule (15704 in this fake)
    await uc.install(head, { gpu: 0, cacheRam: 8192 });
    await uc.install(head, { gpu: 0 });
    expect(plans.at(-1)).toEqual({ gpu: 0, cacheRam: 8192 }); // kept from the unit on disk
    expect(p.log.lines.some((l) => l.startsWith("warn"))).toBe(false);
  });
  test("an installed unit without --cache-ram falls to the box rule out loud, never silently", async () => {
    const { p, head, uc } = await setup();
    p.fs.put(
      "/home/u/.config/systemd/user/rig-bonsai-2-27b.service",
      "[Service]\nExecStart=/r/llama-server -m /r/x.gguf --port 8099\n",
    );
    await uc.install(head, { gpu: 0 });
    expect(p.log.lines.find((l) => l.startsWith("warn"))).toContain(
      "carries no --cache-ram to keep",
    );
  });
  test("uninstall refuses an active unit (exit 2) and otherwise disables, removes and reloads", async () => {
    const { p, head, uc } = await setup();
    await uc.install(head, { gpu: 0 });
    p.systemd.active.add("rig-bonsai-2-27b.service");
    let r = await uc.uninstall(head);
    expect(!r.ok && r.code).toBe(2);
    p.systemd.active.clear();
    r = await uc.uninstall(head);
    expect(r.ok && r.value.removed).toBe(true);
    expect(await p.fs.exists("/home/u/.config/systemd/user/rig-bonsai-2-27b.service")).toBe(false);
    expect(p.systemd.ops.slice(-2)).toEqual(["disable rig-bonsai-2-27b.service", "daemon-reload"]);
  });
  test("the unit shields the server from the memory killer on the running pid, and does not block the start when the grant is absent", async () => {
    const { head, uc } = await setup();
    const r = await uc.render(head, { gpu: 0 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    // a --user unit cannot LOWER oom_score_adj (systemd clamps it silently at the inherited
    // +200), so the pin goes through the choom grant on $MAINPID; "-" keeps a box without the
    // grant starting normally
    expect(r.value.text).toContain(
      "ExecStartPost=-/usr/bin/sudo -n /usr/bin/choom -p $MAINPID -n -800\n",
    );
    expect(r.value.text).not.toContain("OOMScoreAdjust");
  });
  test("status reports the RUNNING oom_score_adj, never a unit property", async () => {
    const { p, head, uc } = await setup();
    await uc.install(head, { gpu: 0 });
    p.systemd.active.add("rig-bonsai-2-27b.service");
    p.systemd.pids.set("rig-bonsai-2-27b.service", 4242);
    p.fs.put("/proc/4242/oom_score_adj", "-800\n");
    expect(await uc.status(head)).toMatchObject({ mainPid: 4242, oomScoreAdj: -800 });
    p.fs.put("/proc/4242/oom_score_adj", "200\n");
    expect(await uc.status(head)).toMatchObject({ oomScoreAdj: 200 });
  });
  test("status reads the file and systemd", async () => {
    const { p, head, uc } = await setup();
    expect(await uc.status(head)).toEqual({
      unit: "rig-bonsai-2-27b.service",
      path: "/home/u/.config/systemd/user/rig-bonsai-2-27b.service",
      installed: false,
      active: false,
      mainPid: null,
    });
    await uc.install(head, { gpu: 0 });
    p.systemd.active.add("rig-bonsai-2-27b.service");
    p.systemd.pids.set("rig-bonsai-2-27b.service", 777);
    expect(await uc.status(head)).toMatchObject({ installed: true, active: true, mainPid: 777 });
  });
  test("renderUnit is a pure function of its inputs", () => {
    const head = { name: "h", title: "H", port: 9 } as Parameters<typeof renderUnit>[0]["head"];
    const a = renderUnit({
      head,
      root: "/r",
      logPath: "/l",
      argv: ["x"],
      env: {},
      gpu: 0,
      self: ["rig"],
    });
    expect(a).toBe(
      renderUnit({ head, root: "/r", logPath: "/l", argv: ["x"], env: {}, gpu: 0, self: ["rig"] }),
    );
    expect(a).toContain("ExecStart=x\n");
    // no -m in argv (this fixture's is bare): ExecStartPre carries no --pack either, old-unit shaped
    expect(a).toContain("ExecStartPre=rig verify h --gpu 0\n");
    expect(a).not.toContain("--pack");
  });
});
