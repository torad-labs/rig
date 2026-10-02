import { describe, expect, test } from "bun:test";
import type { Instance } from "@rig/core";
import { layoutAt } from "@rig/core";
import { fakePorts, REGISTRY, repoRoot } from "@rig/testing";
import { SweepStopped } from "./stopped-sweep.ts";

const HOUR = 3_600_000;
const RIG_IMAGE = `${REGISTRY.host}/rig:glm-5.3-flash-sm120-abc1234-0e11d35a`;

async function setup(boxes: Instance[]) {
  const p = fakePorts();
  p.fs.put("/r/vast.toml", await Bun.file(`${repoRoot}/vast.toml`).text());
  // a fixture, not the repo's registry.toml: that file is the operator's and is not in the public copy
  p.fs.put(
    "/r/registry.toml",
    `[registry]\nhost = "${REGISTRY.host}"\nrepository = "rig"\nbucket = "${REGISTRY.bucket}"\nendpoint = "${REGISTRY.endpoint}"\n`,
  );
  for (const box of boxes) p.rental.instances.set(box.id, box);
  const sweep = new SweepStopped({ ...p, self: ["/r/dist/rig"] }, layoutAt("/r"));
  return { p, sweep };
}

const box = (o: Partial<Instance> & { id: number }): Instance => ({
  status: "exited",
  label: "",
  dph: 0.074,
  ...o,
});

describe("vast sweep", () => {
  test("rig's stopped boxes are destroyed after stopped_hours, counted from the first sweep that saw them stopped", async () => {
    const { p, sweep } = await setup([
      box({ id: 1, label: "rig" }), // rented by `vast up`, its state since cleared
      box({ id: 2, image: RIG_IMAGE }), // rented from the template in the console: no label
      box({ id: 3, label: "rig", status: "running" }), // its own guard watches it
    ]);
    const first = await sweep.run();
    expect(first.ok && first.value).toEqual({
      destroyed: [],
      stopped: [
        { id: 1, status: "exited", hours: 0 },
        { id: 2, status: "exited", hours: 0 },
      ],
    });
    p.clock.t += 23 * HOUR;
    const kept = await sweep.run();
    expect(kept.ok && kept.value.stopped.map((each) => each.hours)).toEqual([23, 23]);
    expect(p.rental.ops.filter((op) => op.startsWith("destroy"))).toEqual([]);
    p.clock.t += HOUR;
    const swept = await sweep.run();
    expect(swept.ok && swept.value).toEqual({ destroyed: [1, 2], stopped: [] });
    expect([...p.rental.instances.keys()]).toEqual([3]);
    expect(JSON.parse(p.fs.text("/r/local/rented-box/stopped.json") ?? "")).toEqual({});
  });

  test("a box kept on purpose, the box `vast up` tracks, and a box started again are never destroyed", async () => {
    const { p, sweep } = await setup([
      box({ id: 52663093, label: "rig-hostgaps" }), // local/box-idle's kept 5080: stopped, never destroyed
      box({ id: 40000001, label: "another-project", image: "pytorch/pytorch:2.4.1" }), // another project's box on the account
      box({ id: 9, image: RIG_IMAGE, label: "mine" }), // a template box someone labelled to keep
      box({ id: 7, label: "rig" }), // the box `vast up` rented: its own idle timer owns it
      box({ id: 8, label: "rig" }),
    ]);
    p.fs.put("/r/local/rented-box/instance.json", JSON.stringify({ instanceId: 7 }));
    await sweep.run();
    p.clock.t += 20 * HOUR;
    p.rental.instances.set(8, box({ id: 8, label: "rig", status: "running" })); // started again
    await sweep.run();
    p.rental.instances.set(8, box({ id: 8, label: "rig" })); // stopped again: counted from now
    p.clock.t += 10 * HOUR;
    const r = await sweep.run();
    expect(r.ok && r.value).toEqual({
      destroyed: [],
      stopped: [{ id: 8, status: "exited", hours: 0 }],
    });
    expect(p.rental.ops.filter((op) => op.startsWith("destroy"))).toEqual([]);
  });

  test("a destroy the listing contradicts fails the sweep and is asked again; a market that cannot be read sweeps nothing", async () => {
    const { p, sweep } = await setup([box({ id: 1, label: "rig" })]);
    await sweep.run();
    p.clock.t += 24 * HOUR;
    const destroy = p.rental.destroy.bind(p.rental);
    p.rental.destroy = async (id: number) => {
      p.rental.ops.push(`destroy ${id}`); // vast answers, and the box stays listed
    };
    const refused = await sweep.run();
    expect(!refused.ok && refused.message).toContain("box 1 STILL listed after destroy");
    expect(Object.keys(JSON.parse(p.fs.text("/r/local/rented-box/stopped.json") ?? ""))).toEqual([
      "1",
    ]);
    p.rental.destroy = destroy;
    const again = await sweep.run();
    expect(again.ok && again.value.destroyed).toEqual([1]);

    p.rental.list = async () => {
      throw new Error("429 Too Many Requests");
    };
    const unread = await sweep.run();
    expect(!unread.ok && unread.message).toContain("vast could not be read: 429 Too Many Requests");
  });

  test("arm installs the hourly timer and its service, running this rig's `vast sweep`", async () => {
    const { p, sweep } = await setup([]);
    await sweep.arm();
    const dir = p.systemd.unitDir();
    expect(p.fs.text(`${dir}/rig-vast-sweep.service`)).toContain(
      "ExecStart=/r/dist/rig vast sweep",
    );
    expect(p.fs.text(`${dir}/rig-vast-sweep.timer`)).toContain("OnCalendar=hourly");
    expect(p.systemd.ops).toContain("enable rig-vast-sweep.timer");
  });
});
