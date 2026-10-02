import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { layoutAt } from "@rig/core";
import type { Engine } from "@rig/engine";
import { loadHead } from "@rig/head";
import {
  type FakePorts,
  fakePorts,
  PULL_KEY,
  REGISTRY,
  repoRoot,
  serveRegistry,
} from "@rig/testing";
import { headDiskGb, PublishTemplate, renderOnstart } from "./box-template.ts";

const root = repoRoot;
const layout = layoutAt("/r");
const engine = { cuda: { version: "13.3", runtime: [] } } as unknown as Engine;
const TAG = "glm-5.3-flash-sm120-abc1234-0e11d35a";
const IMAGE = `registry.example/rig:${TAG}`;
const RECORD = "/r/local/images/glm-5.3-flash-sm120";
const DIGEST = `sha256:${"ab".repeat(32)}`;

/** the registry's bucket holding the tag as `digest`, served by the registry's own worker */
function pushedTo(p: FakePorts, digest = DIGEST) {
  const bucket = p.objectStores.buckets.get(REGISTRY.bucket) ?? new Map();
  p.objectStores.buckets.set(REGISTRY.bucket, bucket);
  bucket.set(`rig/tags/${TAG}`, { bytes: new TextEncoder().encode(digest) });
  bucket.set(`rig/manifests/${digest.replace(":", "/")}`, {
    bytes: new TextEncoder().encode("{}"),
    type: "application/vnd.docker.distribution.manifest.v2+json",
  });
  serveRegistry(p);
}

/** the real GLM head, its image pushed, registry.toml naming the registry */
async function machine(pushed = true) {
  const p = fakePorts();
  for (const file of ["head.toml", "assets/chat-template.jinja"])
    p.fs.put(
      `/r/heads/glm-5.3-flash/${file}`,
      await Bun.file(`${root}/heads/glm-5.3-flash/${file}`).text(),
    );
  p.fs.put(
    "/r/registry.toml",
    `[registry]\nhost = "${REGISTRY.host}"\nrepository = "rig"\nbucket = "${REGISTRY.bucket}"\nendpoint = "${REGISTRY.endpoint}"\n`,
  );
  p.fs.put(
    `${RECORD}/image.json`,
    JSON.stringify({ image: IMAGE, digest: `registry.example/rig@${DIGEST}`, cap: "120", pushed }),
  );
  if (pushed) pushedTo(p);
  const head = await loadHead(p.fs, layout, "glm-5.3-flash");
  if (!head.ok) throw new Error(head.message);
  return { p, head: head.value };
}
const publish = (p: FakePorts, pullKey: string | undefined = PULL_KEY) =>
  new PublishTemplate({ ...p, pullKey: async () => pullKey }, layout, engine);

describe("vast template", () => {
  test("a template is saved only when the registry serves its tag, to its pull key, as the image rig pushed", async () => {
    const { p, head } = await machine();
    const refused = await publish(p, "not-the-key").run(head, {
      idleMinutes: 60,
      maxHours: 24,
      dryRun: false,
    });
    expect(!refused.ok && refused.message).toContain("HTTP 401: the pull key is refused");
    pushedTo(p, `sha256:${"cd".repeat(32)}`); // the tag moved to another image
    const moved = await publish(p).run(head, { idleMinutes: 60, maxHours: 24, dryRun: false });
    expect(!moved.ok && moved.message).toContain(
      `not the ${DIGEST} rig image pushed: push it again`,
    );
    expect(p.rental.templates).toEqual([]);
  });

  test("the pushed image, its registry login, the head's disk and its first profile's offers, and up beside the guard", async () => {
    const { p, head } = await machine();
    const r = await publish(p).run(head, { idleMinutes: 60, maxHours: 24, dryRun: false });
    expect(r.ok).toBe(true);
    const { spec, hashId } = p.rental.templates[0]!;
    expect(hashId).toBeUndefined();
    expect(spec).toMatchObject({
      name: "rig-glm-5.3-flash-sm120",
      image: "registry.example/rig",
      tag: "glm-5.3-flash-sm120-abc1234-0e11d35a",
      diskGb: 160, // 134.3 GB of shards, a tenth more, 10 GB beside, to the next 10
      login: { registry: "registry.example", user: "rig", password: PULL_KEY },
      filters: {
        num_gpus: { eq: 2 },
        gpu_ram: { gte: 92000 },
        compute_cap: { eq: 1200 },
        cuda_max_good: { gte: 13.3 },
        disk_space: { gte: 160 },
      },
    });
    expect(spec.onstart.split("\n").slice(1)).toEqual([
      "mkdir -p /var/log/rig && rm -f /var/log/rig/FAILED",
      "nohup sh -c 'until /opt/rig/dist/rig vast guard glm-5.3-flash --idle-minutes 60 --max-hours 24 --stop-when /var/log/rig/FAILED; do sleep 30; done' >> /var/log/rig/guard.log 2>&1 &",
      "nohup sh -c 'n=0; while :; do t=$(date +%s); /opt/rig/dist/rig up glm-5.3-flash --foreground; [ $(($(date +%s) - t)) -ge 900 ] && n=0; n=$((n + 1)); [ $n -ge 5 ] && break; echo \"rig: rig up exited, try $((n + 1)) of 5 in $((30 * n)) s\"; sleep $((30 * n)); done; touch /var/log/rig/FAILED' >> /var/log/rig/up.log 2>&1 &",
    ]);
    expect(spec.onstart.length).toBeLessThan(4048); // vast's limit on the field
    expect(headDiskGb(head)).toBe(160);
  });

  test("a second run edits the same template in place, by the hash the first one recorded", async () => {
    const { p, head } = await machine();
    await publish(p).run(head, { idleMinutes: 60, maxHours: 24, dryRun: false });
    const again = await publish(p).run(head, {
      idleMinutes: 30,
      maxHours: 24,
      diskGb: 200,
      dryRun: false,
    });
    expect(p.rental.templates[1]).toMatchObject({ hashId: "hash-1", spec: { diskGb: 200 } });
    expect(again.ok && again.value.hashId).toBe("hash-2");
    const recorded = JSON.parse(
      new TextDecoder().decode(p.fs.files.get(`${RECORD}/template.json`)),
    );
    expect(recorded).toMatchObject({ id: 77, hashId: "hash-2", image: IMAGE });
  });

  test("a dry run saves nothing and needs no key; a real one without the pull key refuses", async () => {
    const { p, head } = await machine();
    const dry = await publish(p, "").run(head, { idleMinutes: 60, maxHours: 24, dryRun: true });
    expect(dry.ok && dry.value.image).toBe(IMAGE);
    expect(p.rental.templates).toEqual([]);
    const keyless = await publish(p, "").run(head, {
      idleMinutes: 60,
      maxHours: 24,
      dryRun: false,
    });
    expect(!keyless.ok && keyless.message).toContain("RIG_REGISTRY_PULL_KEY");
    expect(p.rental.templates).toEqual([]);
  });

  test("an image built but never pushed is no image to point a template at", async () => {
    const { p, head } = await machine(false);
    const r = await publish(p).run(head, { idleMinutes: 60, maxHours: 24, dryRun: false });
    expect(!r.ok && r.message).toContain("rig image glm-5.3-flash --push");
  });
});

/** the on-start's supervisor of `rig up`, run by sh as the box runs it: rig, date and sleep are stand-ins in a scratch
 *  directory; `runs` lists how long each rig up lasts, in seconds of the fake clock */
async function supervise(runs: number[]) {
  const dir = mkdtempSync(join(tmpdir(), "rig-onstart-"));
  const bin = join(dir, "bin");
  await Bun.write(
    join(bin, "rig"),
    `#!/bin/sh
n=$(cat ${dir}/runs 2>/dev/null || echo 0); echo $((n + 1)) > ${dir}/runs
lasts=$(sed -n "$((n + 1))p" ${dir}/lasts); echo $(( $(cat ${dir}/clock) + \${lasts:-0} )) > ${dir}/clock
exit 1
`,
  );
  await Bun.write(join(bin, "date"), `#!/bin/sh\ncat ${dir}/clock\n`);
  await Bun.write(join(bin, "sleep"), `#!/bin/sh\necho "$1" >> ${dir}/slept\n`);
  writeFileSync(join(dir, "clock"), "1000\n");
  writeFileSync(join(dir, "lasts"), `${runs.join("\n")}\n`);
  for (const tool of ["rig", "date", "sleep"]) Bun.spawnSync(["chmod", "+x", join(bin, tool)]);
  const line = renderOnstart("h", 60, 24).split("\n")[3]!;
  const script = /^nohup sh -c '(.*)' >> /
    .exec(line)![1]!
    .replaceAll("/opt/rig/dist/rig", join(bin, "rig"))
    .replaceAll("/var/log/rig", dir);
  const run = Bun.spawnSync(["sh", "-c", script], { env: { PATH: `${bin}:/usr/bin:/bin` } });
  const read = (name: string) => {
    try {
      return readFileSync(join(dir, name), "utf8");
    } catch {
      return "";
    }
  };
  return {
    code: run.exitCode,
    out: run.stdout.toString(),
    runs: Number(read("runs")),
    slept: read("slept").trim().split("\n").filter(Boolean).map(Number),
    failed:
      read("FAILED") !== "" || Bun.spawnSync(["test", "-e", join(dir, "FAILED")]).exitCode === 0,
  };
}

describe("the on-start's supervisor", () => {
  test("rig up failing at once is tried five times, 30 s more between each, then gives up by writing FAILED", async () => {
    const r = await supervise([5, 5, 5, 5, 5]);
    expect(r).toMatchObject({ code: 0, runs: 5, slept: [30, 60, 90, 120], failed: true });
    expect(r.out).toContain("rig: rig up exited, try 2 of 5 in 30 s");
  });

  test("a run that served for a quarter of an hour starts the count again", async () => {
    // the third run lasts 20 minutes: it served, then died; five quick failures after it are needed to give up
    const r = await supervise([5, 5, 1200, 5, 5, 5, 5]);
    expect(r).toMatchObject({ runs: 7, slept: [30, 60, 30, 60, 90, 120], failed: true });
  });
});
