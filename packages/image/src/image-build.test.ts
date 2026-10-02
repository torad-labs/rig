import { describe, expect, test } from "bun:test";
import { ExitCode, layoutAt } from "@rig/core";
import type { Engine } from "@rig/engine";
import type { Head } from "@rig/head";
import { type FakePorts, fakePorts, PULL_KEY, REGISTRY, serveRegistry } from "@rig/testing";
import { IMAGES, renderDockerfile } from "./dockerfile.ts";
import { BuildImage, type BuildImageOptions, SMOKE_MODEL } from "./image-build.service.ts";
import { registryCredentials } from "./registry-config.ts";

const layout = layoutAt("/r");
const SHA = "a786bcdc42f79dbb3a513def3eb97cfe5a155e5c";
const engine = {
  fork: { sha: SHA },
  sha7: "a786bcd",
  supports: (cap: string) => cap === "120",
  prebuiltFor: () => undefined,
  binDir: (cap: string) => `/r/local/engine-builds/a786bcd-sm${cap}`,
} as unknown as Engine;
const TARBALL = "/r/local/prebuilt/engine-builds/engine-sm120-a786bcd.tar.gz";
const CONTEXT = "/r/local/images/glm-sm120/context";
const PACKAGES = ["aria2", "ca-certificates", "curl", "xz-utils"];
const RULES = `exclude = ["heads/*/evidence/lens-legs-*/**"]
deny = ["/home/op"]
deny_sha256 = []
allow_binary_sha256 = []
[[rewrite]]
from = "/home/op/lens"
to = "<lens-dir>"
`;
/** a head with a private [derive] asset: the adapter this machine keeps, never an image */
const head = {
  name: "glm",
  derive: [
    { kind: "pq2-lattice-ablation", lora: "assets/lora/adapter.safetensors", lora_sha256: "0" },
  ],
} as unknown as Head;

/** a checkout at commit abc1234 with a clean tree, the pin's tarball, gitleaks, and a card that proves every image */
function machine(): FakePorts {
  const p = fakePorts();
  p.git.heads.set("/r", "abc1234def");
  p.fs.put("/r/public-export.toml", RULES);
  for (const [path, text] of Object.entries({
    "package.json": "{}",
    "bun.lock": "{}",
    "apps/cli/src/main.ts": "main",
    "engine/engine.toml": "[fork]",
    "heads/glm/head.toml": 'args = ["--lens-out", "/home/op/lens/x"]',
    "heads/glm/assets/lora/adapter.safetensors": "private weights",
    "heads/glm/evidence/lens-legs-1/leg.json": "{}",
  }))
    p.git.committed.set(path, text);
  p.fs.files.set(TARBALL, Bun.gzipSync(new TextEncoder().encode("engine bytes")));
  p.fs.put(`/r/local/downloads/${SMOKE_MODEL.file}`, "smoke");
  p.shell.tools.add("gitleaks");
  p.shell.on(/^gitleaks/, { code: 0, stdout: "", stderr: "" });
  // bun bundles what the exported workspaces hold into the one binary
  p.shell.tools.add("bun");
  p.shell.on(/^bun install/, { code: 0, stdout: "", stderr: "" });
  p.shell.on(/^bun build/, (cmd, opts) => {
    const src = [...p.fs.files].filter(([path]) =>
      ["apps", "packages"].some((dir) => path.startsWith(`${opts?.cwd}/${dir}/`)),
    );
    const bundled = src.map(([, bytes]) => new TextDecoder().decode(bytes)).join("\n");
    p.fs.put(cmd[cmd.indexOf("--outfile") + 1]!, `\u007fELF compiled rig\n${bundled}`);
    return { code: 0, stdout: "", stderr: "" };
  });
  p.containers
    .on(/^rig prepare/, { code: 0, stdout: '{"built": true}', stderr: "" })
    .on(/^rig build/, { code: 0, stdout: '{"alreadyBuilt": true}', stderr: "" })
    .on(/--list-devices/, {
      code: 0,
      stdout: "Available devices:\n  CUDA0: NVIDIA RTX\n",
      stderr: "",
    })
    .on(/-m \/smoke.gguf/, { code: 0, stdout: "", stderr: "" });
  p.fs.put(
    "/r/registry.toml",
    `[registry]\nhost = "${REGISTRY.host}"\nrepository = "rig"\nbucket = "${REGISTRY.bucket}"\nendpoint = "${REGISTRY.endpoint}"\n`,
  );
  // docker save's layout, one layer, for a push to read
  p.shell.on(/^tar -xf/, (cmd) => {
    const into = cmd[cmd.indexOf("-C") + 1]!;
    const blob = (content: string) => {
      const digest = `sha256:${new Bun.CryptoHasher("sha256").update(content).digest("hex")}`;
      p.fs.put(`${into}/blobs/sha256/${digest.slice(7)}`, content);
      return { digest, size: content.length };
    };
    const manifest = JSON.stringify({
      config: { mediaType: "c", ...blob("{}") },
      layers: [{ mediaType: "application/vnd.oci.image.layer.v1.tar", ...blob("layer") }],
    });
    p.fs.put(`${into}/index.json`, JSON.stringify({ manifests: [blob(manifest)] }));
    return { code: 0, stdout: "", stderr: "" };
  });
  serveRegistry(p);
  return p;
}
const CREDENTIALS = async () => ({
  accessKeyId: "id",
  secretAccessKey: "secret",
  pullKey: PULL_KEY,
});
const build = (p: FakePorts, options: Partial<BuildImageOptions> = {}) =>
  new BuildImage(p, layout, engine, PACKAGES, CREDENTIALS).run(head, {
    gpu: 0,
    push: false,
    ...options,
  });
const pushedTags = (p: FakePorts) =>
  [...(p.objectStores.buckets.get(REGISTRY.bucket)?.keys() ?? [])].filter((key) =>
    key.includes("/tags/"),
  );
const text = (p: FakePorts, path: string) => new TextDecoder().decode(p.fs.files.get(path));

describe("rig image", () => {
  test("the context is the CLI, and the head and the pin as the public repo publishes them, with the pin's engine for the card's sm", async () => {
    const p = machine();
    const r = await build(p);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value.image).toMatch(/^registry\.example\/rig:glm-sm120-abc1234-[0-9a-f]{8}$/);
    expect(r.value).toMatchObject({
      head: "glm",
      cap: "120",
      engine: SHA,
      pushed: false,
      digest: null,
    });
    const files = [...p.fs.files.keys()].filter((path) => path.startsWith(`${CONTEXT}/`)).sort();
    expect(files.map((path) => path.slice(CONTEXT.length + 1))).toEqual([
      "Dockerfile",
      "engine/engine-sm120-a786bcd.tar.gz",
      "rig/dist/rig",
      "rig/engine/engine.toml",
      "rig/heads/glm/head.toml",
    ]);
    expect(text(p, `${CONTEXT}/rig/heads/glm/head.toml`)).toBe(
      'args = ["--lens-out", "<lens-dir>/x"]',
    );
    // the CLI is compiled from the committed tree beside the context, which is gone after
    expect(p.git.exported.map((each) => each.paths)).toEqual([
      ["engine/engine.toml", "heads/glm"],
      ["package.json", "bun.lock", "bunfig.toml", "apps", "packages", "tools"],
    ]);
    expect(p.git.exported[1]!.into).toBe("/r/local/images/glm-sm120/cli-source");
    expect(await p.fs.exists("/r/local/images/glm-sm120/cli-source")).toBe(false);
    expect(p.containers.builds).toHaveLength(1);
    expect(pushedTags(p)).toEqual([]);
  });

  test("it is proven on the card before anything else: prepare needs nothing, build finds the sm's engine, it decodes on CUDA", async () => {
    const p = machine();
    p.gpu.card(1);
    await build(p, { gpu: 1 });
    expect(p.containers.runs.map((run) => run.cmd.slice(0, 2).join(" "))).toEqual([
      "rig prepare",
      "rig build",
      "/opt/rig/local/engine-builds/a786bcd-sm120/llama-bench --list-devices",
      "/opt/rig/local/engine-builds/a786bcd-sm120/llama-bench -m",
    ]);
    expect(p.containers.runs.every((run) => run.options?.gpu === 1)).toBe(true);
    expect(p.containers.runs[3]!.options?.mounts).toEqual({
      [`/r/local/downloads/${SMOKE_MODEL.file}`]: "/smoke.gguf",
    });
  });

  test("an image that does not prove itself is not pushed: the engine missing, or a decode on the CPU", async () => {
    const p = machine();
    p.containers.on(/^rig prepare/, { code: 0, stdout: '{"built": false}', stderr: "" });
    const r = await build(p, { push: true });
    expect(!r.ok && r.message).toContain("does not find the engine installed");
    expect(pushedTags(p)).toEqual([]);
    const cpu = machine();
    cpu.containers.on(/--list-devices/, { code: 0, stdout: "Available devices:\n", stderr: "" });
    const onCpu = await build(cpu, { push: true });
    expect(!onCpu.ok && onCpu.message).toContain("sees no CUDA device");
    expect(pushedTags(cpu)).toEqual([]);
  });

  test("--push writes it into registry.toml's bucket, served back, and records the digest beside the context", async () => {
    const p = machine();
    const r = await build(p, { push: true });
    expect(r.ok && r.value.pushed).toBe(true);
    expect(r.ok && r.value.digest).toMatch(/^registry\.example\/rig@sha256:[0-9a-f]{64}$/);
    expect(pushedTags(p)).toEqual([`rig/tags/${r.ok ? r.value.image.split(":")[1] : ""}`]);
    expect(p.objectStores.opened[0]).toMatchObject({ bucket: "rig-images", accessKeyId: "id" });
    const recorded = JSON.parse(text(p, "/r/local/images/glm-sm120/image.json"));
    expect(recorded).toEqual(r.ok ? r.value : null);
  });

  test("the credentials are read for a push only, the environment's before the keyring's", async () => {
    const p = machine();
    let reads = 0;
    const counted = async () => {
      reads++;
      return CREDENTIALS();
    };
    await new BuildImage(p, layout, engine, PACKAGES, counted).run(head, { gpu: 0, push: false });
    expect(reads).toBe(0);
    await new BuildImage(p, layout, engine, PACKAGES, counted).run(head, { gpu: 0, push: true });
    expect(reads).toBe(1);

    p.secrets.held.set("rig-registry/r2-access-key-id", "keyring-id");
    p.secrets.held.set("rig-registry/r2-secret-access-key", "keyring-secret");
    p.secrets.held.set("rig-registry/pull", "keyring-pull");
    expect(await registryCredentials(p.secrets, { RIG_REGISTRY_PULL_KEY: "env-pull" })).toEqual({
      accessKeyId: "keyring-id",
      secretAccessKey: "keyring-secret",
      pullKey: "env-pull",
    });
    p.secrets.held.clear();
    expect(await registryCredentials(p.secrets, { RIG_R2_ACCESS_KEY_ID: "" })).toEqual({
      accessKeyId: undefined,
      secretAccessKey: undefined,
      pullKey: undefined,
    });
  });

  test("--push without its credentials or a registry refuses before anything is built", async () => {
    const p = machine();
    const bare = await new BuildImage(p, layout, engine, PACKAGES, async () => ({})).run(head, {
      gpu: 0,
      push: true,
    });
    expect(!bare.ok && bare.message).toContain("RIG_R2_ACCESS_KEY_ID");
    p.fs.files.delete("/r/registry.toml");
    const nowhere = await build(p, { push: true });
    expect(!nowhere.ok && nowhere.message).toContain("no registry.toml");
    expect(p.containers.builds).toEqual([]);
  });

  test("a deny pattern no rewrite covers refuses by path and line, and nothing is built", async () => {
    const p = machine();
    p.git.committed.set("heads/glm/evidence.md", "ok\nran on /home/op/box\n");
    const r = await build(p);
    expect(!r.ok && r.message).toContain("heads/glm/evidence.md:2: /home/op");
    expect(p.containers.builds).toEqual([]);
  });

  test("a private path compiled into the CLI refuses, read from the binary's bytes", async () => {
    const p = machine();
    p.git.committed.set("packages/core/src/leak.ts", "const dir = '/home/op/secret';");
    const r = await build(p);
    expect(!r.ok && r.message).toContain("dist/rig: binary holds /home/op");
    expect(p.containers.builds).toEqual([]);
  });

  test("an engine whose bytes hold a private path refuses, read through its gzip", async () => {
    const p = machine();
    p.fs.files.set(
      TARBALL,
      Bun.gzipSync(new TextEncoder().encode("assert at /home/op/llama.cpp/ggml.c")),
    );
    const r = await build(p);
    expect(!r.ok && r.message).toContain("engine-sm120-a786bcd.tar.gz: binary holds /home/op");
    expect(p.containers.builds).toEqual([]);
  });

  test("gitleaks gates it: a finding refuses by file and rule, and a machine without gitleaks builds nothing", async () => {
    const p = machine();
    p.shell.on(/^gitleaks/, (cmd) => {
      const report = cmd[cmd.indexOf("--report-path") + 1]!;
      p.fs.put(
        report,
        JSON.stringify([
          { File: `${CONTEXT}/rig/heads/glm/head.toml`, StartLine: 3, RuleID: "generic-api-key" },
        ]),
      );
      return { code: 1, stdout: "", stderr: "leaks found" };
    });
    const r = await build(p);
    expect(!r.ok && r.message).toContain("heads/glm/head.toml:3: gitleaks generic-api-key");
    const bare = machine();
    bare.shell.tools.delete("gitleaks");
    const none = await build(bare);
    expect(!none.ok && none.message).toContain("gitleaks is not on this machine");
    expect(bare.containers.builds).toEqual([]);
  });

  test("with no build of the pin for the sm the refusal names how to make one; a published prebuilt bakes none", async () => {
    const p = machine();
    p.fs.files.delete(TARBALL);
    const r = await build(p);
    expect(!r.ok && r.message).toContain(`rig build --prebuilt --sha ${SHA}`);
    const published = { ...engine, prebuiltFor: () => ({ cap: "120" }) } as unknown as Engine;
    const q = machine();
    q.fs.files.delete(TARBALL);
    const r2 = await new BuildImage(q, layout, published, PACKAGES, CREDENTIALS).run(head, {
      gpu: 0,
      push: false,
    });
    expect(r2.ok).toBe(true);
    const dockerfile = text(q, `${CONTEXT}/Dockerfile`);
    expect(dockerfile).not.toContain("--from-tarball");
    expect(dockerfile).toContain("rig build glm --cap 120 \\");
  });

  test("a card the engine is not measured on is refused before anything is staged", async () => {
    const p = machine();
    p.gpu.card(0, { computeCap: "89", name: "NVIDIA L4" });
    const r = await build(p);
    expect(!r.ok && r.code).toBe(ExitCode.Unsupported);
    expect(p.git.exported).toEqual([]);
  });

  test("a changed input is a new tag", async () => {
    const p = machine();
    const first = await build(p);
    p.git.committed.set("heads/glm/head.toml", "args = []");
    const second = await build(p);
    expect(first.ok && second.ok && first.value.image !== second.value.image).toBe(true);
  });
});

describe("the Dockerfile", () => {
  const dockerfile = renderDockerfile({
    head: "glm",
    cap: "120",
    commit: "abc1234",
    engineTarball: "engine-sm120-a786bcd.tar.gz",
    packages: PACKAGES,
  });
  test("every base image is pinned by digest", () => {
    const froms = dockerfile.split("\n").filter((line) => line.startsWith("FROM "));
    expect(froms).toHaveLength(2);
    for (const line of froms) expect(line).toMatch(/@sha256:[0-9a-f]{64}( AS \w+)?$/);
    expect(Object.values(IMAGES).every((image) => dockerfile.includes(image))).toBe(true);
  });
  test("the engine is installed by rig build from the tarball, which with the compat libcuda is bound into that step, never a layer", () => {
    expect(dockerfile).toContain(
      "/opt/rig/dist/rig build glm --cap 120 --from-tarball /tmp/rig-engine/engine-sm120-a786bcd.tar.gz",
    );
    expect(dockerfile).toContain(
      "RUN --mount=type=bind,from=compat,source=/usr/local/cuda/compat,target=/tmp/rig-compat --mount=type=bind,source=engine,target=/tmp/rig-engine ",
    );
    // a COPY of either would be a layer every pull carries, whatever a later step deletes
    expect(dockerfile.split("\n").filter((line) => line.startsWith("COPY"))).toEqual([
      "COPY rig/ /opt/rig/",
    ]);
    expect(dockerfile).toContain(
      "rm -rf /etc/ld.so.conf.d/zz-rig-build.conf /opt/rig/local/downloads",
    );
    expect(dockerfile).toContain("aria2 ca-certificates curl xz-utils openssh-server");
    expect(dockerfile).toContain('CMD ["rig", "up", "glm", "--foreground"]');
    // the CLI arrives compiled: no source, no bun, no build of rig in the image
    expect(dockerfile).toContain("COPY rig/ /opt/rig/");
    expect(dockerfile).not.toMatch(/source\/|\bbun\b/);
  });
});
