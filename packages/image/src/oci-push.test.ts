import { describe, expect, test } from "bun:test";
import { type FakePorts, fakePorts, PULL_KEY, REGISTRY, serveRegistry } from "@rig/testing";
import { MANIFEST_TYPE, pushImage } from "./oci-push.ts";

const WORK = "/r/local/images/glm-sm120/push";
const sha = (bytes: Uint8Array | string) =>
  `sha256:${new Bun.CryptoHasher("sha256").update(bytes).digest("hex")}`;
const LAYERS = ["the base layer", "the engine layer"];
const CONFIG = '{"rootfs":{"diff_ids":[]}}';

/** docker save's OCI layout, unpacked by `tar -xf` into the directory it is given */
function saves(p: FakePorts, layers = LAYERS) {
  p.shell.on(/^tar -xf/, (cmd) => {
    const into = cmd[cmd.indexOf("-C") + 1]!;
    const blob = (content: string) => {
      const digest = sha(content);
      p.fs.put(`${into}/blobs/sha256/${digest.slice(7)}`, content);
      return { digest, size: content.length };
    };
    const manifest = JSON.stringify({
      config: { mediaType: "application/vnd.oci.image.config.v1+json", ...blob(CONFIG) },
      layers: layers.map((layer) => ({
        mediaType: "application/vnd.oci.image.layer.v1.tar",
        ...blob(layer),
      })),
    });
    const described = blob(manifest);
    p.fs.put(
      `${into}/index.json`,
      JSON.stringify({ manifests: [{ mediaType: "x", ...described }] }),
    );
    return { code: 0, stdout: "", stderr: "" };
  });
}
function machine() {
  const p = fakePorts();
  saves(p);
  serveRegistry(p);
  return p;
}
const store = (p: FakePorts) =>
  p.objectStores.open({ ...REGISTRY, accessKeyId: "id", secretAccessKey: "secret" });
const push = (p: FakePorts) =>
  pushImage(
    p,
    store(p),
    REGISTRY,
    PULL_KEY,
    "registry.example/rig:glm-sm120-x",
    "glm-sm120-x",
    WORK,
  );
const objects = (p: FakePorts) => p.objectStores.buckets.get("rig-images")!;

describe("pushImage", () => {
  test("the saved image lands as the registry serves it: gzipped layers, a schema-2 manifest, the tag naming it", async () => {
    const p = machine();
    const r = await push(p);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const tag = new TextDecoder().decode(objects(p).get("rig/tags/glm-sm120-x")!.bytes);
    expect(tag).toBe(r.value);
    const stored = objects(p).get(`rig/manifests/sha256/${r.value.slice(7)}`)!;
    expect(stored.type).toBe(MANIFEST_TYPE);
    const manifest = JSON.parse(new TextDecoder().decode(stored.bytes));
    expect(manifest.mediaType).toBe(MANIFEST_TYPE);
    expect(manifest.config.digest).toBe(sha(CONFIG));
    for (const [i, layer] of manifest.layers.entries()) {
      const blob = objects(p).get(`rig/blobs/sha256/${layer.digest.slice(7)}`)!.bytes;
      expect(layer.mediaType).toBe("application/vnd.docker.image.rootfs.diff.tar.gzip");
      expect(layer.size).toBe(blob.byteLength);
      expect(new TextDecoder().decode(Bun.gunzipSync(new Uint8Array(blob)))).toBe(LAYERS[i]!);
    }
    expect(await p.fs.exists(WORK)).toBe(false);
  });

  test("blobs go up before the manifest, and the tag last; a blob already there is not sent again", async () => {
    const p = machine();
    const order: string[] = [];
    const s = store(p);
    const logged = {
      exists: s.exists,
      put: async (key: string, from: Parameters<typeof s.put>[1], type?: string) => {
        order.push(key.split("/")[1]!);
        return s.put(key, from, type);
      },
    };
    await pushImage(p, logged, REGISTRY, PULL_KEY, "i", "glm-sm120-x", WORK);
    expect(order).toEqual(["blobs", "blobs", "blobs", "manifests", "tags"]);
    order.length = 0;
    await pushImage(p, logged, REGISTRY, PULL_KEY, "i", "glm-sm120-x", WORK);
    expect(order).toEqual(["manifests", "tags"]);
  });

  test("a bucket that refuses a write is a named failure, not a thrown error, and no tag is written", async () => {
    // what the real endpoint did to a push with credentials it did not know (S3Error "UnknownError", thrown
    // out of the CLI as a stack trace with no mention of the bucket): the rehearsal of 2026-10-02
    const p = machine();
    const s = store(p);
    const refusing = {
      exists: s.exists,
      put: async (key: string) => {
        throw Object.assign(new Error("an unexpected error has occurred"), {
          name: "S3Error",
          code: "UnknownError",
          path: key,
        });
      },
    };
    const r = await pushImage(p, refusing, REGISTRY, PULL_KEY, "i", "glm-sm120-x", WORK);
    expect(r.ok).toBe(false);
    expect(!r.ok && r.message).toContain("rig-images");
    expect(!r.ok && r.message).toContain("S3Error");
    expect(!r.ok && r.message).toContain("UnknownError");
    expect(!r.ok && r.message).toContain("rig/blobs/sha256/");
    expect([...objects(p).keys()]).toEqual([]);
    expect(await p.fs.exists(WORK)).toBe(false);
  });

  test("a registry that does not serve back what was written fails the push, by what it served", async () => {
    const p = machine();
    p.http.on(/manifests/, () => ({ status: 401, text: "", headers: {} }));
    const r = await push(p);
    expect(!r.ok && r.message).toContain(
      "does not serve rig:glm-sm120-x (HTTP 401: the pull key is refused)",
    );
    const q = machine();
    const wrongKey = await pushImage(
      q,
      store(q),
      REGISTRY,
      "not-the-key",
      "i",
      "glm-sm120-x",
      WORK,
    );
    expect(!wrongKey.ok && wrongKey.message).toContain("HTTP 401");
  });
});
