import { describe, expect, test } from "bun:test";
import worker, { type Bucket, type StoredObject } from "./worker.ts";

const KEY = "pull-key-for-tests";
const MANIFEST = '{"schemaVersion":2}';
const DIGEST = `sha256:${"a".repeat(64)}`;
const BLOB = `sha256:${"b".repeat(64)}`;
const MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json";

/** an R2 bucket as a map, each object's content type beside it */
function bucket(objects: Record<string, [string, string?]>): Bucket {
  const at = (key: string) => objects[key];
  return {
    async head(key) {
      const found = at(key);
      return found ? { size: found[0].length, httpMetadata: { contentType: found[1] } } : null;
    },
    async get(key) {
      const found = at(key);
      if (!found) return null;
      const object: StoredObject = {
        size: found[0].length,
        httpMetadata: { contentType: found[1] },
        body: new Response(found[0]).body!,
        text: async () => found[0],
      };
      return object;
    },
  };
}
const env = {
  PULL_KEY: KEY,
  IMAGES: bucket({
    "rig/tags/glm-sm120": [DIGEST],
    [`rig/manifests/sha256/${"a".repeat(64)}`]: [MANIFEST, MANIFEST_TYPE],
    [`rig/blobs/sha256/${"b".repeat(64)}`]: ["layer bytes"],
  }),
};
const basic = (user: string, password: string) => `Basic ${btoa(`${user}:${password}`)}`;
const pull = (path: string, init: RequestInit = {}, auth = basic("rig", KEY)) =>
  worker.fetch(
    new Request(`https://registry.example${path}`, {
      ...init,
      headers: { authorization: auth, ...(init.headers ?? {}) },
    }),
    env,
  );

describe("the registry", () => {
  test("a pull without the key is challenged, and a wrong key is refused", async () => {
    const none = await worker.fetch(new Request("https://registry.example/v2/"), env);
    expect(none.status).toBe(401);
    expect(none.headers.get("www-authenticate")).toBe('Basic realm="rig"');
    expect((await pull("/v2/", {}, basic("rig", "wrong"))).status).toBe(401);
    expect((await pull("/v2/", {}, basic("rig", `${KEY}x`))).status).toBe(401);
    expect((await pull("/v2/", {}, `Bearer ${KEY}`)).status).toBe(401);
  });
  test("with the key: the API answers, a tag resolves to its manifest by digest and type", async () => {
    const root = await pull("/v2/");
    expect(root.status).toBe(200);
    expect(root.headers.get("docker-distribution-api-version")).toBe("registry/2.0");
    const byTag = await pull("/v2/rig/manifests/glm-sm120");
    expect(byTag.status).toBe(200);
    expect(await byTag.text()).toBe(MANIFEST);
    expect(byTag.headers.get("content-type")).toBe(MANIFEST_TYPE);
    expect(byTag.headers.get("docker-content-digest")).toBe(DIGEST);
    const byDigest = await pull(`/v2/rig/manifests/${DIGEST}`);
    expect(await byDigest.text()).toBe(MANIFEST);
  });
  test("a blob streams by digest; HEAD gives its size and no body", async () => {
    const blob = await pull(`/v2/rig/blobs/${BLOB}`);
    expect(await blob.text()).toBe("layer bytes");
    expect(blob.headers.get("content-length")).toBe("11");
    const head = await pull(`/v2/rig/blobs/${BLOB}`, { method: "HEAD" });
    expect(head.status).toBe(200);
    expect(head.headers.get("content-length")).toBe("11");
    expect(await head.text()).toBe("");
  });
  test("what is not there is 404 by the spec's code, and a name or a tag cannot reach another key", async () => {
    const missing = await pull(`/v2/rig/blobs/sha256:${"c".repeat(64)}`);
    expect(missing.status).toBe(404);
    expect(((await missing.json()) as { errors: { code: string }[] }).errors[0]!.code).toBe(
      "BLOB_UNKNOWN",
    );
    expect((await pull("/v2/rig/manifests/nope")).status).toBe(404);
    expect((await pull("/v2/../rig/manifests/glm-sm120")).status).toBe(404);
    expect((await pull("/v2/rig/manifests/..%2Ftags%2Fglm-sm120")).status).toBe(404);
    expect((await pull("/v2/Rig/blobs/x")).status).toBe(404);
  });
  test("it serves pulls only: a push is refused even with the key", async () => {
    const push = await pull("/v2/rig/blobs/uploads/", { method: "POST" });
    expect(push.status).toBe(405);
    expect((await pull(`/v2/rig/manifests/glm-sm120`, { method: "PUT", body: "{}" })).status).toBe(
      405,
    );
  });
});
