// The registry a push is read back from, in tests: apps/registry/src/worker.ts itself over a FakeObjectStores
// bucket, answering FakeHttp at https://<host>/. What rig writes and what the Worker serves are
// checked against each other, not each against a copy of the other's layout.

import type { RegistryConfig } from "@rig/image";
import worker, { type Bucket } from "@rig/registry";
import type { FakePorts } from "./fakes.ts";

export const REGISTRY: RegistryConfig = {
  host: "registry.example",
  repository: "rig",
  bucket: "rig-images",
  endpoint: "https://account.r2.example",
};
export const PULL_KEY = "pull-key-for-tests";

export function serveRegistry(p: FakePorts, registry = REGISTRY): void {
  const objects = () => p.objectStores.buckets.get(registry.bucket) ?? new Map();
  const images: Bucket = {
    async head(key) {
      const found = objects().get(key);
      return found
        ? { size: found.bytes.byteLength, httpMetadata: { contentType: found.type } }
        : null;
    },
    async get(key) {
      const found = objects().get(key);
      if (!found) return null;
      return {
        size: found.bytes.byteLength,
        httpMetadata: { contentType: found.type },
        body: new Response(found.bytes).body!,
        text: async () => new TextDecoder().decode(found.bytes),
      };
    },
  };
  p.http.on(
    new RegExp(`^https://${registry.host.replace(/\./g, "\\.")}/`),
    async (url, _body, request) => {
      const response = await worker.fetch(
        new Request(url, { method: request.method, headers: request.headers }),
        { IMAGES: images, PULL_KEY },
      );
      return {
        status: response.status,
        text: await response.text(),
        headers: Object.fromEntries(response.headers),
      };
    },
  );
}
