// An image into the registry's bucket, laid out as apps/registry/src/worker.ts serves it: the image as
// `docker save` writes it (an OCI layout, its layers uncompressed tars), each layer gzipped, the
// blobs first, then a Docker schema-2 manifest (the form every Docker a rented host runs pulls),
// then the tag, so a tag never names a manifest whose blobs are not there. Then read back through
// the registry with the pull key: the tag's digest and every blob's size, as a pull will see them.
import { join } from "node:path";
import type { Containers, FileSystem, Http, ObjectStore, Shell } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import { type RegistryConfig, servedDigest } from "./registry-config.ts";

export const MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json";
const CONFIG_TYPE = "application/vnd.docker.container.image.v1+json";
const LAYER_TYPE = "application/vnd.docker.image.rootfs.diff.tar.gzip";

interface Descriptor {
  mediaType: string;
  digest: string;
  size: number;
}
export interface PushDeps {
  containers: Containers;
  shell: Shell;
  fs: FileSystem;
  http: Http;
}

/** the manifest digest the tag now names */
export async function pushImage(
  deps: PushDeps,
  store: ObjectStore,
  registry: RegistryConfig,
  pullKey: string,
  image: string,
  tag: string,
  work: string,
): Promise<Result<string>> {
  const { fs, shell } = deps;
  await fs.remove(work);
  await fs.mkdirp(join(work, "layout"));
  try {
    const tarball = join(work, "image.tar");
    const saved = await deps.containers.save(image, tarball);
    if (saved.code !== 0)
      return fail(ExitCode.Failure, `docker save ${image}: ${saved.stderr.trim()}`);
    const untar = await shell.run(["tar", "-xf", tarball, "-C", join(work, "layout")], {
      timeoutMs: 600_000,
    });
    if (untar.code !== 0)
      return fail(ExitCode.Failure, `unpacking the saved image: ${untar.stderr.trim()}`);
    await fs.remove(tarball);

    const blob = (digest: string) => join(work, "layout", "blobs", ...digest.split(":"));
    const index = JSON.parse(await fs.readText(join(work, "layout", "index.json"))) as {
      manifests: Descriptor[];
    };
    const saving = index.manifests[0];
    if (!saving) return fail(ExitCode.Failure, "the saved image lists no manifest");
    const manifest = JSON.parse(await fs.readText(blob(saving.digest))) as {
      config: Descriptor;
      layers: Descriptor[];
    };
    const key = (kind: string, digest: string) =>
      `${registry.repository}/${kind}/${digest.replace(":", "/")}`;
    const upload = async (digest: string, from: { file: string } | { bytes: Uint8Array }) => {
      if (!(await store.exists(key("blobs", digest)))) await store.put(key("blobs", digest), from);
    };

    await upload(manifest.config.digest, { file: blob(manifest.config.digest) });
    const layers: Descriptor[] = [];
    for (const layer of manifest.layers) {
      // a layer docker saved compressed ships as it is; an uncompressed one is gzipped here, and
      // the config's diff_ids (the uncompressed digests) stay what they are, as the spec has it
      if (layer.mediaType.endsWith("gzip")) {
        await upload(layer.digest, { file: blob(layer.digest) });
        layers.push({ mediaType: LAYER_TYPE, digest: layer.digest, size: layer.size });
        continue;
      }
      const gz = Bun.gzipSync(new Uint8Array(await fs.readBytes(blob(layer.digest))));
      const digest = `sha256:${new Bun.CryptoHasher("sha256").update(gz).digest("hex")}`;
      await upload(digest, { bytes: gz });
      layers.push({ mediaType: LAYER_TYPE, digest, size: gz.byteLength });
    }
    const pushed = JSON.stringify({
      schemaVersion: 2,
      mediaType: MANIFEST_TYPE,
      config: {
        mediaType: CONFIG_TYPE,
        digest: manifest.config.digest,
        size: manifest.config.size,
      },
      layers,
    });
    const digest = `sha256:${new Bun.CryptoHasher("sha256").update(pushed).digest("hex")}`;
    await store.put(key("manifests", digest), { bytes: pushed }, MANIFEST_TYPE);
    await store.put(`${registry.repository}/tags/${tag}`, { bytes: digest }, "text/plain");

    const served = await readBack(deps.http, registry, pullKey, tag, digest, [
      { digest: manifest.config.digest, size: manifest.config.size },
      ...layers,
    ]);
    return served.ok ? ok(digest) : served;
  } finally {
    await fs.remove(work);
  }
}

/** the registry serves the tag as this digest and every blob at its size, to the pull key */
async function readBack(
  http: Http,
  registry: RegistryConfig,
  pullKey: string,
  tag: string,
  digest: string,
  blobs: { digest: string; size: number }[],
): Promise<Result<void>> {
  const base = `https://${registry.host}/v2/${registry.repository}`;
  const headers = { authorization: `Basic ${btoa(`rig:${pullKey}`)}` };
  const served = await servedDigest(http, registry, pullKey, tag);
  if (!served.ok || served.value !== digest)
    return fail(
      ExitCode.Failure,
      `${served.ok ? `${registry.host} serves ${tag} as ${served.value}` : served.message}, not the ${digest} just written`,
    );
  for (const blob of blobs) {
    const head = await http.request("HEAD", `${base}/blobs/${blob.digest}`, { headers });
    if (head.status !== 200 || head.headers?.["content-length"] !== String(blob.size))
      return fail(
        ExitCode.Failure,
        `${registry.host} serves blob ${blob.digest} as HTTP ${head.status}, ${head.headers?.["content-length"] ?? "?"} bytes, not ${blob.size}`,
      );
  }
  return ok(undefined);
}
