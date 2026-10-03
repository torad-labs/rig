// An image as `docker save` writes it: an OCI layout (index.json, blobs/sha256/…). Two callers read it,
// and they must agree on what the bytes are, so the reading is here once: the push lays the blobs out
// for the registry (oci-push.ts), and a push of an image proven elsewhere compares its config digest
// against the receipt's (image-build.service.ts). The config digest IS docker's image ID: it covers the
// layer diff_ids, the entrypoint and the environment, so two images with it equal are the same image,
// and `docker save` on one machine and `docker load` on another preserve it.
import { join } from "node:path";
import type { Containers, FileSystem, Shell } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";

export interface Descriptor {
  mediaType: string;
  digest: string;
  size: number;
}
export interface SavedImageDeps {
  containers: Containers;
  shell: Shell;
  fs: FileSystem;
}
export interface SavedImage {
  manifest: { config: Descriptor; layers: Descriptor[] };
  /** where a descriptor's bytes are, under the unpacked layout */
  blob(digest: string): string;
}

/** `image` as `docker save` writes it, at `tarball` */
export async function saveImage(
  deps: SavedImageDeps,
  image: string,
  tarball: string,
): Promise<Result<void>> {
  const saved = await deps.containers.save(image, tarball);
  return saved.code === 0
    ? ok(undefined)
    : fail(ExitCode.Failure, `docker save ${image}: ${saved.stderr.trim()}`);
}

/** the manifest of a saved tarball, unpacked under `into`. The caller removes `into`. */
export async function readLayout(
  deps: SavedImageDeps,
  tarball: string,
  into: string,
): Promise<Result<SavedImage>> {
  const { fs, shell } = deps;
  const layout = join(into, "layout");
  await fs.mkdirp(layout);
  const untar = await shell.run(["tar", "-xf", tarball, "-C", layout], { timeoutMs: 600_000 });
  if (untar.code !== 0)
    return fail(ExitCode.Failure, `unpacking the saved image: ${untar.stderr.trim()}`);

  const index = JSON.parse(await fs.readText(join(layout, "index.json"))) as {
    manifests: Descriptor[];
  };
  const saving = index.manifests[0];
  if (!saving) return fail(ExitCode.Failure, "the saved image lists no manifest");
  const blob = (digest: string) => join(layout, "blobs", ...digest.split(":"));
  const manifest = JSON.parse(await fs.readText(blob(saving.digest))) as {
    config: Descriptor;
    layers: Descriptor[];
  };
  return ok({ manifest, blob });
}

/** both, with the tarball inside `work` and removed once read. The caller removes `work`. */
export async function readSavedImage(
  deps: SavedImageDeps,
  image: string,
  work: string,
): Promise<Result<SavedImage>> {
  const tarball = join(work, "image.tar");
  await deps.fs.mkdirp(work);
  const saved = await saveImage(deps, image, tarball);
  if (!saved.ok) return saved;
  const read = await readLayout(deps, tarball, work);
  await deps.fs.remove(tarball);
  return read;
}
