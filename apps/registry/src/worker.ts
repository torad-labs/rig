// rig's image registry: the read half of the OCI distribution API over one R2 bucket, behind one
// key. `rig image --push` writes the bucket directly, so nothing is pushed through this Worker and
// no layer is held to a request-body limit; the Worker only serves pulls, and its key can do
// nothing else. The bucket's layout is rig image's contract (packages/image/):
//   <repository>/blobs/sha256/<hex>       a layer or a config, by digest
//   <repository>/manifests/sha256/<hex>   a manifest by digest, its media type as the object's
//   <repository>/tags/<tag>               the digest the tag names, "sha256:<hex>"

/** the part of an R2 binding this Worker reads */
export interface Bucket {
  get(key: string): Promise<StoredObject | null>;
  head(key: string): Promise<StoredMeta | null>;
}
export interface StoredMeta {
  size: number;
  httpMetadata?: { contentType?: string | undefined };
}
export interface StoredObject extends StoredMeta {
  body: ReadableStream;
  text(): Promise<string>;
}
export interface Env {
  IMAGES: Bucket;
  /** the one password a pull presents (any user name), a Worker secret */
  PULL_KEY: string;
}

const NAME = /^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$/;
const TAG = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;
const DIGEST = /^sha256:[a-f0-9]{64}$/;
const API = { "Docker-Distribution-API-Version": "registry/2.0" };

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (!authorized(request.headers.get("authorization"), env.PULL_KEY))
      return new Response(error("UNAUTHORIZED", "a pull key is required"), {
        status: 401,
        headers: { ...API, "WWW-Authenticate": 'Basic realm="rig"', ...JSON_TYPE },
      });
    if (request.method !== "GET" && request.method !== "HEAD")
      return failure(405, "UNSUPPORTED", "this registry only serves pulls");
    const path = new URL(request.url).pathname;
    if (path === "/v2/" || path === "/v2")
      return new Response("{}", { headers: { ...API, ...JSON_TYPE } });
    const manifest = /^\/v2\/(.+)\/manifests\/([^/]+)$/.exec(path);
    if (manifest) return serveManifest(env.IMAGES, request.method, manifest[1]!, manifest[2]!);
    const blob = /^\/v2\/(.+)\/blobs\/([^/]+)$/.exec(path);
    if (blob) return serveBlob(env.IMAGES, request.method, blob[1]!, blob[2]!);
    return failure(404, "NOT_FOUND", `no route for ${path}`);
  },
};

async function serveManifest(bucket: Bucket, method: string, name: string, reference: string) {
  if (!NAME.test(name)) return failure(404, "NAME_UNKNOWN", `no repository ${name}`);
  let digest = reference;
  if (!DIGEST.test(reference)) {
    if (!TAG.test(reference)) return failure(404, "MANIFEST_UNKNOWN", `no manifest ${reference}`);
    const tag = await bucket.get(`${name}/tags/${reference}`);
    digest = tag ? (await tag.text()).trim() : "";
    if (!DIGEST.test(digest)) return failure(404, "MANIFEST_UNKNOWN", `no tag ${reference}`);
  }
  return serveObject(
    bucket,
    method,
    `${name}/manifests/${digest.replace(":", "/")}`,
    digest,
    "MANIFEST_UNKNOWN",
  );
}

function serveBlob(bucket: Bucket, method: string, name: string, digest: string) {
  if (!NAME.test(name)) return failure(404, "NAME_UNKNOWN", `no repository ${name}`);
  if (!DIGEST.test(digest)) return failure(404, "BLOB_UNKNOWN", `no blob ${digest}`);
  return serveObject(
    bucket,
    method,
    `${name}/blobs/${digest.replace(":", "/")}`,
    digest,
    "BLOB_UNKNOWN",
  );
}

async function serveObject(
  bucket: Bucket,
  method: string,
  key: string,
  digest: string,
  unknown: string,
) {
  const found = method === "HEAD" ? await bucket.head(key) : await bucket.get(key);
  if (!found) return failure(404, unknown, `no ${digest}`);
  const headers = {
    ...API,
    "Content-Type": found.httpMetadata?.contentType ?? "application/octet-stream",
    "Content-Length": String(found.size),
    "Docker-Content-Digest": digest,
  };
  const body = method === "HEAD" ? null : (found as StoredObject).body;
  return new Response(body, { headers });
}

/** Basic auth whose password is the pull key, compared in constant time */
function authorized(header: string | null, key: string): boolean {
  if (!key || !header?.startsWith("Basic ")) return false;
  let decoded: string;
  try {
    decoded = atob(header.slice("Basic ".length));
  } catch {
    return false;
  }
  const given = new TextEncoder().encode(decoded.slice(decoded.indexOf(":") + 1));
  const want = new TextEncoder().encode(key);
  let differ = given.length ^ want.length;
  for (let i = 0; i < want.length; i++) differ |= (given[i] ?? 0) ^ want[i]!;
  return differ === 0;
}

const JSON_TYPE = { "Content-Type": "application/json" };
const error = (code: string, message: string) => JSON.stringify({ errors: [{ code, message }] });
const failure = (status: number, code: string, message: string) =>
  new Response(error(code, message), { status, headers: { ...API, ...JSON_TYPE } });
