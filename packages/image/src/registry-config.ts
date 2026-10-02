// registry.toml: where `rig image --push` publishes and what a rented box pulls from. The registry
// (apps/registry/src/worker.ts) serves pulls at `host` behind a pull key; rig writes the image straight into
// its bucket. No credential lives in the file: the operator's keyring holds the bucket's upload token and the key a pull
// presents (service rig-registry), each read only by the step that needs it; the environment's RIG_R2_ACCESS_KEY_ID,
// RIG_R2_SECRET_ACCESS_KEY and RIG_REGISTRY_PULL_KEY, where set (CI), come first. Absent (the public copy of rig), an
// image is built and proven but has nowhere to go.
import { join } from "node:path";
import type { FileSystem, Http, Secrets } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import * as v from "valibot";

const RegistrySchema = v.strictObject({
  registry: v.strictObject({
    host: v.pipe(v.string(), v.regex(/^[a-z0-9.-]+(:\d+)?$/, "a host name, no scheme")),
    repository: v.pipe(v.string(), v.regex(/^[a-z0-9]+(?:[._-][a-z0-9]+)*$/)),
    bucket: v.pipe(v.string(), v.minLength(3)),
    endpoint: v.pipe(v.string(), v.startsWith("https://")),
  }),
});
export type RegistryConfig = v.InferOutput<typeof RegistrySchema>["registry"];

export interface RegistryCredentials {
  accessKeyId?: string | undefined;
  secretAccessKey?: string | undefined;
  pullKey?: string | undefined;
}

/** the digest the registry serves `tag` as, to this pull key, the way a box's pull asks: a key it refuses or a tag it
 *  does not hold fails, by its status */
export async function servedDigest(
  http: Http,
  registry: RegistryConfig,
  pullKey: string,
  tag: string,
): Promise<Result<string>> {
  const url = `https://${registry.host}/v2/${registry.repository}/manifests/${tag}`;
  const headers = { authorization: `Basic ${btoa(`rig:${pullKey}`)}` };
  const served = await http.request("HEAD", url, { headers });
  const digest = served.headers?.["docker-content-digest"];
  if (served.status === 200 && digest) return ok(digest);
  const why = served.status === 401 ? ": the pull key is refused" : "";
  return fail(
    ExitCode.Failure,
    `${registry.host} does not serve ${registry.repository}:${tag} (HTTP ${served.status}${why})`,
  );
}

/** the keyring entries: `secret-tool lookup service rig-registry key <key>` */
const KEYRING = "rig-registry";

/** the registry's credentials, each from the environment where it is set, else from the keyring */
export async function registryCredentials(
  secrets: Secrets,
  env: Readonly<Record<string, string | undefined>>,
): Promise<RegistryCredentials> {
  const read = async (name: string, key: string) =>
    env[name] || (await secrets.lookup(KEYRING, key)) || undefined;
  return {
    accessKeyId: await read("RIG_R2_ACCESS_KEY_ID", "r2-access-key-id"),
    secretAccessKey: await read("RIG_R2_SECRET_ACCESS_KEY", "r2-secret-access-key"),
    pullKey: await read("RIG_REGISTRY_PULL_KEY", "pull"),
  };
}

/** the registry, or null when this checkout names none */
export async function loadRegistryConfig(
  fs: FileSystem,
  root: string,
): Promise<Result<RegistryConfig | null>> {
  const file = join(root, "registry.toml");
  if (!(await fs.exists(file))) return ok(null);
  let data: unknown;
  try {
    data = Bun.TOML.parse(await fs.readText(file));
  } catch (error) {
    return fail(ExitCode.Failure, `registry.toml does not parse: ${(error as Error).message}`);
  }
  const parsed = v.safeParse(RegistrySchema, data);
  if (!parsed.success)
    return fail(
      ExitCode.Failure,
      `registry.toml is invalid: ${parsed.issues.map((issue) => `${v.getDotPath(issue) ?? "(root)"}: ${issue.message}`).join("; ")}`,
    );
  return ok(parsed.output.registry);
}
