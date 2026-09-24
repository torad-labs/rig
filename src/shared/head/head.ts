import { join } from "node:path";
import * as v from "valibot";
import type { Layout } from "../layout.ts";
import type { FileSystem } from "../ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../result.ts";
import {
  deriveAsset,
  draftSidecar,
  type HeadConfig,
  HeadSchema,
  headInvariants,
} from "./head-config.ts";

/** A loaded head: its config plus where its files are. Paths are absolute. */
export interface Head extends HeadConfig {
  dir: string;
  /** resolve a head-relative path such as "assets/chat-template.jinja" */
  path(rel: string): string;
  /** where this head's packs live under local/ */
  packsDir: string;
  sourcePath: string;
  servedPath: string;
  /** the served pack's path and pin as head.toml declares them, even when this machine is
   *  undrived and served/servedPath point at source instead: a verify asked to check the exact
   *  file a rendered unit's ExecStart names still needs the pin that file was published against */
  declaredServed: { path: string; sha256: string };
  /** the draft head's own file, when [speculative] declares a sidecar (an in-pack head has none) */
  draftPath?: string;
  /** why this machine serves the source pack although head.toml declares a [derive] step */
  undrived?: string;
}

export function parseHeadToml(text: string): Result<HeadConfig> {
  let data: unknown;
  try {
    data = Bun.TOML.parse(text);
  } catch (e) {
    return fail(ExitCode.Failure, `head.toml does not parse: ${(e as Error).message}`);
  }
  const parsed = v.safeParse(HeadSchema, data);
  if (!parsed.success) {
    const issues = parsed.issues
      .map((i) => `${v.getDotPath(i) ?? "(root)"}: ${i.message}`)
      .join("; ");
    return fail(ExitCode.Failure, `head.toml is invalid: ${issues}`);
  }
  const errs = headInvariants(parsed.output);
  if (errs.length) return fail(ExitCode.Failure, `head.toml violates: ${errs.join("; ")}`);
  return ok(parsed.output);
}

export async function loadHead(
  fs: FileSystem,
  layout: Layout,
  name: string,
): Promise<Result<Head>> {
  const dir = join(layout.headsDir, name);
  const file = join(dir, "head.toml");
  if (!(await fs.exists(file)))
    return fail(ExitCode.Failure, `no head named ${JSON.stringify(name)}: ${file} does not exist`);
  const cfg = parseHeadToml(await fs.readText(file));
  if (!cfg.ok) return cfg;
  if (cfg.value.name !== name)
    return fail(
      ExitCode.Failure,
      `${file} names itself ${JSON.stringify(cfg.value.name)} but lives in heads/${name}`,
    );
  const packsDir = layout.packs(name);
  const sidecar = cfg.value.speculative && draftSidecar(cfg.value.speculative);
  const sourcePath = join(packsDir, cfg.value.source.file);
  const servedPath = join(packsDir, cfg.value.served.file);
  const located = {
    dir,
    path: (rel: string) => join(dir, rel),
    packsDir,
    sourcePath,
    ...(sidecar ? { draftPath: join(packsDir, sidecar.file) } : {}),
  };
  const declaredServed = { path: servedPath, sha256: cfg.value.served.sha256 };
  const undrived = await undrivedReason(fs, dir, servedPath, cfg.value.derive);
  if (!undrived.ok) return undrived;
  if (undrived.value) {
    const { derive: _, ...plain } = cfg.value;
    return ok({
      ...plain,
      ...located,
      served: plain.source,
      servedPath: sourcePath,
      declaredServed,
      undrived: undrived.value,
    });
  }
  return ok({ ...cfg.value, ...located, servedPath, declaredServed });
}

/** A [derive] step's asset (an adapter, a draft head) is private: pinned by sha256 in head.toml,
 *  fetched out of band, never in git. The served pack is every step or none: a machine missing any
 *  asset and the pack they derive serves the source pack; the pack present keeps the steps, so an
 *  asset going missing never swaps a pack that is already derived. Presence, never `exists`: an
 *  EACCES, EIO or stale mount on any path is not "not there" and must refuse loudly, not silently
 *  fall back to serving the source pack. */
async function undrivedReason(
  fs: FileSystem,
  dir: string,
  servedPath: string,
  derive: HeadConfig["derive"],
): Promise<Result<string | undefined>> {
  if (!derive) return ok(undefined);
  const missing: string[] = [];
  for (const step of derive) {
    const asset = deriveAsset(step).path;
    const state = await presence(fs, join(dir, asset));
    if (!state.ok) return state;
    if (state.value === "absent") missing.push(asset);
  }
  const served = await presence(fs, servedPath);
  if (!served.ok) return served;
  if (missing.length === 0 || served.value === "present") return ok(undefined);
  return ok(
    `the [derive] asset ${missing.join(", ")} is not on this machine, nor the pack it derives: serving the source pack`,
  );
}

/** `fs.presence`, with a non-ENOENT error turned into a refusal that names the path and the code,
 *  instead of propagating a raw exception out of a function whose contract is `Result` */
async function presence(fs: FileSystem, path: string): Promise<Result<"present" | "absent">> {
  try {
    return ok(await fs.presence(path));
  } catch (e) {
    const code = (e as NodeJS.ErrnoException)?.code ?? "unknown error";
    return fail(
      ExitCode.Failure,
      `cannot tell whether ${path} exists (${code}) — refusing to guess`,
    );
  }
}

export async function listHeads(fs: FileSystem, layout: Layout): Promise<string[]> {
  if (!(await fs.exists(layout.headsDir))) return [];
  const names: string[] = [];
  for (const n of await fs.list(layout.headsDir))
    if (await fs.exists(join(layout.headsDir, n, "head.toml"))) names.push(n);
  return names.sort();
}
