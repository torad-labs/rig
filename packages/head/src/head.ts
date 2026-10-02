import { join } from "node:path";
import type { Artifact, FileSystem, Layout } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import * as v from "valibot";
import {
  type Derive,
  deriveAsset,
  draftSidecar,
  type HeadConfig,
  HeadSchema,
  headInvariants,
  type PinnedFile,
  publicSteps,
} from "./head-config.ts";

/** a pinned file where this machine keeps it: its absolute path beside its pin */
export interface PackFile extends Artifact {
  /** the path head.toml pins, relative to the repo it is fetched from and to the head's packs directory */
  file: string;
  bytes: number;
}

/** A loaded head: its config plus where its files are. Paths are absolute. */
export interface Head extends HeadConfig {
  dir: string;
  /** resolve a head-relative path such as "assets/chat-template.jinja" */
  path(rel: string): string;
  /** where this head's packs live under local/ */
  packsDir: string;
  /** the source pack's files, in order: the first is the one the server loads, a split GGUF's shards beside it */
  sourceFiles: PackFile[];
  sourcePath: string;
  /** the pack this machine serves (the derived one, the public one or the source), its first file the one -m loads */
  servedFiles: PackFile[];
  servedPath: string;
  /** the derived pack as head.toml declares it, even when this machine is undrived and serves another: a verify asked
   *  to check the exact file a rendered unit's ExecStart names still needs the pin that file was published against */
  declaredServed?: PackFile;
  /** the [public] pack, when head.toml declares one, whichever pack this machine serves */
  declaredPublic?: PackFile;
  /** where a derive step's asset lives: a public one (a url) in packsDir, a private one in dir */
  assetPath(step: Derive): string;
  /** the draft head's own file, when [speculative] declares a sidecar (an in-pack head has none) */
  draftPath?: string;
  /** why this machine serves the public or the source pack although head.toml declares a [derive] step */
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
  const at = (pin: PinnedFile): PackFile => ({ ...pin, path: join(packsDir, pin.file) });
  const sidecar = cfg.value.speculative && draftSidecar(cfg.value.speculative);
  const sourceFiles = cfg.value.source.files.map(at);
  const sourcePath = sourceFiles[0]!.path;
  const located = {
    dir,
    path: (rel: string) => join(dir, rel),
    packsDir,
    sourceFiles,
    sourcePath,
    assetPath: (step: Derive) => {
      const asset = deriveAsset(step);
      return join(asset.url ? packsDir : dir, asset.path);
    },
    ...(sidecar ? { draftPath: join(packsDir, sidecar.file) } : {}),
  };
  const declaredServed = cfg.value.served && at(cfg.value.served);
  const declaredPublic = cfg.value.public && at(cfg.value.public);
  const pinned = {
    ...located,
    ...(declaredServed ? { declaredServed } : {}),
    ...(declaredPublic ? { declaredPublic } : {}),
  };
  if (!declaredServed) {
    return ok({ ...cfg.value, ...pinned, servedFiles: sourceFiles, servedPath: sourcePath });
  }
  const undrived = await undrivedReason(fs, located, declaredServed.path, cfg.value.derive);
  if (!undrived.ok) return undrived;
  if (undrived.value) {
    const { derive, served: _served, ...plain } = cfg.value;
    const pub = cfg.value.public;
    if (pub && derive && declaredPublic) {
      return ok({
        ...plain,
        ...pinned,
        derive: publicSteps(derive),
        served: pub,
        servedFiles: [declaredPublic],
        servedPath: declaredPublic.path,
        undrived: `${undrived.value}: serving the public pack ${pub.file} (the source pack and the public [derive] steps)`,
      });
    }
    return ok({
      ...plain,
      ...pinned,
      servedFiles: sourceFiles,
      servedPath: sourcePath,
      undrived: `${undrived.value}: serving the source pack`,
    });
  }
  return ok({
    ...cfg.value,
    ...pinned,
    servedFiles: [declaredServed],
    servedPath: declaredServed.path,
  });
}

/** A [derive] step's private asset (an adapter, a draft head without a url) is pinned by sha256 in
 *  head.toml, fetched out of band, never in git; a public one `rig fetch` gets, so its absence here
 *  is not a reason. The served pack is every step or none: a machine missing a private asset and
 *  the pack the steps derive serves the [public] pack when there is one, else the source pack; the
 *  pack present keeps the steps, so an asset going missing never swaps a pack that is already
 *  derived. Presence, never `exists`: an EACCES, EIO or stale mount on any path is not "not there"
 *  and must refuse loudly, not silently fall back to a lesser pack. */
async function undrivedReason(
  fs: FileSystem,
  head: Pick<Head, "assetPath">,
  servedPath: string,
  derive: HeadConfig["derive"],
): Promise<Result<string | undefined>> {
  if (!derive) return ok(undefined);
  const missing: string[] = [];
  for (const step of derive) {
    const asset = deriveAsset(step);
    if (asset.url) continue;
    const state = await presence(fs, head.assetPath(step));
    if (!state.ok) return state;
    if (state.value === "absent") missing.push(asset.path);
  }
  const served = await presence(fs, servedPath);
  if (!served.ok) return served;
  if (missing.length === 0 || served.value === "present") return ok(undefined);
  return ok(
    `the [derive] asset ${missing.join(", ")} is not on this machine, nor the pack it derives`,
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

/** the head's private [derive] assets (no url), head-relative: what leaves this machine only when
 *  asked by name (`vast up --private`), and never in an image */
export function privateAssets(head: Pick<HeadConfig, "derive">): string[] {
  return (head.derive ?? []).map(deriveAsset).flatMap((asset) => (asset.url ? [] : [asset.path]));
}
