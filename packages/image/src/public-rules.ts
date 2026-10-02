// What may leave this machine, and in what form: public-export.toml's rules, applied one file at a
// time by every path that publishes rig (tools/public-export.ts to the public repo, `rig image
// build` to a registry), so the two never drift. An exclude glob removes a file; a rewrite gives a
// private string its public form; a deny pattern (in a path or a line), a denied sha256 or a binary
// nobody has read and named refuses the publication, each by name.
import { join } from "node:path";
import type { FileSystem } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import { Glob } from "bun";

export interface PublicRules {
  exclude: string[];
  rewrite: { from: string; to: string }[];
  deny: string[];
  deny_sha256: string[];
  allow_binary_sha256: string[];
}

export async function loadPublicRules(fs: FileSystem, root: string): Promise<Result<PublicRules>> {
  const file = join(root, "public-export.toml");
  if (!(await fs.exists(file)))
    return fail(ExitCode.Failure, `${file} is missing: nothing may be published without its rules`);
  const data = Bun.TOML.parse(await fs.readText(file)) as Partial<PublicRules>;
  const { exclude, rewrite, deny, deny_sha256, allow_binary_sha256 } = data;
  if (
    !Array.isArray(exclude) ||
    !Array.isArray(rewrite) ||
    !Array.isArray(deny) ||
    deny.length === 0 ||
    !Array.isArray(deny_sha256) ||
    !Array.isArray(allow_binary_sha256)
  )
    return fail(
      ExitCode.Failure,
      `${file} needs exclude, [[rewrite]], deny_sha256, allow_binary_sha256 and a non-empty deny list`,
    );
  return ok({ exclude, rewrite, deny, deny_sha256, allow_binary_sha256 });
}

/** the exclude glob that removes `path` (repo-relative), if one does */
export function excludedBy(rules: Pick<PublicRules, "exclude">, path: string): string | undefined {
  return rules.exclude.find((pattern) => new Glob(pattern).match(path));
}

/** one file as it may be published: its text with every rewrite applied (a binary unchanged), and each check it
 *  fails. Excludes are the caller's: a file it removes is never read. */
export function publicFile(
  rules: Omit<PublicRules, "exclude">,
  path: string,
  bytes: Uint8Array,
): { bytes: Uint8Array; hits: string[] } {
  const hits: string[] = [];
  // the deny patterns name private places, and a path is one: a file under a directory named after
  // one ships that name whatever its contents say
  for (const pattern of rules.deny) {
    if (path.includes(pattern)) hits.push(`${path}: the path itself holds ${pattern}`);
  }
  const sha256 = new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
  if (rules.deny_sha256.includes(sha256)) hits.push(`${path}: denied content, sha256 ${sha256}`);
  if (bytes.includes(0)) {
    // A binary is opaque to the line scan below — UTF-16, compressed or packed bytes hold a path it
    // cannot see — so the rule runs the other way here: a binary ships because someone read it and
    // named its hash. deny_sha256 above still wins, and the UTF-8 scan stays as a second net.
    if (!rules.allow_binary_sha256.includes(sha256)) {
      hits.push(
        `${path}: an unlisted binary, sha256 ${sha256} — read it, then name it in allow_binary_sha256`,
      );
    }
    hits.push(...rawHits(rules, path, bytes));
    return { bytes, hits };
  }
  let text = new TextDecoder().decode(bytes);
  for (const { from, to } of rules.rewrite) text = text.replaceAll(from, to);
  text.split("\n").forEach((line, index) => {
    for (const pattern of rules.deny) {
      if (line.includes(pattern)) hits.push(`${path}:${index + 1}: ${pattern}`);
    }
  });
  return { bytes: new TextEncoder().encode(text), hits };
}

/** each deny pattern `bytes` holds anywhere, as raw bytes: a binary's check, and a built artifact's
 *  (an engine compiled here carries its source paths in its assert strings) */
export function rawHits(
  rules: Pick<PublicRules, "deny">,
  path: string,
  bytes: Uint8Array,
): string[] {
  const raw = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return rules.deny.flatMap((pattern) =>
    raw.includes(pattern) ? [`${path}: binary holds ${pattern}`] : [],
  );
}
