// A file whose identity is its sha256. Every pack rig fetches, derives or serves is one of these,
// and the two rules below are what make "it is on disk" mean "it is the right bytes":
//   * a path is trusted only after its hash matched; "the file exists" is never enough
//   * nothing writes a served path in place. Writes go to a sibling and are renamed over
//     (rename is atomic), because a running llama-server has the served file mmapped and an
//     in-place truncation is a SIGBUS in the live head, not a problem at the next restart.
import type { FileSystem, Hasher } from "./ports/index.ts";

export interface Artifact {
  path: string;
  sha256: string;
}
/** `unreadable` is there but cannot be read (EACCES, EIO, a stale mount): never "missing", which
 *  a caller answers by fetching or deriving over the path */
export type ArtifactState = "ok" | "missing" | "mismatch" | { unreadable: string };
/** publishing names the hash it produced on a mismatch, so a new derivation can be pinned from the refusal */
export type PublishState = "ok" | "missing" | { mismatch: string } | { unreadable: string };

export async function checkArtifact(
  fs: FileSystem,
  hasher: Hasher,
  a: Artifact,
): Promise<ArtifactState> {
  const hash = await readHash(fs, hasher, a.path);
  if (typeof hash === "object" || hash === "missing") return hash;
  return hash === a.sha256 ? "ok" : "mismatch";
}

/** why a file is not the artifact, for a refusal line */
export function artifactProblem(state: Exclude<ArtifactState, "ok">): string {
  if (state === "missing") return "missing";
  if (state === "mismatch") return "not the pinned bytes (sha256 differs)";
  return `unreadable (${state.unreadable})`;
}

/** the file's sha256, "missing" for ENOENT/ENOTDIR only (a dangling link included), or the errno
 *  that kept it from being read */
async function readHash(
  fs: FileSystem,
  hasher: Hasher,
  path: string,
): Promise<string | "missing" | { unreadable: string }> {
  try {
    if ((await fs.presence(path)) === "absent") return "missing";
    return await hasher.sha256File(path);
  } catch (e) {
    const code = (e as NodeJS.ErrnoException)?.code;
    if (code === "ENOENT" || code === "ENOTDIR") return "missing";
    return { unreadable: code ?? (e as Error)?.message ?? String(e) };
  }
}

/** The sibling path a producer writes to before publishing `final`. */
export const stagingPath = (final: string, tag: string) => `${final}.${tag}`;

/** Verify the staged file's hash, then rename it into place; on mismatch remove it and report
 *  the hash the staged file had. */
export async function publishArtifact(
  fs: FileSystem,
  hasher: Hasher,
  staged: string,
  a: Artifact,
): Promise<PublishState> {
  const produced = await readHash(fs, hasher, staged);
  if (typeof produced === "object" || produced === "missing") return produced;
  if (produced !== a.sha256) {
    await fs.remove(staged);
    return { mismatch: produced };
  }
  await fs.rename(staged, a.path);
  return "ok";
}
