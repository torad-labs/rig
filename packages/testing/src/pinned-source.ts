// The engine source at the pin where it is on this box, for the tests that hold a manifest to the source itself.
// CI fetches no engine source, so such a test is skipped there by name (test.skipIf), never returned from early.

/** the pinned source where it is on the box, clean: the submodule at the pin, else the one-commit fetch */
export function pinnedSource(root: string, sha: string): string | null {
  for (const dir of [
    `${root}/engine/llama.cpp`,
    `${root}/local/engine-sources/llama.cpp-${sha.slice(0, 7)}`,
  ]) {
    const git = (...args: string[]) =>
      Bun.spawnSync(["git", "-C", dir, ...args])
        .stdout.toString()
        .trim();
    if (
      git("rev-parse", "HEAD") === sha &&
      git("status", "--porcelain", "--untracked-files=no") === ""
    )
      return dir;
  }
  return null;
}
