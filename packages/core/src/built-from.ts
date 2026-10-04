// What `bun run build` (tools/build.ts) stamps into dist/rig, and what `rig vast up` and `lab` read back before they ship
// it to a box: the tree of the commit it was built from. A tree, not the commit: a tag a history rewrite moves to
// another commit keeps its tree, and must build the same bytes (release.yml). DIRTY follows the tree when a source the
// compile read was not the tree's.
export const BUILT_FROM = {
  /** what the stamp names, as git rev-parse resolves it */
  ref: "HEAD^{tree}",
  /** the flag the compiled rig prints its stamp under */
  flag: "--built-from",
  /** after the tree: a source the compile read differed from it */
  dirty: "-dirty",
} as const;
