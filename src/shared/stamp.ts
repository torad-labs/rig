// A moment as a path-safe token: 20260920T082930Z. Sortable, one per second, no colon or dash
// for a file name, and the same shape wherever rig names a run or a backup.
export const compactStamp = (ms: number) =>
  new Date(ms)
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d+Z$/, "Z");
