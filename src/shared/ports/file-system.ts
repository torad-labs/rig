// FileSystem: one seam between rig and the machine. A port names a capability, never a tool.
export interface FileStat {
  size: number;
  mtimeMs: number;
  isDirectory: boolean;
  isSymlink: boolean;
}
export interface FileSystem {
  readText(path: string): Promise<string>;
  readBytes(path: string): Promise<Uint8Array>;
  /** `length` bytes from `offset`, without reading the rest of a multi-GB file */
  readRange(path: string, offset: number, length: number): Promise<Uint8Array>;
  writeText(path: string, text: string): Promise<void>;
  writeBytes(path: string, bytes: Uint8Array): Promise<void>;
  /** write `bytes` at `offset` inside an existing file, leaving the rest untouched */
  writeAt(path: string, offset: number, bytes: Uint8Array): Promise<void>;
  exists(path: string): Promise<boolean>;
  /** like `exists`, but never folds a permission error, an I/O error or a stale mount into
   *  "absent": only ENOENT, and ENOTDIR (a path component that would have to be a directory is
   *  a file instead, so the path cannot exist either way), mean the path is not there.
   *  Everything else throws, so a caller deciding something from a path's presence (an adapter
   *  that is missing vs. one this machine cannot read) is forced to refuse loudly instead of
   *  silently choosing "absent". */
  presence(path: string): Promise<"present" | "absent">;
  stat(path: string): Promise<FileStat | null>;
  mkdirp(path: string): Promise<void>;
  /** atomic on one filesystem: the destination is either the old thing or the new thing */
  rename(from: string, to: string): Promise<void>;
  remove(path: string): Promise<void>;
  list(dir: string): Promise<string[]>;
  copy(from: string, to: string): Promise<void>;
  /** copy a directory tree, symlinks and modes preserved */
  copyTree(from: string, to: string): Promise<void>;
  /** hard link when the two paths share a filesystem, else a copy */
  linkOrCopy(from: string, to: string): Promise<void>;
  /** the canonical path; throws ENOENT for a path that does not exist, like the OS call */
  realpath(path: string): Promise<string>;
}
