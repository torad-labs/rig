import { closeSync, promises as fsp, openSync, writeSync } from "node:fs";
import { dirname } from "node:path";
import type { FileStat, FileSystem } from "../ports/index.ts";

export class BunFileSystem implements FileSystem {
  readText(path: string) {
    return Bun.file(path).text();
  }
  async readBytes(path: string) {
    return new Uint8Array(await Bun.file(path).arrayBuffer());
  }
  async readRange(path: string, offset: number, length: number) {
    return new Uint8Array(
      await Bun.file(path)
        .slice(offset, offset + length)
        .arrayBuffer(),
    );
  }
  async writeText(path: string, text: string) {
    await fsp.mkdir(dirname(path), { recursive: true });
    await Bun.write(path, text);
  }
  async writeBytes(path: string, bytes: Uint8Array) {
    await fsp.mkdir(dirname(path), { recursive: true });
    await Bun.write(path, bytes);
  }
  async writeAt(path: string, offset: number, bytes: Uint8Array) {
    const fd = openSync(path, "r+");
    try {
      let done = 0;
      while (done < bytes.length)
        done += writeSync(fd, bytes, done, bytes.length - done, offset + done);
    } finally {
      closeSync(fd);
    }
  }
  async exists(path: string) {
    try {
      await fsp.lstat(path);
      return true;
    } catch {
      return false;
    }
  }
  async presence(path: string): Promise<"present" | "absent"> {
    try {
      await fsp.lstat(path);
      return "present";
    } catch (e) {
      const code = (e as NodeJS.ErrnoException).code;
      // ENOTDIR: a path component that would have to be a directory is a file instead, so the
      // path cannot exist either way — the same case as ENOENT. Anything else (EACCES, EIO, a
      // stale mount …) is answerable only by refusing, never by guessing "absent".
      if (code === "ENOENT" || code === "ENOTDIR") return "absent";
      throw e;
    }
  }
  async stat(path: string): Promise<FileStat | null> {
    try {
      const entry = await fsp.lstat(path);
      return {
        size: entry.size,
        mtimeMs: entry.mtimeMs,
        isDirectory: entry.isDirectory(),
        isSymlink: entry.isSymbolicLink(),
      };
    } catch {
      return null;
    }
  }
  async mkdirp(path: string) {
    await fsp.mkdir(path, { recursive: true });
  }
  async rename(from: string, to: string) {
    await fsp.rename(from, to);
  }
  async remove(path: string) {
    await fsp.rm(path, { recursive: true, force: true });
  }
  async list(dir: string) {
    return (await fsp.readdir(dir)).sort();
  }
  async copy(from: string, to: string) {
    await fsp.mkdir(dirname(to), { recursive: true });
    await Bun.write(to, Bun.file(from));
  }
  async copyTree(from: string, to: string) {
    await fsp.cp(from, to, { recursive: true, verbatimSymlinks: true, preserveTimestamps: true });
  }
  async linkOrCopy(from: string, to: string) {
    await fsp.mkdir(dirname(to), { recursive: true });
    try {
      await fsp.link(from, to);
    } catch {
      await this.copy(from, to);
    }
  }
  realpath(path: string) {
    return fsp.realpath(path);
  }
}
