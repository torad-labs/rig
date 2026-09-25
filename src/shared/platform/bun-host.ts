import { promises as fsp } from "node:fs";
import os from "node:os";
import type { Host, Shell } from "../ports/index.ts";

/** a LISTEN row in /proc/net/tcp{,6}: `sl local_address rem_address st … inode`, the port in hex */
const LISTEN = "0A";

export class BunHost implements Host {
  constructor(private readonly shell: Shell) {}
  /** read from /proc the way ss reads it, so a machine without iproute2 (a minimal container, a
   *  rented box's image) answers too: the listener's socket inode, then the process whose fd links
   *  to it. Only this user's processes are readable, and the unit runs as this user. */
  async listeningPid(port: number) {
    const inodes = new Set<string>();
    for (const table of ["/proc/net/tcp", "/proc/net/tcp6"]) {
      const text = await Bun.file(table)
        .text()
        .catch(() => "");
      for (const row of text.split("\n").slice(1)) {
        const [, local, , state, , , , , , inode] = row.trim().split(/\s+/);
        const listening =
          state === LISTEN && Number.parseInt(local?.split(":")[1] ?? "", 16) === port;
        if (listening && inode && inode !== "0") inodes.add(inode);
      }
    }
    if (inodes.size === 0) return null;
    for (const pid of await fsp.readdir("/proc")) {
      if (!/^\d+$/.test(pid)) continue;
      const fds = await fsp.readdir(`/proc/${pid}/fd`).catch(() => [] as string[]);
      for (const fd of fds) {
        const target = await fsp.readlink(`/proc/${pid}/fd/${fd}`).catch(() => "");
        if (inodes.has(/^socket:\[(\d+)\]$/.exec(target)?.[1] ?? "")) return Number(pid);
      }
    }
    return null;
  }
  async glibc() {
    const result = await this.shell.run(["getconf", "GNU_LIBC_VERSION"]);
    return result.code === 0 ? (/^glibc (\d+\.\d+)/.exec(result.stdout.trim())?.[1] ?? null) : null;
  }
  hostname() {
    return os.hostname();
  }
  cpuCount() {
    return os.availableParallelism();
  }
  async ramMiB() {
    try {
      const limit = (await Bun.file("/sys/fs/cgroup/memory.max").text()).trim();
      if (limit !== "max") return Math.floor(Number(limit) / 1048576);
    } catch {}
    return Math.floor(os.totalmem() / 1048576);
  }
}
