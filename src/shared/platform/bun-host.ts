import os from "node:os";
import type { Host, Shell } from "../ports/index.ts";

export class BunHost implements Host {
  constructor(private readonly shell: Shell) {}
  async listeningPid(port: number) {
    // ss names the owner only for sockets this user may see; the unit runs as this user
    const result = await this.shell.run(["ss", "-ltnpH", `sport = :${port}`]);
    const pid = /pid=(\d+)/.exec(result.stdout)?.[1];
    return result.code === 0 && pid ? Number(pid) : null;
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
