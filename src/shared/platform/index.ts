import type { Ports } from "../ports/index.ts";
import { BunFileSystem } from "./bun-file-system.ts";
import { BunHasher } from "./bun-hasher.ts";
import { BunHost } from "./bun-host.ts";
import { BunShell } from "./bun-shell.ts";
import { ConsoleLog } from "./console-log.ts";
import { FetchHttp } from "./fetch-http.ts";
import { GitCli } from "./git-cli.ts";
import { NvidiaSmiGpu } from "./nvidia-smi-gpu.ts";
import { OpenSsh } from "./open-ssh.ts";
import { SystemClock } from "./system-clock.ts";
import { SystemctlSystemd } from "./systemctl-systemd.ts";
import { VastAiRental } from "./vast-ai-rental.ts";

/** The real machine. The only place adapters are constructed. */
export function realPorts(): Ports {
  const shell = new BunShell();
  return {
    shell,
    fs: new BunFileSystem(),
    http: new FetchHttp(),
    gpu: new NvidiaSmiGpu(shell),
    systemd: new SystemctlSystemd(shell),
    git: new GitCli(shell),
    hasher: new BunHasher(),
    host: new BunHost(shell),
    rental: new VastAiRental(shell),
    ssh: new OpenSsh(shell),
    clock: new SystemClock(),
    log: new ConsoleLog(),
  };
}
