import type { Ports } from "@rig/core";
import { BunFileSystem } from "./bun-file-system.ts";
import { BunHasher } from "./bun-hasher.ts";
import { BunHost } from "./bun-host.ts";
import { BunShell } from "./bun-shell.ts";
import { ConsoleLog } from "./console-log.ts";
import { DockerContainers } from "./docker-containers.ts";
import { FetchHttp } from "./fetch-http.ts";
import { GitCli } from "./git-cli.ts";
import { GpuLeaseCardLease } from "./gpu-lease-card-lease.ts";
import { NvidiaSmiGpu } from "./nvidia-smi-gpu.ts";
import { OpenSsh } from "./open-ssh.ts";
import { S3ObjectStores } from "./s3-object-stores.ts";
import { SecretToolSecrets } from "./secret-tool-secrets.ts";
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
    cardLease: new GpuLeaseCardLease(shell),
    systemd: new SystemctlSystemd(shell),
    git: new GitCli(shell),
    hasher: new BunHasher(),
    host: new BunHost(shell),
    rental: new VastAiRental(shell),
    ssh: new OpenSsh(shell),
    clock: new SystemClock(),
    containers: new DockerContainers(shell),
    objectStores: new S3ObjectStores(),
    secrets: new SecretToolSecrets(shell),
    log: new ConsoleLog(),
  };
}
