// The seams between rig and the machine, one file per capability; this file is the set every
// use case is handed and the one import site for any of them.

import type { Clock } from "./clock.ts";
import type { FileSystem } from "./file-system.ts";
import type { Git } from "./git.ts";
import type { Gpu } from "./gpu.ts";
import type { Hasher } from "./hasher.ts";
import type { Host } from "./host.ts";
import type { Http } from "./http.ts";
import type { Log } from "./log.ts";
import type { Rental } from "./rental.ts";
import type { Shell } from "./shell.ts";
import type { Ssh } from "./ssh.ts";
import type { Systemd } from "./systemd.ts";

export type { Clock } from "./clock.ts";
export type { FileStat, FileSystem } from "./file-system.ts";
export type { Git } from "./git.ts";
export type { Gpu, GpuInfo } from "./gpu.ts";
export type { Hasher } from "./hasher.ts";
export type { Host } from "./host.ts";
export type { Http, HttpResponse } from "./http.ts";
export type { Log } from "./log.ts";
export type { Instance, Offer, Rental } from "./rental.ts";
export type { Process, RunOptions, RunResult, Shell, SpawnOptions } from "./shell.ts";
export type { Ssh, SshTarget } from "./ssh.ts";
export type { Systemd } from "./systemd.ts";

export interface Ports {
  shell: Shell;
  fs: FileSystem;
  http: Http;
  gpu: Gpu;
  systemd: Systemd;
  git: Git;
  hasher: Hasher;
  host: Host;
  rental: Rental;
  ssh: Ssh;
  clock: Clock;
  log: Log;
}
