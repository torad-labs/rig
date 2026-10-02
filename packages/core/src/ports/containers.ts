// Containers: one seam between rig and the machine. A port names a capability, never a tool.
import type { RunResult } from "./shell.ts";

export interface ContainerRun {
  /** a card by nvidia-smi index, handed in as the container toolkit's CDI device */
  gpu?: number;
  /** host path → container path, read-only */
  mounts?: Record<string, string>;
}
export interface Containers {
  /** the image `context` holds a Dockerfile for, tagged `tag`, with each label; its output to `logPath` */
  build(
    context: string,
    tag: string,
    labels: Record<string, string>,
    logPath: string,
  ): Promise<RunResult>;
  /** `cmd` in a fresh container of `image`, removed after */
  run(image: string, cmd: readonly string[], options?: ContainerRun): Promise<RunResult>;
  /** the image as an OCI layout tar (index.json, blobs/sha256/…) at `tarball` */
  save(image: string, tarball: string): Promise<RunResult>;
}
