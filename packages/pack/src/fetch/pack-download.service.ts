// The source pack, every file of it by sha256, from Hugging Face, or adopted from a file the machine
// already has (`--from`, a one-file pack), because a 7.2 GB download a user already did once should
// not happen twice; when the head declares one, its draft head the same way; and every public
// [derive] asset (a step with a url) of the pack this machine derives, from that url. A download lands in <file>.part and
// is renamed only after the hash matches: a served path is never written in place
// (shared/artifact.ts says why). The download itself is shared/download.ts.

import type { FileSystem, Hasher, Log, Shell } from "@rig/core";
import {
  type Artifact,
  artifactProblem,
  checkArtifact,
  ExitCode,
  fail,
  fetchPinned,
  ok,
  type Result,
} from "@rig/core";
import type { Head } from "@rig/head";
import { deriveAsset, draftSidecar } from "@rig/head";

export interface DownloadPackOptions {
  /** a copy of the pinned pack already on this machine, linked or copied instead of fetched */
  from?: string | undefined;
}

export type PackDownloadState = "present" | "fetched" | "adopted";

export interface DownloadPackReport {
  /** the file the server loads: the pack's first */
  path: string;
  /** the pack's: "present" when every file was, else what its last missing one took */
  state: PackDownloadState;
  /** every file of a pack split over several, in order */
  files?: { path: string; state: PackDownloadState }[] | undefined;
  draft?: { path: string; state: PackDownloadState } | undefined;
  /** the public [derive] assets, in step order */
  assets?: { path: string; state: PackDownloadState }[] | undefined;
}

export interface DownloadPackDeps {
  shell: Shell;
  fs: FileSystem;
  hasher: Hasher;
  log: Log;
}

/** a file pinned on Hugging Face: the repo, the revision, the file, its sha256 */
interface Pin {
  repo: string;
  rev: string;
  file: string;
  sha256: string;
}

/** where a pinned file is downloaded from, and how the log names it */
interface Remote {
  url: string;
  named: string;
}

const onHub = (pin: Pin): Remote => ({
  url: `https://huggingface.co/${pin.repo}/resolve/${pin.rev}/${pin.file}`,
  named: `${pin.file} (${pin.repo} @ ${pin.rev.slice(0, 7)})`,
});

export class DownloadPack {
  constructor(private readonly deps: DownloadPackDeps) {}

  async run(head: Head, options: DownloadPackOptions = {}): Promise<Result<DownloadPackReport>> {
    const { repo, rev } = head.source;
    if (options.from && head.sourceFiles.length > 1)
      return fail(
        ExitCode.Failure,
        `--from adopts one file, and ${head.name}'s pack is split over ${head.sourceFiles.length}`,
      );
    // every shard at once: each downloads to its own .part and is hashed as it lands, beside the others' downloads rather
    // than before them; every one is awaited, so none is left downloading when another failed
    const states = await Promise.all(
      head.sourceFiles.map((file) => {
        const pin = { repo, rev, file: file.file, sha256: file.sha256 };
        return this.obtain(head, onHub(pin), file, "pack", options.from);
      }),
    );
    const files: { path: string; state: PackDownloadState }[] = [];
    for (const [i, file] of head.sourceFiles.entries()) {
      const state = states[i]!;
      if (!state.ok) return state;
      files.push({ path: file.path, state: state.value });
    }
    const fetched = files.filter((file) => file.state !== "present").at(-1);
    const report: DownloadPackReport = {
      path: head.sourcePath,
      state: fetched?.state ?? "present",
      ...(files.length > 1 ? { files } : {}),
    };
    const sidecar = head.speculative && draftSidecar(head.speculative);
    if (sidecar && head.draftPath) {
      const draft = { path: head.draftPath, sha256: sidecar.sha256 };
      const draftState = await this.obtain(head, onHub(sidecar), draft, "draft head", undefined);
      if (!draftState.ok) return draftState;
      report.draft = { path: draft.path, state: draftState.value };
    }
    for (const step of head.derive ?? []) {
      const { url, path, sha256 } = deriveAsset(step);
      if (!url) continue;
      const asset = { path: head.assetPath(step), sha256 };
      const remote = { url, named: `${path} (${url})` };
      const state = await this.obtain(head, remote, asset, `${step.kind} asset`, undefined);
      if (!state.ok) return state;
      report.assets = [...(report.assets ?? []), { path: asset.path, state: state.value }];
    }
    return ok(report);
  }

  /** present already, adopted from `from`, or fetched */
  private async obtain(
    head: Head,
    remote: Remote,
    target: Artifact,
    what: string,
    from: string | undefined,
  ): Promise<Result<PackDownloadState>> {
    const current = await checkArtifact(this.deps.fs, this.deps.hasher, target);
    if (current === "ok") return ok("present");
    if (typeof current === "object") {
      const message = `${target.path} is ${artifactProblem(current)} — refusing to fetch over it`;
      return fail(ExitCode.Failure, message);
    }
    if (from) {
      const adopted = await this.adopt(head, from, target);
      if (!adopted.ok) return adopted;
      return ok("adopted");
    }
    const fetched = await this.download(remote, target, what);
    if (!fetched.ok) return fetched;
    return ok("fetched");
  }

  /** a file already here, if it is the pinned bytes, linked or copied into place */
  private async adopt(head: Head, from: string, target: Artifact): Promise<Result<void>> {
    const state = await checkArtifact(this.deps.fs, this.deps.hasher, {
      path: from,
      sha256: target.sha256,
    });
    if (state !== "ok") {
      return fail(ExitCode.Failure, `${from} is ${artifactProblem(state)}`);
    }
    await this.deps.fs.mkdirp(head.packsDir);
    await this.deps.fs.linkOrCopy(from, target.path);
    return ok(undefined);
  }

  private async download(remote: Remote, target: Artifact, what: string): Promise<Result<void>> {
    this.deps.log.info(`fetching ${remote.named}`);
    return fetchPinned(this.deps, remote.url, target, what);
  }
}
