// The source pack, by sha256, from Hugging Face, or adopted from a file the machine already has
// (`--from`), because a 7.2 GB download a user already did once should not happen twice; when the
// head declares one, its draft head the same way; and every public [derive] asset (a step with a
// url) of the pack this machine derives, from that url. A download lands in <file>.part and
// is renamed only after the hash matches: a served path is never written in place
// (shared/artifact.ts says why). The download itself is shared/download.ts.
import { type Artifact, artifactProblem, checkArtifact } from "../../shared/artifact.ts";
import { fetchPinned } from "../../shared/download.ts";
import type { Head } from "../../shared/head/head.ts";
import { deriveAsset, draftSidecar } from "../../shared/head/head-config.ts";
import type { FileSystem, Hasher, Log, Shell } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";

export interface DownloadPackOptions {
  /** a copy of the pinned pack already on this machine, linked or copied instead of fetched */
  from?: string | undefined;
}

export type PackDownloadState = "present" | "fetched" | "adopted";

export interface DownloadPackReport {
  path: string;
  state: PackDownloadState;
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
    const pack = { path: head.sourcePath, sha256: head.source.sha256 };
    const packState = await this.obtain(head, onHub(head.source), pack, "pack", options.from);
    if (!packState.ok) return packState;
    const report: DownloadPackReport = { path: pack.path, state: packState.value };
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
