// A pinned file fetched over the network: the download command line as data, and fetchPinned,
// which is the one way rig downloads anything. aria2c with 8 connections saturates a rented host's
// link; a home line does as well with curl, which resumes a partial file, and curl is also what
// reads a file:// URL (a mirror, or a build under test), which aria2c refuses. Either writes the
// staging file beside the target, and the target appears only by a rename after its sha256
// matched (artifact.ts says why nothing is written in place).
import { basename, dirname } from "node:path";
import { type Artifact, publishArtifact, stagingPath } from "./artifact.ts";
import type { FileSystem, Hasher, Shell } from "./ports/index.ts";
import { ExitCode, fail, ok, type Result } from "./result.ts";

export interface DownloadRequest {
  url: string;
  /** the directory the file lands in */
  dir: string;
  /** the staging file's name inside `dir`, "<file>.part" */
  partName: string;
  /** the staging file's full path */
  partPath: string;
}

export const DOWNLOAD_TIMEOUT_MS = 6 * 3_600_000;

export function aria2cArgv(request: DownloadRequest): string[] {
  return [
    "aria2c",
    "-x",
    "8",
    "-s",
    "8",
    "-k",
    "16M",
    "--file-allocation=none",
    "-c",
    "--console-log-level=warn",
    "--summary-interval=30",
    "-d",
    request.dir,
    "-o",
    request.partName,
    request.url,
  ];
}

export function curlArgv(request: DownloadRequest): string[] {
  return ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", request.partPath, request.url];
}

/** `url` downloaded to `<target>.part` and renamed over `target.path` only once its sha256 is the
 *  pinned one; bytes that are not are removed and refused, the refusal naming `what` */
export async function fetchPinned(
  deps: { shell: Shell; fs: FileSystem; hasher: Hasher },
  url: string,
  target: Artifact,
  what: string,
): Promise<Result<void>> {
  const dir = dirname(target.path);
  await deps.fs.mkdirp(dir);
  const request = {
    url,
    dir,
    partName: `${basename(target.path)}.part`,
    partPath: stagingPath(target.path, "part"),
  };
  const parallel = /^https?:\/\//.test(url) && (await deps.shell.which("aria2c")) !== null;
  const argv = parallel ? aria2cArgv(request) : curlArgv(request);
  const download = await deps.shell.run(argv, { timeoutMs: DOWNLOAD_TIMEOUT_MS });
  if (download.code !== 0) {
    const lastLine = download.stderr.trim().split("\n").at(-1) ?? "";
    return fail(
      ExitCode.Failure,
      `download failed (${argv[0]} exit ${download.code}): ${lastLine}`,
    );
  }

  const state = await publishArtifact(deps.fs, deps.hasher, request.partPath, target);
  if (state === "missing") return fail(ExitCode.Failure, "the download left no file");
  if (typeof state === "object" && "unreadable" in state)
    return fail(
      ExitCode.Failure,
      `the download could not be read back (${state.unreadable}): ${request.partPath}`,
    );
  if (state !== "ok") {
    return fail(
      ExitCode.Failure,
      `the downloaded file is not the pinned ${what} (sha256 differs) — removed`,
    );
  }
  return ok(undefined);
}
