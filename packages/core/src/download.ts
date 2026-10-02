// A pinned file fetched over the network: the download command line as data, and fetchPinned,
// which is the one way rig downloads anything. aria2c with 16 connections a file saturates a rented host's
// link; a home line does as well with curl, which resumes a partial file, and curl is also what
// reads a file:// URL (a mirror, or a build under test), which aria2c refuses. Either writes the
// staging file beside the target, and the target appears only by a rename after its sha256
// matched (artifact.ts says why nothing is written in place).
import { basename, dirname } from "node:path";
import { type Artifact, publishArtifact, stagingPath } from "./artifact.ts";
import type { FileSystem, Hasher, Log, Shell } from "./ports/index.ts";
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

/** 16 connections, aria2c's ceiling for one server; a 503 or 429 from the hub, a stalled or refused connection, retried
 *  every 10 s for 30 tries (aria2c's defaults are 5 tries and no wait, and with no wait a 503 is not retried at all), and
 *  a 404, a wrong pin, failed after 3 */
export function aria2cArgv(request: DownloadRequest): string[] {
  return [
    "aria2c",
    "-x",
    "16",
    "-s",
    "16",
    "-k",
    "16M",
    "--max-tries=30",
    "--retry-wait=10",
    "--max-file-not-found=3",
    "--connect-timeout=30",
    "--timeout=60",
    "--lowest-speed-limit=64K",
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

/** aria2c's lines worth a log: its progress every --summary-interval ("[#0202aa 298MiB/522MiB(57%) CN:16 DL:188MiB
 *  ETA:1s]", as it prints it to a file or a pipe) and what --console-log-level=warn lets through */
const PROGRESS = /^\[#|\[(WARN|ERROR)\]|errorCode/;

/** `url` downloaded to `<target>.part` and renamed over `target.path` only once its sha256 is the
 *  pinned one; bytes that are not are removed and refused, the refusal naming `what`. With a log, aria2c's progress
 *  reaches it as the download runs, each line named by the file */
export async function fetchPinned(
  deps: { shell: Shell; fs: FileSystem; hasher: Hasher; log?: Log },
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
  const { log } = deps;
  const name = basename(target.path);
  const download = await deps.shell.run(argv, {
    timeoutMs: DOWNLOAD_TIMEOUT_MS,
    ...(parallel && log
      ? { onLine: (line: string) => PROGRESS.test(line) && log.info(`${name}: ${line.trim()}`) }
      : {}),
  });
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
