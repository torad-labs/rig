// What `rig tag` reads before it tags a release: the driver-only gate passed on this commit, on a fresh machine that
// fetched the prebuilt engine.toml pins from its published URL as a user does. The gate writes it (only for a run
// without --prebuilt, in a checkout with no change git sees, so the commit names what was gated) and removes it when
// such a run fails. v0.1.11 was tagged with a CLI that could not install the pin it shipped with: the fresh-machine
// run on the published pin was a line in a procedure, and a line can be skipped.
//
// The gate runs on a rented box (rig/CLAUDE.md: every GPU test does), so the receipt and its log are carried to the
// machine that tags. The file therefore names its log BESIDE itself, never by the absolute path the box had, which
// would not exist here — `rig tag` opens what the field says, so a receipt naming /root/... would refuse a run that
// passed. A read resolves the name against the receipt's own directory. An absolute log still reads as written, so
// every receipt made before this (v0.1.12's among them) stays valid.
import { basename, dirname, isAbsolute, join } from "node:path";
import type { FileSystem, Layout } from "@rig/core";

export interface E2eReceipt {
  /** HEAD of the checkout the gate packed */
  commit: string;
  /** the container image the machine started from */
  base: string;
  head: string | null;
  /** when the gate passed, ISO 8601 */
  passed: string;
  /** the run's log: a name beside the receipt as written, resolved to a path by `readReceipt` */
  log: string;
}

/** where the receipt for `commit` and its log are kept */
export function receiptPaths(layout: Layout, commit: string): { receipt: string; log: string } {
  const dir = join(layout.localDir, "release");
  return { receipt: join(dir, `e2e-${commit}.json`), log: join(dir, `e2e-${commit}.log`) };
}

/** what the gate writes in the receipt's `log`: the log's name, so the pair is readable wherever it is carried */
export function receiptLogName(logPath: string): string {
  return basename(logPath);
}

/** the receipt for `commit` with its `log` resolved to a path on THIS machine, or null when there is none or it
 *  names another commit. A relative name resolves against the receipt's directory; an absolute one is kept. */
export async function readReceipt(
  fs: FileSystem,
  layout: Layout,
  commit: string,
): Promise<E2eReceipt | null> {
  const paths = receiptPaths(layout, commit);
  if (!(await fs.exists(paths.receipt))) return null;
  const read = JSON.parse(await fs.readText(paths.receipt)) as Partial<E2eReceipt>;
  if (read.commit !== commit || typeof read.log !== "string") return null;
  const log = isAbsolute(read.log) ? read.log : join(dirname(paths.receipt), read.log);
  return { ...(read as E2eReceipt), log };
}
