// `rig tag`: the release tag, v<the CLI's version> as release.yml requires, on this checkout's HEAD, pushed to origin.
// Refused unless HEAD is origin/main (a release is tagged where its PR landed) with no change git sees, and the
// driver-only gate passed on that commit fetching the published pin as a user does: its receipt (e2e-receipt.ts),
// written by `rig e2e` in this checkout.
import type { FileSystem, Layout, Log, Shell } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import { readReceipt, receiptPaths } from "./e2e-receipt.ts";

export interface TagReleaseReport {
  tag: string;
  commit: string;
  /** the receipt the tag was allowed by */
  receipt: string;
}

export interface TagReleaseDeps {
  shell: Shell;
  fs: FileSystem;
  log: Log;
}

const GIT_TIMEOUT_MS = 120_000;

export class TagRelease {
  constructor(
    private readonly deps: TagReleaseDeps,
    private readonly layout: Layout,
  ) {}

  async run(): Promise<Result<TagReleaseReport>> {
    const head = await this.git("rev-parse", "HEAD");
    if (!head.ok) return head;
    const commit = head.value.trim();
    const short = commit.slice(0, 7);
    const status = await this.git("status", "--porcelain");
    if (!status.ok) return status;
    if (status.value.trim() !== "")
      return fail(
        ExitCode.Failure,
        `${this.layout.root} has changes git sees: a tag names a commit, so the gate's must be all there is`,
      );
    const fetched = await this.git("fetch", "origin", "main");
    if (!fetched.ok) return fetched;
    const main = await this.git("rev-parse", "origin/main");
    if (!main.ok) return main;
    if (main.value.trim() !== commit)
      return fail(
        ExitCode.Failure,
        `HEAD ${short} is not origin/main ${main.value.trim().slice(0, 7)}: a release is tagged on what main holds, once its checks pass`,
      );
    const pkg = await this.git("show", `${commit}:apps/cli/package.json`);
    if (!pkg.ok) return pkg;
    const version = (JSON.parse(pkg.value) as { version?: unknown }).version;
    if (typeof version !== "string" || !/^\d+\.\d+\.\d+$/.test(version))
      return fail(ExitCode.Failure, `apps/cli/package.json at ${short} names no version`);
    const tag = `v${version}`;
    const remote = await this.git("ls-remote", "--tags", "origin", `refs/tags/${tag}`);
    if (!remote.ok) return remote;
    if (remote.value.trim() !== "") return fail(ExitCode.Failure, `${tag} is already on origin`);

    const { fs } = this.deps;
    const receipt = await readReceipt(fs, this.layout, commit);
    if (!receipt)
      return fail(
        ExitCode.Failure,
        `no fresh-machine e2e passed on ${short}: run \`rig e2e <pack.gguf>\` in this checkout without --prebuilt, so a machine with only the driver installs the published pin as a user does`,
      );
    if (!(await fs.exists(receipt.log)) || !/^== PASS$/m.test(await fs.readText(receipt.log)))
      return fail(ExitCode.Failure, `${receipt.log} holds no passing run`);

    // a tag made here by a run whose push failed is pushed again; one naming another commit is never moved
    const local = await this.deps.shell.run(
      ["git", "-C", this.layout.root, "rev-parse", "-q", "--verify", `refs/tags/${tag}^{commit}`],
      { timeoutMs: GIT_TIMEOUT_MS },
    );
    const tagged = local.code === 0 ? local.stdout.trim() : null;
    if (tagged !== null && tagged !== commit)
      return fail(
        ExitCode.Failure,
        `a local ${tag} names ${tagged.slice(0, 7)}, not HEAD ${short}`,
      );
    if (tagged === null) {
      const made = await this.git("tag", tag, commit);
      if (!made.ok) return made;
    }
    const pushed = await this.git("push", "origin", `refs/tags/${tag}`);
    if (!pushed.ok) return pushed;
    this.deps.log.info(`${tag} on ${short}, allowed by ${receipt.log}`);
    return ok({ tag, commit, receipt: receiptPaths(this.layout, commit).receipt });
  }

  /** a git command in this checkout that must succeed: its stdout, or a failure naming it */
  private async git(...args: string[]): Promise<Result<string>> {
    const run = await this.deps.shell.run(["git", "-C", this.layout.root, ...args], {
      timeoutMs: GIT_TIMEOUT_MS,
    });
    if (run.code !== 0)
      return fail(
        ExitCode.Failure,
        `git ${args.slice(0, 2).join(" ")} failed: ${run.stderr.trim().split("\n").slice(-3).join("\n")}`,
      );
    return ok(run.stdout);
  }
}
