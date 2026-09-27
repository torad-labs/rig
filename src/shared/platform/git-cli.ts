import type { Git, Shell } from "../ports/index.ts";

export class GitCli implements Git {
  constructor(private readonly shell: Shell) {}
  async revParse(dir: string, ref: string) {
    const result = await this.shell.run(["git", "-C", dir, "rev-parse", ref]);
    return result.code === 0 ? result.stdout.trim() : null;
  }
  async isClean(dir: string) {
    const result = await this.shell.run([
      "git",
      "-C",
      dir,
      "status",
      "--porcelain",
      "--untracked-files=normal",
    ]);
    return result.code === 0 && result.stdout.trim() === "";
  }
  async fetchCommit(dir: string, repo: string, sha: string) {
    for (const step of [
      ["init", "-q"],
      ["remote", "add", "origin", repo],
      ["fetch", "-q", "--depth", "1", "origin", sha],
      ["checkout", "-q", "FETCH_HEAD"],
    ]) {
      const result = await this.shell.run(["git", "-C", dir, ...step], { timeoutMs: 600_000 });
      if (result.code !== 0)
        throw new Error(`git ${step.join(" ")} in ${dir}: ${result.stderr.trim()}`);
    }
  }
}
