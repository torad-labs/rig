import type { Process, RunOptions, RunResult, Shell, SpawnOptions } from "../ports/index.ts";

export class BunShell implements Shell {
  async run(cmd: readonly string[], opts: RunOptions = {}): Promise<RunResult> {
    const child = Bun.spawn([...cmd], {
      ...(opts.cwd ? { cwd: opts.cwd } : {}),
      env: opts.env ? { ...process.env, ...opts.env } : process.env,
      stdin: opts.stdin === undefined ? "ignore" : new TextEncoder().encode(opts.stdin),
      stdout: "pipe",
      stderr: "pipe",
    });
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (opts.timeoutMs) timer = setTimeout(() => child.kill("SIGKILL"), opts.timeoutMs);
    const [stdout, stderr, code] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ]);
    if (timer) clearTimeout(timer);
    return { code, stdout, stderr };
  }
  spawn(cmd: readonly string[], opts: SpawnOptions = {}): Process {
    const child = Bun.spawn([...cmd], {
      ...(opts.cwd ? { cwd: opts.cwd } : {}),
      env: opts.env ? { ...process.env, ...opts.env } : process.env,
      stdin: "ignore",
      stdout: opts.stdoutPath ? Bun.file(opts.stdoutPath) : "inherit",
      stderr: opts.stderrPath
        ? Bun.file(opts.stderrPath)
        : opts.stdoutPath
          ? Bun.file(opts.stdoutPath)
          : "inherit",
    });
    if (opts.detached) child.unref();
    return {
      pid: child.pid,
      kill: (signal = "SIGTERM") => child.kill(signal),
      exited: child.exited,
    };
  }
  async which(name: string): Promise<string | null> {
    return Bun.which(name);
  }
}
