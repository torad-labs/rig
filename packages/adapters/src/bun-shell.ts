import type { Process, RunOptions, RunResult, Shell, SpawnOptions } from "@rig/core";

/** the whole text of `stream`, handing each line to `onLine` as it completes, the last one when the stream ends */
async function readLines(
  stream: ReadableStream<Uint8Array>,
  onLine: (line: string) => void,
): Promise<string> {
  const decoder = new TextDecoder();
  let text = "";
  let pending = "";
  for await (const chunk of stream) {
    const piece = decoder.decode(chunk, { stream: true });
    text += piece;
    const lines = (pending + piece).split(/\r?\n|\r/);
    pending = lines.pop() ?? "";
    for (const line of lines) onLine(line);
  }
  const rest = pending + decoder.decode();
  text += rest.slice(pending.length);
  if (rest) onLine(rest);
  return text;
}

export class BunShell implements Shell {
  /** a timed-out run gets a TERM, and a KILL this long after it if it is still running */
  private readonly killAfterMs: number;
  constructor(opts: { killAfterMs?: number } = {}) {
    this.killAfterMs = opts.killAfterMs ?? 10_000;
  }
  async run(cmd: readonly string[], opts: RunOptions = {}): Promise<RunResult> {
    let child: ReturnType<typeof Bun.spawn<"ignore" | Uint8Array, "pipe", "pipe">>;
    try {
      child = Bun.spawn([...cmd], {
        ...(opts.cwd ? { cwd: opts.cwd } : {}),
        env: opts.env ? { ...process.env, ...opts.env } : process.env,
        stdin: opts.stdin === undefined ? "ignore" : new TextEncoder().encode(opts.stdin),
        stdout: "pipe",
        stderr: "pipe",
      });
    } catch (e) {
      // a machine without the tool (no systemctl in a container, no loginctl): the shell's 127, which
      // every caller already handles as a failed run, not an exception out of the command
      if ((e as { code?: string }).code === "ENOENT") {
        return { code: 127, stdout: "", stderr: `${cmd[0]}: command not found` };
      }
      throw e;
    }
    // TERM first: a tool that starts helpers of its own stops them on a TERM, and a KILL leaves them running (nsys's
    // agent, killed with its launcher at a roofline timeout on 2026-09-27, ran on under systemd --user)
    let timer: ReturnType<typeof setTimeout> | undefined;
    let kill: ReturnType<typeof setTimeout> | undefined;
    if (opts.timeoutMs)
      timer = setTimeout(() => {
        child.kill("SIGTERM");
        kill = setTimeout(() => child.kill("SIGKILL"), this.killAfterMs);
      }, opts.timeoutMs);
    const read = (stream: ReadableStream<Uint8Array>, name: "stdout" | "stderr") => {
      const onLine = opts.onLine;
      return onLine ? readLines(stream, (line) => onLine(line, name)) : new Response(stream).text();
    };
    const [stdout, stderr, code] = await Promise.all([
      read(child.stdout, "stdout"),
      read(child.stderr, "stderr"),
      child.exited,
    ]);
    clearTimeout(timer);
    clearTimeout(kill);
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
