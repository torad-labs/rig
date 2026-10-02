import type { Log } from "@rig/core";

// rig's lines on stderr. Written to a file or a pipe (a box's /var/log/rig/up.log, a script's capture) each line starts
// with its UTC time to the second, so a boot's phases, a download's rate and a hash's length can be read off its log;
// a terminal gets the bare line.
export class ConsoleLog implements Log {
  constructor(
    private readonly prefix = "rig",
    private readonly stamped = !process.stderr.isTTY,
    private readonly write: (line: string) => void = (line) => console.error(line),
  ) {}
  info(msg: string) {
    this.write(this.line(msg));
  }
  warn(msg: string) {
    this.write(this.line(`WARN ${msg}`));
  }
  error(msg: string) {
    this.write(this.line(`ERROR ${msg}`));
  }
  private line(msg: string): string {
    const head = `${this.prefix}: ${msg}`;
    return this.stamped ? `${new Date().toISOString().slice(0, 19)}Z ${head}` : head;
  }
}
