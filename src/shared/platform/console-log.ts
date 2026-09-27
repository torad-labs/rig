import type { Log } from "../ports/index.ts";

export class ConsoleLog implements Log {
  constructor(private readonly prefix = "rig") {}
  info(msg: string) {
    console.error(`${this.prefix}: ${msg}`);
  }
  warn(msg: string) {
    console.error(`${this.prefix}: WARN ${msg}`);
  }
  error(msg: string) {
    console.error(`${this.prefix}: ERROR ${msg}`);
  }
}
