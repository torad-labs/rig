import type { Clock } from "../ports/index.ts";

export class SystemClock implements Clock {
  now() {
    return Date.now();
  }
  sleep(ms: number) {
    return new Promise<void>((wake) => setTimeout(wake, ms));
  }
}

/** Human output on stderr, so stdout stays a machine-readable channel for --json commands. */
