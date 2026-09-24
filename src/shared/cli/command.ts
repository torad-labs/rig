// What every feature exposes to main.ts: a name, a usage line, and a run over parsed args that
// returns the process exit code. Output conventions: human text on stderr through Log, machine
// output (--json) on stdout only. The helpers below are the shape every command shares: the
// head it names, and a Result reported as an exit code, a JSON object or one line.
import type { Head } from "../head/head.ts";
import type { Log } from "../ports/index.ts";
import { ExitCode, type Result } from "../result.ts";
import { type Args, flagBool } from "./args.ts";

export interface Command {
  name: string;
  usage: string;
  run(args: Args): Promise<number>;
}

export type LoadHead = (name: string) => Promise<Result<Head>>;

/** the head a command names, loaded; no name or a head that does not load is the exit code */
export async function withHead(
  name: string | undefined,
  usage: string,
  load: LoadHead,
  log: Log,
  run: (head: Head) => Promise<number>,
): Promise<number> {
  if (!name) {
    log.error(`usage: rig ${usage}`);
    return ExitCode.Usage;
  }
  const head = await load(name);
  if (!head.ok) {
    log.error(head.message);
    return head.code;
  }
  return run(head.value);
}

/** a failure is logged and becomes its exit code; a success is shown */
export function report<T>(log: Log, result: Result<T>, show: (value: T) => void): number {
  if (!result.ok) {
    log.error(result.message);
    return result.code;
  }
  show(result.value);
  return 0;
}

/** `report`, shown as the JSON object with --json and otherwise as the one line `describe` makes */
export function reportLine<T>(
  log: Log,
  args: Args,
  result: Result<T>,
  describe: (value: T) => string,
): number {
  return report(log, result, (value) => {
    if (flagBool(args, "json")) printJson(value);
    else log.info(describe(value));
  });
}

/** `report`, shown only as the JSON object with --json: the log already said what happened */
export function reportJson<T>(log: Log, args: Args, result: Result<T>): number {
  return report(log, result, (value) => {
    if (flagBool(args, "json")) printJson(value);
  });
}

export const printJson = (value: unknown) => console.log(JSON.stringify(value, null, 2));
