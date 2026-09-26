// A small, predictable argument parser: `rig <command> [subcommand] [positionals] [--flag[=value]]`.
// Flags are `--name value`, `--name=value` or bare `--name` (true). No abbreviations, no clustering:
// a wizard generating a command line should not have to guess.
export interface Args {
  positionals: string[];
  flags: Record<string, string | boolean>;
  /** the single-dash tokens before -- (`-gpu 1`): rig's flags take two dashes and the engine's go after --, so main
   *  refuses these rather than let them pass as a head name or a server argument */
  dashed: string[];
}

export function parseArgs(argv: readonly string[]): Args {
  const positionals: string[] = [];
  const flags: Record<string, string | boolean> = {};
  const dashed: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === undefined) break;
    if (arg === "--") {
      positionals.push(...argv.slice(i + 1));
      break;
    }
    if (!arg.startsWith("--")) {
      if (/^-[A-Za-z]/.test(arg)) dashed.push(arg);
      positionals.push(arg);
      continue;
    }
    const eq = arg.indexOf("=");
    if (eq > 0) {
      flags[arg.slice(2, eq)] = arg.slice(eq + 1);
      continue;
    }
    const next = argv[i + 1];
    // a single-dash word is never a value (`--restart -gpu 1`): it stays a token main refuses by name
    if (next !== undefined && !next.startsWith("--") && !/^-[A-Za-z]/.test(next)) {
      flags[arg.slice(2)] = next;
      i++;
    } else {
      flags[arg.slice(2)] = true;
    }
  }
  return { positionals, flags, dashed };
}

export const flagStr = (args: Args, name: string): string | undefined => {
  const value = args.flags[name];
  return typeof value === "string" ? value : undefined;
};

/** true for a bare flag or `true`, false when absent or `false`; any other value is a UsageError: `--json bonsai` took
 *  the head's name as the flag's value, and reading that as false would run the command without it */
export const flagBool = (args: Args, name: string): boolean => {
  const value = args.flags[name];
  if (value === undefined || value === false || value === "false") return false;
  if (value === true || value === "true") return true;
  throw new UsageError(`--${name} takes no value, not ${JSON.stringify(value)}`);
};

/** the flags a command was given that its usage does not name: the usage line is the list of what it takes, so a flag
 *  outside it is refused rather than ignored (`serve --cache-type-k q8_0`, an engine flag before --, would otherwise
 *  serve the tier's own K and say nothing) */
export function unknownFlags(args: Args, usage: string): string[] {
  const takes = new Set((usage.match(/--[a-z][a-z-]*/g) ?? []).map((flag) => flag.slice(2)));
  return Object.keys(args.flags).filter((flag) => !takes.has(flag));
}

/** a flag given a value its command cannot use: main stops the command with its usage (exit 64)
 *  rather than let it run on the default the operator did not ask for (`--gpu 1x` gating on the
 *  card that serves the live head) */
export class UsageError extends Error {}

const flagParsed = (args: Args, name: string, form: RegExp, what: string) => {
  const value = args.flags[name];
  if (value === undefined) return undefined;
  if (typeof value !== "string" || !form.test(value)) {
    const given = typeof value === "string" ? JSON.stringify(value) : "nothing";
    throw new UsageError(`--${name} takes ${what}, not ${given}`);
  }
  return Number(value);
};

/** undefined when absent; a value that is not a whole number is a UsageError */
export const flagInt = (args: Args, name: string): number | undefined =>
  flagParsed(args, name, /^-?\d+$/, "a whole number");

/** undefined when absent; a value that is not a plain decimal (1.5, 2) is a UsageError */
export const flagNumber = (args: Args, name: string): number | undefined =>
  flagParsed(args, name, /^\d+(\.\d+)?$/, "a number");
