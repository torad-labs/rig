// A small, predictable argument parser: `rig <command> [subcommand] [positionals] [--flag[=value]]`.
// Flags are `--name value`, `--name=value` or bare `--name` (true). No abbreviations, no clustering:
// a wizard generating a command line should not have to guess.
export interface Args {
  positionals: string[];
  flags: Record<string, string | boolean>;
}

export function parseArgs(argv: readonly string[]): Args {
  const positionals: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === undefined) break;
    if (arg === "--") {
      positionals.push(...argv.slice(i + 1));
      break;
    }
    if (!arg.startsWith("--")) {
      positionals.push(arg);
      continue;
    }
    const eq = arg.indexOf("=");
    if (eq > 0) {
      flags[arg.slice(2, eq)] = arg.slice(eq + 1);
      continue;
    }
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      flags[arg.slice(2)] = next;
      i++;
    } else {
      flags[arg.slice(2)] = true;
    }
  }
  return { positionals, flags };
}

export const flagStr = (args: Args, name: string): string | undefined => {
  const value = args.flags[name];
  return typeof value === "string" ? value : undefined;
};

export const flagBool = (args: Args, name: string): boolean =>
  args.flags[name] === true || args.flags[name] === "true";

export const flagInt = (args: Args, name: string): number | undefined => {
  const value = flagStr(args, name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : undefined;
};
