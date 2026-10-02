// The engine's argument table (common/arg.cpp) declares each option as add_opt(common_arg({every
// spelling}, …)); a boolean option declares its on and off spellings as two groups. A flag rig
// renders is refused in a head's free-form lists under EVERY spelling the table accepts, so the
// denominator of that refusal is derived from the table, never from rig's own spelling.

/** every spelling the table accepts for each flag in `flags`; a flag the table does not declare throws */
export function argAliases(argCpp: string, flags: readonly string[]): Record<string, string[]> {
  const groups: string[][] = [];
  for (const m of argCpp.matchAll(/common_arg\(\s*((?:\{[^{}]*\}\s*,\s*)+)/g)) {
    const names = [...m[1]!.matchAll(/"(-[^"]*)"/g)].map((n) => n[1]!);
    if (names.length) groups.push(names);
  }
  const out: Record<string, string[]> = {};
  for (const flag of flags) {
    const spellings = new Set(groups.filter((g) => g.includes(flag)).flat());
    if (spellings.size === 0) throw new Error(`${flag} is not declared in the engine's arg table`);
    out[flag] = [...spellings].sort();
  }
  return out;
}
