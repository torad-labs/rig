import { expect, test } from "bun:test";
import { Glob } from "bun";
import { flagBool, flagInt, flagNumber, parseArgs, UsageError, unknownFlags } from "./args.ts";

test("positionals, --k v, --k=v, bare --k and -- terminator", () => {
  const a = parseArgs([
    "up",
    "bonsai-2-27b",
    "--gpu",
    "1",
    "--cache-ram=8192",
    "--restart",
    "--",
    "--not-a-flag",
  ]);
  expect(a.positionals).toEqual(["up", "bonsai-2-27b", "--not-a-flag"]);
  expect(flagInt(a, "gpu")).toBe(1);
  expect(flagInt(a, "cache-ram")).toBe(8192);
  expect(flagBool(a, "restart")).toBe(true);
  expect(flagInt(a, "missing")).toBeUndefined();
});

test("a numeric flag given a value it cannot use is a usage error, never its default", () => {
  const a = parseArgs(["--gpu", "1x", "--slots=", "--ctx", "245k", "--jobs", "--max-price", "0.5"]);
  expect(() => flagInt(a, "gpu")).toThrow(new UsageError('--gpu takes a whole number, not "1x"'));
  expect(() => flagInt(a, "slots")).toThrow('--slots takes a whole number, not ""');
  expect(() => flagInt(a, "ctx")).toThrow(UsageError);
  expect(() => flagInt(a, "jobs")).toThrow("--jobs takes a whole number, not nothing");
  expect(flagNumber(a, "max-price")).toBe(0.5);
  expect(() => flagNumber(parseArgs(["--max-price", "NaN"]), "max-price")).toThrow(UsageError);
  expect(() => flagInt(parseArgs(["--gpu", "1e3"]), "gpu")).toThrow(UsageError);
  expect(flagInt(parseArgs(["--gpu", "-1"]), "gpu")).toBe(-1);
});

test("a flag outside a command's usage is named, whatever its form", () => {
  const usage = "serve <head> [--gpu N] [--cache-ram MiB] [--plan] [-- extra llama-server args]";
  expect(unknownFlags(parseArgs(["h", "--gpu", "1", "--plan"]), usage)).toEqual([]);
  expect(unknownFlags(parseArgs(["h", "--cache-type-k", "q8_0", "--gpu=1"]), usage)).toEqual([
    "cache-type-k",
  ]);
  expect(unknownFlags(parseArgs(["h", "--", "--cache-type-k", "q8_0"]), usage)).toEqual([]); // the engine's, after --
});

test("a single-dash token before -- is set aside by name; after --, and as a flag's value, it is not", () => {
  expect(parseArgs(["up", "bonsai-2-27b", "-gpu", "1"]).dashed).toEqual(["-gpu"]);
  expect(parseArgs(["serve", "h", "--", "-ctk", "q8_0"]).dashed).toEqual([]);
  expect(parseArgs(["serve", "h", "--gpu", "-1"]).dashed).toEqual([]); // a value, whatever it looks like
});

test("every flag a command reads is one its usage names, so main's refusal never drops a real one", async () => {
  const root = `${import.meta.dir}/../../features`;
  let files = 0;
  for await (const file of new Glob("**/*.command.ts").scan(root)) {
    const source = await Bun.file(`${root}/${file}`).text();
    // the usage string main checks, not the file: an error message naming a flag is not the usage
    const literal = /const USAGE =\s*"([^"]*)"/.exec(source)?.[1];
    expect(literal, `${file} declares no const USAGE string`).toBeDefined();
    const usage = new Set([...(literal ?? "").matchAll(/--([a-z][a-z-]*)/g)].map((m) => m[1]));
    const reads = [
      ...[...source.matchAll(/\(args,\s*"([a-z][a-z-]*)"/g)].map((m) => m[1]),
      ...[...source.matchAll(/args\.flags(?:\.([a-z]+)|\["([a-z][a-z-]*)"\])/g)].map(
        (m) => m[1] ?? m[2],
      ),
    ];
    if (/\breport(Line|Json)\(/.test(source)) reads.push("json"); // command.ts's helpers read --json for it
    expect({ file, missing: reads.filter((flag) => !usage.has(flag)) }).toEqual({
      file,
      missing: [],
    });
    files++;
  }
  expect(files).toBeGreaterThanOrEqual(11);
});

test("a flag never takes a single-dash word as its value, and a boolean flag refuses any value but true or false", () => {
  const args = parseArgs(["up", "bonsai-2-27b", "--restart", "-gpu", "1"]);
  expect(args.dashed).toEqual(["-gpu"]);
  expect(args.flags.restart).toBe(true);
  expect(parseArgs(["serve", "h", "--gpu", "-1"]).flags.gpu).toBe("-1"); // a negative number is a value
  expect(flagBool(parseArgs(["x", "--json"]), "json")).toBe(true);
  expect(flagBool(parseArgs(["x", "--json=false"]), "json")).toBe(false);
  expect(() => flagBool(parseArgs(["fetch", "--json", "bonsai-2-27b"]), "json")).toThrow(
    UsageError,
  );
});
