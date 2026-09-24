import { expect, test } from "bun:test";
import { flagBool, flagInt, parseArgs } from "./args.ts";

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
