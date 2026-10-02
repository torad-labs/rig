import { describe, expect, test } from "bun:test";
import { ConsoleLog } from "./console-log.ts";

describe("ConsoleLog", () => {
  test("to a file or a pipe each line starts with its UTC time; to a terminal it is bare", () => {
    const lines: string[] = [];
    new ConsoleLog("rig", true, (line) => lines.push(line)).warn("slow");
    new ConsoleLog("rig", false, (line) => lines.push(line)).info("== fetch");
    expect(lines[0]).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ rig: WARN slow$/);
    expect(lines[1]).toBe("rig: == fetch");
  });
});
