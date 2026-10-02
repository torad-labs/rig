import { describe, expect, test } from "bun:test";
import { ExitCode, fail, ok } from "@rig/core";
import type { Head } from "@rig/head";
import type { BuildImage, ImageReport } from "@rig/image";
import { FakeLog } from "@rig/testing";
import { buildImageCommand } from "./image.command.ts";

const head = { name: "glm-5.3-flash" } as Head;
const report = (pushed: boolean): ImageReport => ({
  image: "registry.torad.ai/rig:glm-5.3-flash-sm120-abc1234-0e11d35a",
  digest: pushed ? "registry.torad.ai/rig@sha256:aa" : null,
  head: head.name,
  cap: "120",
  commit: "abc1234",
  engine: "e11e67c29",
  pushed,
});
const image = (pushed: boolean) =>
  ({ run: async () => ok(report(pushed)) }) as unknown as BuildImage;
const run = (
  pushed: boolean,
  publish: Parameters<typeof buildImageCommand>[3],
  log = new FakeLog(),
) =>
  buildImageCommand(image(pushed), async () => ok(head), log, publish).run({
    positionals: [head.name],
    flags: pushed ? { push: true } : {},
    dashed: [],
  });

describe("image", () => {
  test("a push is followed by the head's template on the image it pushed; a build alone saves none", async () => {
    const saved: string[] = [];
    const publish = async (h: Head) => {
      saved.push(h.name);
      return ok({ name: "rig-glm-5.3-flash-sm120", hashId: "a79f7a77" });
    };
    const log = new FakeLog();
    expect(await run(true, publish, log)).toBe(ExitCode.Ok);
    expect(saved).toEqual(["glm-5.3-flash"]);
    expect(log.lines).toContain("info template rig-glm-5.3-flash-sm120: a79f7a77");
    expect(await run(false, publish)).toBe(ExitCode.Ok);
    expect(saved).toEqual(["glm-5.3-flash"]);
  });

  test("a template that is not saved fails the push it follows, and says the image is up", async () => {
    const log = new FakeLog();
    const code = await run(true, async () => fail(ExitCode.Failure, "vast answered 401"), log);
    expect(code).toBe(ExitCode.Failure);
    expect(log.lines.join("\n")).toContain(
      "pushed registry.torad.ai/rig@sha256:aa, but its template was not saved: vast answered 401",
    );
  });
});
