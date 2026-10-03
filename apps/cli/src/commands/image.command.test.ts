import { describe, expect, test } from "bun:test";
import { ExitCode, fail, ok } from "@rig/core";
import type { Head } from "@rig/head";
import type { BuildImage, ImageReport } from "@rig/image";
import { FakeLog } from "@rig/testing";
import type { Args } from "../cli/args.ts";
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
/** which half ran, so a --push-proven can be seen not to have built anything */
const halves = (pushed: boolean) => {
  const called: Array<{
    half: "run" | "build" | "prove" | "pushProven";
    options: Record<string, unknown>;
  }> = [];
  const service = {
    run: async (_h: Head, options: Record<string, unknown>) => {
      called.push({ half: "run", options });
      return ok(report(pushed));
    },
    build: async (_h: Head, options: Record<string, unknown>) => {
      called.push({ half: "build", options });
      return ok(report(false));
    },
    prove: async (_h: Head, options: Record<string, unknown>) => {
      called.push({ half: "prove", options });
      return ok(report(false));
    },
    pushProven: async (_h: Head, options: Record<string, unknown>) => {
      called.push({ half: "pushProven", options });
      return ok(report(true));
    },
  } as unknown as BuildImage;
  return { called, service };
};
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

  test("--push-proven pushes without building, --build and --prove are the other two halves, and no two are combined", async () => {
    const proven = halves(false);
    const log = new FakeLog();
    expect(
      await buildImageCommand(
        proven.service,
        async () => ok(head),
        log,
        async () => ok({ name: "rig-glm-5.3-flash-sm120", hashId: "a79f7a77" }),
      ).run({
        positionals: [head.name],
        flags: { "push-proven": "/r/local/images/glm-5.3-flash-sm120/image.json" },
        dashed: [],
      }),
    ).toBe(ExitCode.Ok);
    expect(proven.called).toEqual([
      {
        half: "pushProven",
        options: { receipt: "/r/local/images/glm-5.3-flash-sm120/image.json" },
      },
    ]);
    // it pushed, so the template follows it as a --push does
    expect(log.lines).toContain("info template rig-glm-5.3-flash-sm120: a79f7a77");

    const half = async (flags: Args["flags"]) => {
      const h = halves(false);
      const log_ = new FakeLog();
      const code = await buildImageCommand(h.service, async () => ok(head), log_, null).run({
        positionals: [head.name],
        flags,
        dashed: [],
      });
      return { ...h, code, said: log_.lines.join("\n") };
    };
    // the build half has no card to read: it names the sm, and the one-shot form is what asks for a card
    const building = await half({ build: "/tmp/built.tar", cap: "120", "from-tarball": "/t.tgz" });
    expect(building.code).toBe(ExitCode.Ok);
    expect(building.called).toEqual([
      { half: "build", options: { cap: "120", fromTarball: "/t.tgz", out: "/tmp/built.tar" } },
    ]);
    // the proof half names the card, and the receipt carried to it
    const proving = await half({ prove: "/r/image.json", gpu: "1" });
    expect(proving.code).toBe(ExitCode.Ok);
    expect(proving.called).toEqual([
      { half: "prove", options: { receipt: "/r/image.json", gpu: 1 } },
    ]);
    expect((await half({ prove: "/r/image.json" })).called[0]?.options).toMatchObject({ gpu: 0 });
    const oneShot = await half({ gpu: "1" });
    expect(oneShot.called[0]).toMatchObject({ half: "run", options: { gpu: 1, push: false } });

    // each half is its own run on its own machine, never combined with another or with the one-shot's flags
    for (const [flags, named] of [
      [{ "push-proven": "/r/image.json", push: true }, "--push-proven"],
      [{ "push-proven": "/r/image.json", prove: "/r/image.json" }, "--prove"],
      [{ build: "/tmp/x.tar", cap: "120", prove: "/r/image.json" }, "--build"],
      [{ build: "/tmp/x.tar", cap: "120", push: true }, "--build"],
      [{ build: "/tmp/x.tar", cap: "120", gpu: "0" }, "--build"],
      [{ build: "/tmp/x.tar" }, "--cap"],
      [{ build: true, cap: "120" }, "--build"],
      [{ prove: true }, "--prove"],
      [{ "push-proven": true }, "--push-proven"],
      [{ cap: "120" }, "--cap"],
      [{ prove: "/r/image.json", "from-tarball": "/t.tgz" }, "--prove"],
      [{ prove: "/r/image.json", push: true }, "--prove"],
    ] as const) {
      const refused = await half(flags);
      expect(refused.code).toBe(ExitCode.Usage);
      expect(refused.said).toContain(named);
      expect(refused.called).toEqual([]);
    }
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
