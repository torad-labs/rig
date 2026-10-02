import { describe, expect, test } from "bun:test";
import { layoutAt } from "@rig/core";
import { fakePorts } from "@rig/testing";
import { imageDir, publishedImage } from "./published.ts";

const layout = layoutAt("/r");
const record = (cap: string, pushed: boolean) =>
  JSON.stringify({
    image: `registry.example/rig:glm-sm${cap}-x`,
    digest: null,
    head: "glm",
    cap,
    commit: "c",
    engine: "e",
    pushed,
  });

describe("publishedImage", () => {
  test("the one pushed image of the head, with the directory it is recorded in", async () => {
    const p = fakePorts();
    p.fs.put(`${imageDir(layout, "glm", "120")}/image.json`, record("120", true));
    p.fs.put(`${imageDir(layout, "glm", "90")}/image.json`, record("90", false)); // built, never pushed
    p.fs.put(`${imageDir(layout, "glm-mini", "120")}/image.json`, record("120", true)); // another head
    const r = await publishedImage(p.fs, layout, "glm");
    expect(r.ok && r.value).toEqual({
      record: JSON.parse(record("120", true)),
      dir: "/r/local/images/glm-sm120",
    });
  });
  test("none pushed, or one for each of several sms, is refused by name", async () => {
    const p = fakePorts();
    const none = await publishedImage(p.fs, layout, "glm");
    expect(!none.ok && none.message).toBe("no pushed image of glm: run rig image glm --push first");
    p.fs.put(`${imageDir(layout, "glm", "120")}/image.json`, record("120", true));
    p.fs.put(`${imageDir(layout, "glm", "90")}/image.json`, record("90", true));
    const two = await publishedImage(p.fs, layout, "glm");
    expect(!two.ok && two.message).toContain("glm has pushed images for sm_120, sm_90");
  });
});
