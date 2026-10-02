import { describe, expect, test } from "bun:test";
import { ExitCode, layoutAt } from "@rig/core";
import { loadEngine } from "@rig/engine";
import { loadHead } from "@rig/head";
import { ServeHead } from "@rig/serve";
import { fakePorts, putHead, repoRoot } from "@rig/testing";
import { serveHeadCommand } from "./serve.command.ts";

async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/opt/rig");
  putHead(p.fs, "/opt/rig", await Bun.file(`${repoRoot}/heads/bonsai-2-27b/head.toml`).text());
  p.fs.put("/opt/rig/engine/engine.toml", await Bun.file(`${repoRoot}/engine/engine.toml`).text());
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  return { p, head: head.value, uc: new ServeHead(p, engine.value) };
}

describe("serve command", () => {
  test("a flag serve renders, after --, is refused before anything starts: it would win over the checked plan", async () => {
    const { p, head, uc } = await setup();
    const serve = serveHeadCommand(uc, async () => ({ ok: true, value: head }), p.log);
    const code = await serve.run({
      positionals: ["bonsai-2-27b", "-ctk", "q5_1", "--ctx_size", "1", "--temp", "0"],
      flags: {},
      dashed: [],
    });
    expect(code).toBe(ExitCode.Usage);
    expect(p.log.lines.join("\n")).toContain("REFUSING: -ctk, --ctx_size after --");
    expect(p.shell.spawned).toEqual([]);
  });
});
