import { describe, expect, test } from "bun:test";
import { basename } from "node:path";
import { layoutAt } from "@rig/core";
import { fakePorts } from "@rig/testing";
import { receiptPaths } from "./e2e-receipt.ts";
import { TagRelease } from "./tag-release.service.ts";

const COMMIT = "54d558e0a1b2c3d4e5f60718293a4b5c6d7e8f90";
const OTHER = "f846aa1000000000000000000000000000000000";
const layout = layoutAt("/r");
const paths = receiptPaths(layout, COMMIT);

/** a release checkout at COMMIT, origin/main there too, the CLI at 0.1.13, no v0.1.13 anywhere, and the gate's
 *  receipt for COMMIT unless `receipt` is false; `git` overrides one command's answer */
function setup(
  opts: { receipt?: boolean; git?: Record<string, { code?: number; stdout: string }> } = {},
) {
  const p = fakePorts();
  const answers: Record<string, { code?: number; stdout: string }> = {
    "rev-parse HEAD": { stdout: `${COMMIT}\n` },
    "status --porcelain": { stdout: "" },
    "fetch origin main": { stdout: "" },
    "rev-parse origin/main": { stdout: `${COMMIT}\n` },
    [`show ${COMMIT}:apps/cli/package.json`]: { stdout: '{ "version": "0.1.13" }\n' },
    "ls-remote --tags origin refs/tags/v0.1.13": { stdout: "" },
    "rev-parse -q --verify refs/tags/v0.1.13^{commit}": { code: 1, stdout: "" },
    [`tag v0.1.13 ${COMMIT}`]: { stdout: "" },
    "push origin refs/tags/v0.1.13": { stdout: "" },
    ...opts.git,
  };
  p.shell.on(/^git -C \/r (.+)$/, (cmd) => {
    const answer = answers[cmd.slice(3).join(" ")];
    return answer
      ? { code: answer.code ?? 0, stdout: answer.stdout, stderr: "" }
      : { code: 128, stdout: "", stderr: `no answer for ${cmd.join(" ")}` };
  });
  if (opts.receipt !== false) {
    p.fs.put(paths.log, "== decode\n== PASS\n");
    p.fs.put(
      paths.receipt,
      JSON.stringify({
        commit: COMMIT,
        base: "ubuntu:22.04",
        head: null,
        passed: "2026-10-03T23:00:00.000Z",
        log: paths.log,
      }),
    );
  }
  const changes = () =>
    p.shell.calls.filter((c) => c[3] === "tag" || c[3] === "push").map((c) => c.slice(3).join(" "));
  return { p, tag: new TagRelease(p, layout), changes };
}

describe("tag: the release tag, only on a commit the fresh-machine e2e passed on", () => {
  test("tags HEAD as the CLI's version and pushes the tag, naming the receipt it read", async () => {
    const { tag, changes } = setup();
    const r = await tag.run();
    expect(r.ok && r.value).toEqual({ tag: "v0.1.13", commit: COMMIT, receipt: paths.receipt });
    expect(changes()).toEqual([`tag v0.1.13 ${COMMIT}`, "push origin refs/tags/v0.1.13"]);
  });
  test("no receipt for the commit: refused, naming the run that makes one, and nothing is tagged", async () => {
    const { tag, changes } = setup({ receipt: false });
    const r = await tag.run();
    expect(!r.ok && r.message).toContain(`no fresh-machine e2e passed on ${COMMIT.slice(0, 7)}`);
    expect(!r.ok && r.message).toContain("rig e2e <pack.gguf>");
    expect(changes()).toEqual([]);
  });
  // The gate runs on a rented box, so the receipt and its log are carried here. The receipt names its
  // log beside itself, and tag resolves it against the receipt's directory, never against the path the
  // box had. A pair that travelled together therefore tags; one whose log did not travel refuses.
  test("a receipt carried from the box, naming its log beside itself, tags; the same receipt without its log refuses by the path here", async () => {
    const carried = setup();
    carried.p.fs.put(
      paths.receipt,
      JSON.stringify({
        commit: COMMIT,
        base: "ubuntu:22.04",
        head: null,
        passed: "2026-10-03T23:00:00.000Z",
        log: basename(paths.log),
      }),
    );
    const r = await carried.tag.run();
    expect(r.ok && r.value.tag).toBe("v0.1.13");
    expect(carried.changes()).toEqual([`tag v0.1.13 ${COMMIT}`, "push origin refs/tags/v0.1.13"]);

    const alone = setup();
    alone.p.fs.put(paths.receipt, JSON.stringify({ commit: COMMIT, log: basename(paths.log) }));
    await alone.p.fs.remove(paths.log);
    const missing = await alone.tag.run();
    expect(!missing.ok && missing.message).toContain(`${paths.log} holds no passing run`);
    expect(alone.changes()).toEqual([]);

    // and a carried receipt whose log travelled but holds no pass is refused, not tagged
    const failed = setup();
    failed.p.fs.put(paths.receipt, JSON.stringify({ commit: COMMIT, log: basename(paths.log) }));
    failed.p.fs.put(paths.log, "== rig build\n");
    const nope = await failed.tag.run();
    expect(!nope.ok && nope.message).toContain("holds no passing run");
    expect(failed.changes()).toEqual([]);
  });
  test("a receipt whose log holds no pass, or that names another commit, is no receipt", async () => {
    const failed = setup();
    failed.p.fs.put(paths.log, "== rig build\n");
    const r = await failed.tag.run();
    expect(!r.ok && r.message).toContain(`${paths.log} holds no passing run`);
    expect(failed.changes()).toEqual([]);
    const moved = setup();
    moved.p.fs.put(paths.receipt, JSON.stringify({ commit: OTHER, log: paths.log }));
    const other = await moved.tag.run();
    expect(!other.ok && other.message).toContain("no fresh-machine e2e passed");
    expect(moved.changes()).toEqual([]);
  });
  test("a checkout with changes git sees, or a HEAD that is not origin/main, is refused before any receipt is read", async () => {
    for (const [git, reason] of [
      [{ "status --porcelain": { stdout: " M README.md\n" } }, "has changes git sees"],
      [{ "rev-parse origin/main": { stdout: `${OTHER}\n` } }, "is not origin/main"],
    ] as const) {
      const { tag, changes } = setup({ git });
      const r = await tag.run();
      expect(!r.ok && r.message).toContain(reason);
      expect(changes()).toEqual([]);
    }
  });
  test("a version already tagged on origin is refused; a local tag at HEAD (a push that failed) is pushed, one elsewhere refused", async () => {
    const published = setup({
      git: {
        "ls-remote --tags origin refs/tags/v0.1.13": { stdout: `${COMMIT}\trefs/tags/v0.1.13\n` },
      },
    });
    const r = await published.tag.run();
    expect(!r.ok && r.message).toContain("v0.1.13 is already on origin");
    expect(published.changes()).toEqual([]);
    const retried = setup({
      git: { "rev-parse -q --verify refs/tags/v0.1.13^{commit}": { stdout: `${COMMIT}\n` } },
    });
    expect((await retried.tag.run()).ok).toBe(true);
    expect(retried.changes()).toEqual(["push origin refs/tags/v0.1.13"]);
    const elsewhere = setup({
      git: { "rev-parse -q --verify refs/tags/v0.1.13^{commit}": { stdout: `${OTHER}\n` } },
    });
    const refused = await elsewhere.tag.run();
    expect(!refused.ok && refused.message).toContain(`a local v0.1.13 names ${OTHER.slice(0, 7)}`);
    expect(elsewhere.changes()).toEqual([]);
  });
});
