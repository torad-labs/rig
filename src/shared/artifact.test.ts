import { describe, expect, test } from "bun:test";
import { fakePorts, sha256Of } from "../../test/fakes/index.ts";
import { artifactProblem, checkArtifact, publishArtifact } from "./artifact.ts";

const BYTES = "the pinned bytes";
const pin = { path: "/r/local/packs/x.gguf", sha256: sha256Of(BYTES) };

describe("checkArtifact", () => {
  test("ok, missing and mismatch, by presence and hash", async () => {
    const p = fakePorts();
    expect(await checkArtifact(p.fs, p.hasher, pin)).toBe("missing");
    p.fs.put(pin.path, BYTES);
    expect(await checkArtifact(p.fs, p.hasher, pin)).toBe("ok");
    p.fs.put(pin.path, "other bytes");
    expect(await checkArtifact(p.fs, p.hasher, pin)).toBe("mismatch");
  });
  // "missing" is answered by fetching or deriving over the path, so an error that is not ENOENT
  // must never read as one: an unreadable directory, an I/O error, a file open refuses
  test("an EACCES or EIO reads as unreadable, naming the code, never as missing", async () => {
    const p = fakePorts();
    p.fs.deny(pin.path, "EIO");
    expect(await checkArtifact(p.fs, p.hasher, pin)).toEqual({ unreadable: "EIO" });
    p.fs.denied.clear();
    p.fs.put(pin.path, BYTES);
    p.hasher.unreadable.set(pin.path, "EACCES");
    expect(await checkArtifact(p.fs, p.hasher, pin)).toEqual({ unreadable: "EACCES" });
    // a link whose target is gone: lstat sees it, open says ENOENT — that one is missing
    p.hasher.unreadable.set(pin.path, "ENOENT");
    expect(await checkArtifact(p.fs, p.hasher, pin)).toBe("missing");
  });
  test("publishing an unreadable staged file reports it, never a mismatch or a rename", async () => {
    const p = fakePorts();
    p.fs.put(`${pin.path}.part`, BYTES);
    p.hasher.unreadable.set(`${pin.path}.part`, "EIO");
    expect(await publishArtifact(p.fs, p.hasher, `${pin.path}.part`, pin)).toEqual({
      unreadable: "EIO",
    });
    expect(p.fs.renames).toEqual([]);
  });
  test("a refusal line for each state", () => {
    expect(artifactProblem("missing")).toBe("missing");
    expect(artifactProblem("mismatch")).toBe("not the pinned bytes (sha256 differs)");
    expect(artifactProblem({ unreadable: "EACCES" })).toBe("unreadable (EACCES)");
  });
});
