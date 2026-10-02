import { describe, expect, test } from "bun:test";
import { fakePorts } from "@rig/testing";
import { fetchPinned } from "./download.ts";

const BYTES = "pinned bytes";
const sha256 = new Bun.CryptoHasher("sha256").update(BYTES).digest("hex");

/** a machine with aria2c and curl, each writing the pinned bytes to the file it is told to */
function machine() {
  const p = fakePorts();
  p.shell.tools.add("aria2c");
  p.shell.on(/^aria2c/, (cmd) => {
    p.fs.put(`${cmd[cmd.indexOf("-d") + 1]}/${cmd[cmd.indexOf("-o") + 1]}`, BYTES);
    return { code: 0, stdout: "", stderr: "" };
  });
  p.shell.on(/^curl/, (cmd) => {
    p.fs.put(cmd[cmd.indexOf("-o") + 1]!, BYTES);
    return { code: 0, stdout: "", stderr: "" };
  });
  return p;
}

describe("fetchPinned", () => {
  test("aria2c fetches http(s); curl fetches a file:// URL, which aria2c refuses", async () => {
    for (const [url, tool] of [
      ["https://example.org/x.tar.gz", "aria2c"],
      ["file:///prebuilt/x.tar.gz", "curl"],
    ] as const) {
      const p = machine();
      const r = await fetchPinned(p, url, { path: "/r/local/downloads/x.tar.gz", sha256 }, "x");
      expect(r.ok).toBe(true);
      expect(p.shell.calls.map((c) => c[0])).toEqual([tool]);
      expect(p.fs.text("/r/local/downloads/x.tar.gz")).toBe(BYTES);
    }
  });

  test("aria2c's progress and warnings reach the log named by the file; its summary table does not", async () => {
    const p = machine();
    p.shell.on(/^aria2c/, (cmd) => {
      p.fs.put(`${cmd[cmd.indexOf("-d") + 1]}/${cmd[cmd.indexOf("-o") + 1]}`, BYTES);
      return {
        code: 0,
        stdout:
          "[#0202aa 298MiB/522MiB(57%) CN:16 DL:188MiB ETA:1s]\n\nDownload Results:\ngid   |stat|avg speed  |path/URI\n",
        stderr: "09/30 21:50:01 [WARN] CUID#7 - Download aborted. URI=https://example.org/x\n",
      };
    });
    const r = await fetchPinned(
      { ...p, log: p.log },
      "https://example.org/x.tar.gz",
      { path: "/r/local/downloads/x.tar.gz", sha256 },
      "x",
    );
    expect(r.ok).toBe(true);
    expect(p.log.lines.filter((line) => line.includes("x.tar.gz:"))).toEqual([
      expect.stringContaining("x.tar.gz: [#0202aa 298MiB/522MiB(57%) CN:16 DL:188MiB ETA:1s]"),
      expect.stringContaining("x.tar.gz: 09/30 21:50:01 [WARN] CUID#7 - Download aborted."),
    ]);
  });
});
