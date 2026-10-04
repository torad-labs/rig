import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { rxBytes, SAMPLER_PATHS, samplerScript, windowRead } from "./box-sampler.ts";

// The sampler and its reads are shell text, so they are run under sh here: the unit tests of the idle check feed it
// canned output and cannot tell an awk program that parses from one that does not.
const sh = (script: string, env: Record<string, string> = {}) =>
  Bun.spawnSync(["sh", "-c", script], { env: { PATH: "/usr/bin:/bin", ...env } });

function tmp(): string {
  return mkdtempSync(join(tmpdir(), "rig-sampler-"));
}

// /proc/net/dev as a host writes it: a short name is padded before its colon and a long one is not, so the bytes sit in
// a different field on different lines (lo: 5.7e10, eth0: 1000, enp5s0: 2000, br-0a1b2c3d4e5f: 300)
const PROC_NET_DEV = [
  "Inter-|   Receive                                                |  Transmit",
  " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed",
  "    lo: 50000000000 15724797    0    0    0     0          0         0 50000000000 15724797    0    0    0     0       0          0",
  "  eth0:    1000      12    0    0    0     0          0         0      500       5    0    0    0     0       0          0",
  "enp5s0:    2000      34    0    0    0     0          0         0      900       9    0    0    0     0       0          0",
  "br-0a1b2c3d4e5f:     300       7    0    0    0     0          0         0        0       0    0    0    0     0       0          0",
  "",
].join("\n");

describe("the received bytes", () => {
  test("every interface but loopback is summed, whatever its name's length", () => {
    const dir = tmp();
    const file = join(dir, "dev");
    writeFileSync(file, PROC_NET_DEV);
    expect(sh(rxBytes(file)).stdout.toString()).toBe("3300");
  });
  test("the fixture tells a located byte count from a positional one: field 3 of the long names is their packets", () => {
    const dir = tmp();
    const file = join(dir, "dev");
    writeFileSync(file, PROC_NET_DEV);
    const positional = `awk -F'[: ]+' '/:/ && $2 != "lo" { s += $3 } END { printf "%.0f", s }' ${file}`;
    expect(sh(positional).stdout.toString()).not.toBe("3300");
  });
  test("a host with no /proc/net/dev gives an empty reading, not an error", () => {
    expect(sh(rxBytes("/nonexistent/dev")).stdout.toString()).toBe(""); // the sampler writes 0 KB/s for it
  });
});

describe("the window read", () => {
  const now = () => Math.floor(Date.now() / 1000);
  function samples(lines: Array<[number, string]>): string {
    const file = join(tmp(), "samples");
    writeFileSync(file, `${lines.map(([age, rest]) => `${now() - age} ${rest}`).join("\n")}\n`);
    return file;
  }
  test("prints the card's peak and the average received over the window, and ignores what is older", () => {
    const file = samples([
      [3000, "99 90000"], // an hour-old busy sample, out of a 360 s window
      [300, "4 100"],
      [200, "62 300"],
      [100, "3 200"],
      [10, "0 0"],
    ]);
    expect(sh(windowRead(file, 360, 512)).stdout.toString()).toBe("window=62\ndownload=150\n");
  });
  // A window's average is diluted by the ticks it spent not pulling: rental 2b's pull of the pack, 437.5 MB/s by hand,
  // read 218.7 in the window it ended in. The ticks that pulled are averaged on their own, and counted, so a pull is
  // measured at its rate and a short one is told from a sustained one.
  test("prints the average over the ticks that pulled at least the floor, and how many there were", () => {
    const file = samples([
      [300, "0 448000"],
      [200, "0 450000"],
      [100, "0 446000"],
      [50, "0 300"], // the pull ended: an ssh session's trickle, under the floor
      [10, "0 0"],
    ]);
    expect(sh(windowRead(file, 360, 512)).stdout.toString()).toBe(
      "window=0\ndownload=268860\npull=448000 3\n",
    );
  });
  test("a window whose card read 0 throughout prints its peak as 0, not as no reading", () => {
    const file = samples([
      [100, "0 0"],
      [10, "0 0"],
    ]);
    expect(sh(windowRead(file, 360, 512)).stdout.toString()).toBe("window=0\ndownload=0\n");
  });
  test("a window with no tick at the floor prints no pull", () => {
    const file = samples([
      [100, "3 200"],
      [10, "0 511"],
    ]);
    expect(sh(windowRead(file, 360, 512)).stdout.toString()).toBe("window=3\ndownload=355\n");
  });
  test("lines of the first generation carry no download column: the card is read, the download is absent", () => {
    const file = samples([
      [100, "5"],
      [50, "40"],
    ]);
    expect(sh(windowRead(file, 360, 512)).stdout.toString()).toBe("window=40\n");
  });
  test("an empty window prints nothing", () => {
    expect(sh(windowRead(samples([[3000, "99 1"]]), 360, 512)).stdout.toString()).toBe("");
    expect(sh(windowRead("/nonexistent/samples", 360, 512)).stdout.toString()).toBe("");
  });
});

describe("the sampler script", () => {
  test("run under sh it writes its pid and one line a tick: epoch, the busiest card, KB/s received", async () => {
    const dir = tmp();
    const paths = {
      script: join(dir, "sampler.sh"),
      pid: join(dir, "pid"),
      firstPid: join(dir, "first.pid"),
      samples: join(dir, "samples"),
    };
    writeFileSync(paths.script, `${samplerScript(paths, 1).join("\n")}\n`);
    const bin = join(dir, "bin");
    Bun.spawnSync(["mkdir", bin]);
    writeFileSync(join(bin, "nvidia-smi"), "#!/bin/sh\nprintf '37\\n12\\n'\n");
    Bun.spawnSync(["chmod", "+x", join(bin, "nvidia-smi")]);
    const proc = Bun.spawn(["sh", paths.script], { env: { PATH: `${bin}:/usr/bin:/bin` } });
    await Bun.sleep(3500);
    proc.kill();
    expect(readFileSync(paths.pid, "utf8").trim()).toMatch(/^\d+$/);
    const lines = readFileSync(paths.samples, "utf8").trim().split("\n");
    expect(lines.length).toBeGreaterThanOrEqual(2);
    for (const line of lines) expect(line).toMatch(/^\d{10} 37 \d+$/);
  });
  test("its default paths are the ones the first generation did not use, so the two never share a pid file", () => {
    expect(SAMPLER_PATHS.pid).not.toBe(SAMPLER_PATHS.firstPid);
  });
});
