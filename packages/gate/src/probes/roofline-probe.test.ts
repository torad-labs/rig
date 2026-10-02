import { Database } from "bun:sqlite";
import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FakeShell } from "@rig/testing";
import type { ProbeContext } from "./probe.ts";
import {
  type Capture,
  type DeviceProfile,
  deviceOf,
  parseBenchRate,
  parseNode,
  parseProfile,
  parseProfiles,
  readRooflineCapture,
  rooflineLines,
  rooflineOf,
  rooflineProbe,
  type Token,
  type TwoCardRoofline,
  tokensOf,
  twoCardLines,
  twoCardRoofline,
  weightRole,
} from "./roofline-probe.ts";

const dir = mkdtempSync(join(tmpdir(), "rig-roofline-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// ggml_cuda_init's lines as the fork prints them (the 5080, 2026-09-27)
const INIT = [
  "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 15880 MiB):",
  "  Device 0: NVIDIA GeForce RTX 5080, compute capability 12.0, VMM: yes, VRAM: 15880 MiB",
  "    profile: 84 SMs at 2640 MHz, 65536 regs/SM, 100 KiB smem/SM (99 KiB/block opt-in), 1536 threads/SM, 24 blocks/SM, L2 65536 KiB (persisting 47104 KiB), DRAM 1000 GB/s (15001 MHz x 256 bit)",
].join("\n");
const CARD = { name: "NVIDIA GeForce RTX 5080", sms: 84, dramGBs: 1000, line: "" };

const UP = "MUL_MAT ffn_up-0 r=2000000 w=0 w0=blk.0.ffn_up.weight";
const GATE = "fused MUL_MAT ffn_gate-0 r=2000000 w=0 w0=blk.0.ffn_gate.weight";
const GLU = "fused SWIGLU ffn_swiglu-0 r=0 w=0";
const NORM = "RMS_NORM norm-1 r=20480 w=20480";
const DOWN = "MUL_MAT ffn_down-0 r=1000000 w=0 w0=blk.0.ffn_down.weight";
const tid = 7;
const k = (kernel: string, call: number, start: number, end: number) => ({
  kernel,
  call,
  start,
  end,
  tid,
});

/** a prefill (one node, not a decode's shape), a first decode that gives the next its start, and two measured decodes,
 *  ns, three nodes each. Token 1: the fused up + gate + SWIGLU launch runs 3,100-5,100; the norm, under PDL, starts at
 *  4,000 and waits for it, ending at 5,600; the down 5,600-6,600; a kernel launched outside a node 6,700-6,800. Token 2:
 *  the fused launch 9,000-11,000, the down 11,000-12,000; its norm launched nothing. */
const CAPTURE: Capture = {
  ranges: [
    { start: -5000, end: -4000, tid, text: "graph" },
    { start: -4990, end: -4980, tid, text: UP },
    { start: 0, end: 1000, tid, text: "graph" },
    { start: 10, end: 20, tid, text: UP },
    { start: 30, end: 40, tid, text: NORM },
    { start: 50, end: 60, tid, text: DOWN },
    { start: 2000, end: 3000, tid, text: "graph" },
    { start: 2010, end: 2020, tid, text: UP },
    { start: 2030, end: 2040, tid, text: NORM },
    { start: 2050, end: 2060, tid, text: DOWN },
    { start: 7000, end: 8000, tid, text: "graph" },
    { start: 7010, end: 7020, tid, text: UP },
    { start: 7030, end: 7040, tid, text: NORM },
    { start: 7050, end: 7060, tid, text: DOWN },
  ],
  marks: [
    { at: 2015, tid, text: GATE },
    { at: 2016, tid, text: GLU },
    { at: 7015, tid, text: GATE },
    { at: 7016, tid, text: GLU },
  ],
  kernels: [
    k("mmvq_pq2_mma", -4985, -4900, -4800),
    k("mmvq_pq2_mma", 15, 100, 600),
    k("mmvq_pq2_mma", 2012, 3100, 5100),
    k("rms_norm_fwht_cuda", 2035, 4000, 5600),
    k("mmvq_pq2_mma", 2055, 5600, 6600),
    k("k_bin_bcast", 2500, 6700, 6800),
    k("mmvq_pq2_mma", 7012, 9000, 11000),
    k("mmvq_pq2_mma", 7055, 11000, 12000),
  ],
};
const FUSED = "MUL_MAT ffn_up + MUL_MAT ffn_gate + SWIGLU";

describe("roofline parsing", () => {
  test("the card is the engine's device and profile lines; an engine without the profile line is named", () => {
    const card = parseProfile(`${INIT}\nmain: done`);
    if (typeof card === "string") throw new Error(card);
    expect([card.name, card.sms, card.dramGBs]).toEqual(["NVIDIA GeForce RTX 5080", 84, 1000]);
    expect(card.line).toStartWith("profile: 84 SMs");
    expect(parseProfile(INIT.split("\n").slice(0, 2).join("\n"))).toContain(
      "no device profile line",
    );
  });
  test("a node range names its op, node, bytes and weights; a name with spaces parses, anything else is null", () => {
    expect(parseNode(UP)).toEqual({
      op: "MUL_MAT",
      name: "ffn_up-0",
      read: 2000000,
      written: 0,
      weight: "blk.0.ffn_up.weight",
    });
    expect(parseNode("CPY cache_k_l3 (view) (copy of Kcur-3) r=4096 w=2048")).toMatchObject({
      op: "CPY",
      name: "cache_k_l3 (view) (copy of Kcur-3)",
      weight: null,
    });
    expect(parseNode("graph")).toBeNull();
    expect(weightRole("blk.12.ffn_up_exps.weight")).toBe("ffn_up_exps");
  });
  test("a plain run's rate is llama-bench's tg row, at depth or not; without the row it is null", () => {
    // the 5070 Ti's rows, 2026-09-27
    const row = (test: string, rate: string) =>
      `| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   7.12 GiB |    27.32 B | CUDA       |  99 |   q4_0 |   q4_0 |    f16 |   1 | ${test} | ${rate} |`;
    expect(parseBenchRate(row("           tg64", "        83.86 ± 1.60"), 64)).toBe(83.86);
    expect(parseBenchRate(row("  tg64 @ d16384", "        79.90 ± 1.01"), 64)).toBe(79.9);
    expect(parseBenchRate(row("           tg64", "        83.86 ± 1.60"), 16)).toBeNull();
  });
});

describe("roofline", () => {
  test("each kernel is charged the time it ends past the kernels before it; the host has the rest of the plain wall", () => {
    const tokens = tokensOf(CAPTURE);
    if (typeof tokens === "string") throw new Error(tokens);
    expect(tokens.map((t) => [t.start, t.end])).toEqual([
      [600, 6800],
      [6800, 12000],
    ]);
    expect(tokens[0]?.launches.map((l) => [l.key, l.bytes])).toEqual([
      [FUSED, 4000000],
      ["RMS_NORM", 40960],
      ["MUL_MAT ffn_down", 1000000],
      ["(outside a node)", 0],
    ]);
    const r = rooflineOf(tokens, CARD, 6);
    // the capture's span (6,200 + 5,200) / 2; charged (2,000 + 500 + 1,000 + 100 + 2,000 + 1,000) / 2
    expect([r.tokens, r.wallUs, r.captureUs, r.kernelUs]).toEqual([2, 6, 5.7, 3.3]);
    expect(r.hostUs).toBeCloseTo(2.7, 9);
    expect(r.groups.map((g) => [g.key, g.launches, g.us, g.mb])).toEqual([
      ["RMS_NORM", 0.5, 0.25, 0.02048],
      ["(outside a node)", 0.5, 0.05, 0],
      ["MUL_MAT ffn_down", 1, 1, 1],
      [FUSED, 1, 2, 4],
    ]);
    expect(r.floorUs).toBeCloseTo(5.02048, 9);
    // the norm's own 1,600 ns would count 1,100 of waiting on the fused launch
    // the fused launch and the down are mmvq_pq2_mma's; the norm is no matmul
    expect(r.fits.map((f) => [f.kernel, f.launches])).toEqual([["mmvq_pq2_mma", 4]]);
    expect(r.fits[0]?.fixedUs).toBeCloseTo(2 / 3, 9);
    expect(r.fits[0]?.gbs).toBeCloseTo(3000, 6);
    expect(r.fits[0]?.rmsUs).toBeCloseTo(0, 9);
    const lines = rooflineLines({ ...r, card: { ...CARD, line: "profile: 84 SMs" } });
    expect(lines.slice(1, 3)).toEqual([
      "a token: 6 us plain (166666.7 tok/s): 3 us of kernels (3 with each launch at its median), 3 us the host's; its bytes at 1,000 GB/s take 5 us, 83.7 % of the wall",
      "the capture: 2 tokens of 6 us (175438.6 tok/s), the host slowed by nsys's tracing",
    ]);
    expect(lines.at(-1)).toContain(`| ${FUSED} [mmvq_pq2_mma]`);
  });
  test("a hole in most tokens, at a different layer each, moves a group's mean, not its rank or the streaming fit", () => {
    // three tokens of a 5 us norm, a 4 MB up at 2 us and two layers' 1 MB downs at 1 us, back to back; the first
    // token's first down and the second's second take 30 us, as when the 5070 Ti time-slices with the desktop, and the
    // third's up has a quantize before it that does
    const tokens: Token[] = [
      [30, 1, 0],
      [1, 30, 0],
      [1, 1, 30],
    ].map(([a = 0, b = 0, q = 0], i) => {
      const at = i * 100_000;
      const run = (kernel: string, start: number, us: number) =>
        k(kernel, at + start, at + start, at + start + us * 1000);
      const down = (start: number, us: number) => ({
        key: "MUL_MAT ffn_down",
        bytes: 1e6,
        matmul: true,
        kernels: [run("mmvq_pq2_mma", start, us)],
      });
      const up = q > 0 ? [run("quantize_q8_1", 5000, q)] : [];
      const d0 = 7000 + q * 1000;
      return {
        start: at,
        end: d0 + (a + b) * 1000,
        launches: [
          { key: "RMS_NORM", bytes: 0, matmul: false, kernels: [run("rms_norm_f32", 0, 5)] },
          {
            key: "MUL_MAT ffn_up",
            bytes: 4e6,
            matmul: true,
            kernels: [...up, run("mmvq_pq2_mma", 5000 + q * 1000, 2)],
          },
          down(d0, a),
          down(d0 + a * 1000, b),
        ],
      };
    });
    const r = rooflineOf(tokens, CARD, 20);
    // by the mean the downs would rank first, 19.3 us over their 2 us floor; by a token's total (31, 31, 2), 29
    expect(r.groups.map((g) => [g.key, g.medianUs])).toEqual([
      ["RMS_NORM", 5],
      ["MUL_MAT ffn_down", 2],
      ["MUL_MAT ffn_up", 2],
    ]);
    expect(r.groups[1]?.us).toBeCloseTo(64 / 3, 9);
    expect([r.kernelUs, r.medianKernelUs]).toEqual([115 / 3, 9]);
    // the third token's up is mmvq_pq2_mma's point though its quantize took longer
    expect(r.fits.map((f) => [f.kernel, f.launches])).toEqual([["mmvq_pq2_mma", 9]]);
    expect(r.fits[0]?.fixedUs).toBeCloseTo(2 / 3, 9);
    expect(r.fits[0]?.gbs).toBeCloseTo(3000, 6);
    expect(r.fits[0]?.rmsUs).toBeCloseTo(0, 9);
  });
  test("a kernel whose launch sizes span less than 2x gets no fit", () => {
    const close = {
      ...CAPTURE,
      ranges: CAPTURE.ranges.map((r) =>
        r.text === DOWN ? { ...r, text: DOWN.replace("r=1000000", "r=3000000") } : r,
      ),
    };
    const tokens = tokensOf(close);
    if (typeof tokens === "string") throw new Error(tokens);
    expect(rooflineOf(tokens, CARD, 6).fits).toEqual([]);
  });
  test("a capture without the node ranges, ending in one decode, or with a graph that launched nothing, is an error", () => {
    expect(tokensOf({ ranges: [], marks: [], kernels: [] })).toContain(
      "is GGML_CUDA_NVTX in this engine?",
    );
    // token 1 with a fourth node: the last evaluation is the only one of its shape
    const extra = { start: 2070, end: 2080, tid, text: NORM };
    expect(tokensOf({ ...CAPTURE, ranges: [...CAPTURE.ranges, extra] })).toBe(
      "the capture ends with 1 graph evaluation of 3 nodes, and a token needs the one before it",
    );
    const idle = { ...CAPTURE, kernels: CAPTURE.kernels.filter((x) => x.call > 1000) };
    expect(tokensOf(idle)).toBe("a graph evaluation launched no kernel");
  });
});

/** the capture as nsys exports it, reduced to what the probe reads; devices[i] tags kernel i (all 0 when absent) */
function writeCapture(path: string, c: Capture, devices?: number[]) {
  rmSync(path, { force: true });
  const db = new Database(path, { create: true });
  db.run("create table StringIds (id integer primary key, value text not null)");
  db.run(
    'create table NVTX_EVENTS (start integer, "end" integer, eventType integer, globalTid integer, text text, textId integer)',
  );
  db.run(
    "create table CUPTI_ACTIVITY_KIND_RUNTIME (correlationId integer, start integer, globalTid integer)",
  );
  db.run(
    'create table CUPTI_ACTIVITY_KIND_KERNEL (correlationId integer, shortName integer, start integer, "end" integer, deviceId integer)',
  );
  const ids = new Map<string, number>();
  const id = (value: string) => {
    if (!ids.has(value)) {
      ids.set(value, ids.size + 1);
      db.run("insert into StringIds values (?, ?)", [ids.size, value]);
    }
    return ids.get(value) as number;
  };
  db.transaction(() => {
    // ranges with their text inline, one registered through StringIds, as nsys writes either
    for (const [i, r] of c.ranges.entries()) {
      const inline = i % 2 === 0;
      db.run("insert into NVTX_EVENTS values (?, ?, 59, ?, ?, ?)", [
        r.start,
        r.end,
        r.tid,
        inline ? r.text : null,
        inline ? null : id(r.text),
      ]);
    }
    for (const m of c.marks)
      db.run("insert into NVTX_EVENTS values (?, null, 34, ?, ?, null)", [m.at, m.tid, m.text]);
    db.run("insert into NVTX_EVENTS values (0, 0, 75, 1, 'a domain', null)");
    for (const [i, x] of c.kernels.entries()) {
      db.run("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?)", [i + 1, x.call, x.tid]);
      db.run("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?)", [
        i + 1,
        id(x.kernel),
        x.start,
        x.end,
        devices?.[i] ?? x.device ?? 0,
      ]);
    }
  })();
  db.close();
}

describe("roofline capture", () => {
  test("reads the ranges, marks and kernels nsys exported, text inline or registered", () => {
    const path = join(dir, "read.sqlite");
    writeCapture(path, CAPTURE);
    const read = readRooflineCapture(path);
    const byStart = <T extends { start: number }>(xs: T[]) =>
      [...xs].sort((a, b) => a.start - b.start);
    expect(byStart(read.ranges)).toEqual(byStart(CAPTURE.ranges));
    expect(read.marks).toEqual(CAPTURE.marks);
    expect(byStart(read.kernels)).toEqual(byStart(CAPTURE.kernels));
  });
});

describe("roofline probe", () => {
  // llama-bench's rows for the plain runs: 50 tok/s, 20,000 us a token
  const RATES = [
    "|          test |                  t/s |",
    "| ------------: | -------------------: |",
    "|           tg2 |         50.00 ± 1.00 |",
    "|   tg2 @ d16384 |         50.00 ± 1.00 |",
  ].join("\n");
  function nsys(stderr: string, plain = RATES) {
    const shell = new FakeShell();
    shell.tools.add("nsys");
    const env: Record<string, string>[] = [];
    shell.on(/^\S*llama-bench /, () => ({ code: 0, stdout: plain, stderr: "" }));
    shell.on(/^nsys profile /, (_cmd, opts) => {
      env.push(opts?.env ?? {});
      return { code: 0, stdout: "", stderr };
    });
    shell.on(/^nsys export /, (cmd) => {
      writeCapture(cmd[cmd.indexOf("-o") + 1] as string, CAPTURE);
      return { code: 0, stdout: "", stderr: "" };
    });
    return { shell, env };
  }
  const ctx = (shell: FakeShell) =>
    ({
      head: { servedPath: "/packs/served.gguf" },
      cache: { k: "q4_0", v: "q4_0", s: "f16" },
      gates: { roofline: { gen: 2, depths: [0, 16384] } },
      binDir: "/r/local/engine-builds/abc1234-sm120",
      gpu: 1,
      cap: "120",
      shell,
      runDir: dir,
    }) as unknown as ProbeContext;
  const BENCH = [
    "/r/local/engine-builds/abc1234-sm120/llama-bench",
    "-m",
    "/packs/served.gguf",
    "-ngl",
    "99",
    "-fa",
    "1",
    "-ctk",
    "q4_0",
    "-ctv",
    "q4_0",
    "-cts",
    "f16",
    "-p",
    "0",
    "-n",
    "2",
    "-d",
  ];

  test("at each depth: a plain llama-bench for the wall, then the capture with the node ranges on and graphs off", async () => {
    const { shell, env } = nsys(INIT);
    const result = await rooflineProbe.run(ctx(shell));
    expect(result.pass).toBe("measured");
    expect(result.summary).toBe(
      "depth 0: 20,000 us a token against a 5 us floor; most over its floor: RMS_NORM, +0 us; depth 16384: 20,000 us a token against a 5 us floor; most over its floor: RMS_NORM, +0 us",
    );
    expect(shell.calls.slice(0, 2)).toEqual([
      [...BENCH, "0", "-r", "3"],
      [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx",
        "--sample",
        "none",
        "--cpuctxsw",
        "none",
        "-f",
        "true",
        "-o",
        join(dir, "roofline-d0"),
        ...BENCH,
        "0",
        "-r",
        "1",
      ],
    ]);
    expect(shell.calls[3]).toEqual([...BENCH, "16384", "-r", "3"]);
    expect(env[0]).toMatchObject({
      CUDA_VISIBLE_DEVICES: "1",
      GGML_CUDA_NVTX: "1",
      GGML_CUDA_DISABLE_GRAPHS: "1",
    });
    expect(result.lines[0]).toBe("depth 0:");
    expect(result.data).toMatchObject({
      cache: { k: "q4_0", v: "q4_0", s: "f16", k_bias: false },
      gen: 2,
    });
  });
  test("an engine without the profile line, a plain run with no rate, or no nsys, fails and says why", async () => {
    const old = await rooflineProbe.run(ctx(nsys(INIT.split("\n").slice(0, 2).join("\n")).shell));
    expect([old.pass, old.summary]).toEqual([
      false,
      "the engine printed no device profile line (ggml_cuda_init's `profile:`): the pinned fork predates it",
    ]);
    const norate = await rooflineProbe.run(ctx(nsys(INIT, "llama-bench: error").shell));
    expect([norate.pass, norate.summary]).toEqual([
      false,
      "the plain llama-bench at depth 0 exited 0 with no tg2 rate",
    ]);
    expect((await rooflineProbe.run(ctx(new FakeShell()))).summary).toBe("nsys is not on PATH");
  });
});

describe("two-card roofline", () => {
  // the GLM box's init: two cards, one profile line each
  const INIT2 = [
    "ggml_cuda_init: found 2 CUDA devices (Total VRAM: 196608 MiB):",
    "  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes, VRAM: 98304 MiB",
    "    profile: 148 SMs at 2610 MHz, 65536 regs/SM, 100 KiB smem/SM (99 KiB/block opt-in), 1536 threads/SM, 24 blocks/SM, L2 131072 KiB (persisting 47104 KiB), DRAM 1792 GB/s (15001 MHz x 512 bit)",
    "  Device 1: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes, VRAM: 98304 MiB",
    "    profile: 148 SMs at 2610 MHz, 65536 regs/SM, 100 KiB smem/SM (99 KiB/block opt-in), 1536 threads/SM, 24 blocks/SM, L2 131072 KiB (persisting 47104 KiB), DRAM 1792 GB/s (15001 MHz x 512 bit)",
  ].join("\n");

  test("every device profile parses in printed order; none is the pinned-fork error", () => {
    const cards = parseProfiles(INIT2);
    if (typeof cards === "string") throw new Error(cards);
    expect(cards.map((c) => [c.name, c.sms, c.dramGBs])).toEqual([
      ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 148, 1792],
      ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 148, 1792],
    ]);
    expect(cards[0]?.line).toStartWith("profile: 148 SMs");
    // one card reads exactly as the single-card parse
    const one = parseProfiles(INIT);
    const uno = parseProfile(INIT);
    if (typeof one === "string" || typeof uno === "string") throw new Error("no profile");
    expect(one).toEqual([uno]);
    expect(parseProfiles("no devices here")).toContain("no device profile line");
  });

  test("the device-tagged read attaches each kernel's card; the plain read has no device key", () => {
    const path = join(dir, "dev.sqlite");
    const both: Capture = {
      ranges: [],
      marks: [],
      kernels: [
        { ...k("a", 1, 0, 10), device: 0 },
        { ...k("b", 2, 0, 10), device: 1 },
      ],
    };
    writeCapture(path, both);
    const tagged = readRooflineCapture(path, { withDevice: true });
    expect(tagged.kernels.map((x) => x.device)).toEqual([0, 1]);
    const plain = readRooflineCapture(path);
    expect(plain.kernels.every((x) => !("device" in x))).toBe(true);
  });

  test("each device charges its own timeline; the token floor sums under layer and maxes under tensor", () => {
    const kd = (device: number, start: number, end: number) => ({
      kernel: "m",
      call: start,
      start,
      end,
      tid,
      device,
    });
    const launch = (kernels: Token["launches"][number]["kernels"]) => ({
      key: "MUL_MAT w",
      bytes: 1e6,
      matmul: false,
      kernels,
    });
    // fully overlapping: one timeline would charge 100 + 0, two charge 100 and 50
    const tokens: Token[] = [
      { start: 0, end: 100000, launches: [launch([kd(0, 0, 100000), kd(1, 0, 50000)])] },
    ];
    const cards: DeviceProfile[] = [
      { name: "c0", sms: 1, dramGBs: 1000, line: "" },
      { name: "c1", sms: 1, dramGBs: 500, line: "" },
    ];
    const layer = twoCardRoofline(cards, tokens, 1000, "layer");
    expect(layer.perDevice.map((r) => r.kernelUs)).toEqual([100, 50]);
    expect(layer.floorUs).toBeCloseTo(3, 9);
    const tensor = twoCardRoofline(cards, tokens, 1000, "tensor");
    expect(tensor.perDevice.map((r) => r.kernelUs)).toEqual([100, 50]);
    expect(tensor.floorUs).toBeCloseTo(2, 9);
    expect(twoCardLines(layer)[2]).toContain("(layer)");
    expect(twoCardLines(tensor)[2]).toContain("(tensor)");
  });

  test("the probe run on two cards reports per-device totals and the split floor", async () => {
    const shell = new FakeShell();
    shell.tools.add("nsys");
    shell.on(/^\S*llama-bench /, () => ({
      code: 0,
      stdout: [
        "|          test |                  t/s |",
        "| ------------: | -------------------: |",
        "|           tg2 |         50.00 ± 1.00 |",
      ].join("\n"),
      stderr: "",
    }));
    shell.on(/^nsys profile /, () => ({ code: 0, stdout: "", stderr: INIT2 }));
    shell.on(/^nsys export /, (cmd) => {
      writeCapture(cmd[cmd.indexOf("-o") + 1] as string, CAPTURE, [0, 0, 0, 0, 1, 1, 1, 1]);
      return { code: 0, stdout: "", stderr: "" };
    });
    const ctx = {
      head: { servedPath: "/packs/served.gguf" },
      cache: { k: "q4_0", v: "q4_0", s: "f16" },
      gates: { roofline: { gen: 2, depths: [0] } },
      binDir: "/r/local/engine-builds/abc1234-sm120",
      gpu: 1,
      cap: "120",
      shell,
      runDir: dir,
    } as unknown as ProbeContext;
    const result = await rooflineProbe.run(ctx);
    expect(result.pass).toBe("measured");
    expect(result.summary).toContain("(layer)");
    expect(result.lines[1]).toStartWith("NVIDIA RTX PRO 6000");
    expect(result.lines[4]).toStartWith("card 0 groups:");
    // the tagged kernels land on their own card: both floors nonzero
    const data = result.data as { depths: { twoCard: TwoCardRoofline }[] };
    const floors = data.depths[0]?.twoCard.perDevice.map((r) => r.floorUs) ?? [];
    expect(floors.length).toBe(2);
    expect(floors[0]).toBeGreaterThan(0);
    expect(floors[1]).toBeGreaterThan(0);
  });
});

describe("layer-split tokens", () => {
  // under -sm layer a token is two graph evaluations: card 0's split (up + down)
  // then card 1's (up + norm + down, it holds the output head). Six evaluations
  // of alternating shape are two measured tokens plus the start giver.
  const LAYER: Capture = {
    ranges: [
      { start: 0, end: 100, tid, text: "graph" },
      { start: 10, end: 20, tid, text: UP },
      { start: 30, end: 40, tid, text: DOWN },
      { start: 200, end: 300, tid, text: "graph" },
      { start: 210, end: 220, tid, text: UP },
      { start: 230, end: 240, tid, text: NORM },
      { start: 250, end: 260, tid, text: DOWN },
      { start: 400, end: 500, tid, text: "graph" },
      { start: 410, end: 420, tid, text: UP },
      { start: 430, end: 440, tid, text: DOWN },
      { start: 600, end: 700, tid, text: "graph" },
      { start: 610, end: 620, tid, text: UP },
      { start: 630, end: 640, tid, text: NORM },
      { start: 650, end: 660, tid, text: DOWN },
      { start: 800, end: 900, tid, text: "graph" },
      { start: 810, end: 820, tid, text: UP },
      { start: 830, end: 840, tid, text: DOWN },
      { start: 1000, end: 1100, tid, text: "graph" },
      { start: 1010, end: 1020, tid, text: UP },
      { start: 1030, end: 1040, tid, text: NORM },
      { start: 1050, end: 1060, tid, text: DOWN },
    ],
    marks: [],
    kernels: [
      k("mmvq_pq2_mma", 15, 110, 160),
      k("mmvq_pq2_mma", 35, 160, 190),
      k("mmvq_pq2_mma", 215, 310, 360),
      k("rms_norm_fwht_cuda", 235, 360, 390),
      k("mmvq_pq2_mma", 255, 390, 420),
      k("mmvq_pq2_mma", 415, 510, 560),
      k("mmvq_pq2_mma", 435, 560, 590),
      k("mmvq_pq2_mma", 615, 710, 760),
      k("rms_norm_fwht_cuda", 635, 760, 790),
      k("mmvq_pq2_mma", 655, 790, 820),
      k("mmvq_pq2_mma", 815, 910, 960),
      k("mmvq_pq2_mma", 835, 960, 990),
      k("mmvq_pq2_mma", 1015, 1110, 1160),
      k("rms_norm_fwht_cuda", 1035, 1160, 1190),
      k("mmvq_pq2_mma", 1055, 1190, 1220),
    ],
  };

  test("two evaluations of different shape make one token, chained end to start", () => {
    const tokens = tokensOf(LAYER);
    if (typeof tokens === "string") throw new Error(tokens);
    expect(tokens.length).toBe(2);
    expect(tokens[0]?.launches.map((l) => l.key)).toEqual([
      "MUL_MAT ffn_up",
      "MUL_MAT ffn_down",
      "MUL_MAT ffn_up",
      "RMS_NORM",
      "MUL_MAT ffn_down",
    ]);
    expect(tokens[1]?.start).toBe(tokens[0]?.end);
    expect(tokens[1]?.launches.map((l) => l.key)).toEqual(tokens[0]?.launches.map((l) => l.key));
  });
});

describe("tensor-split tokens", () => {
  // under -sm tensor each step of a token is a graph evaluation on each card in turn, the two of equal nodes: here
  // card 0's up, card 1's up, then card 0's norm + down and card 1's. The node counts repeat at 1 (each evaluation's
  // twin on the other card) as well as at the token's 4; three tokens are two measured ones plus the start giver.
  const kd = (kernel: string, call: number, start: number, end: number, device: number) => ({
    ...k(kernel, call, start, end),
    device,
  });
  const token = (b: number) => ({
    ranges: [
      { start: b, end: b + 100, tid, text: "graph" },
      { start: b + 10, end: b + 20, tid, text: UP },
      { start: b + 200, end: b + 300, tid, text: "graph" },
      { start: b + 210, end: b + 220, tid, text: UP },
      { start: b + 400, end: b + 500, tid, text: "graph" },
      { start: b + 410, end: b + 420, tid, text: NORM },
      { start: b + 430, end: b + 440, tid, text: DOWN },
      { start: b + 600, end: b + 700, tid, text: "graph" },
      { start: b + 610, end: b + 620, tid, text: NORM },
      { start: b + 630, end: b + 640, tid, text: DOWN },
    ],
    kernels: [
      kd("mmvq_pq2_mma", b + 15, b + 110, b + 160, 0),
      kd("mmvq_pq2_mma", b + 215, b + 310, b + 360, 1),
      kd("rms_norm_f32", b + 415, b + 510, b + 530, 0),
      kd("mmvq_pq2_mma", b + 435, b + 530, b + 560, 0),
      kd("rms_norm_f32", b + 615, b + 710, b + 730, 1),
      kd("mmvq_pq2_mma", b + 635, b + 730, b + 760, 1),
    ],
  });
  const tokens3 = [0, 1000, 2000].map(token);
  const TENSOR: Capture = {
    ranges: tokens3.flatMap((t) => t.ranges),
    marks: [],
    kernels: tokens3.flatMap((t) => t.kernels),
  };

  test("a token is every step on every card, not one card's evaluation, chained end to start", () => {
    const tokens = tokensOf(TENSOR);
    if (typeof tokens === "string") throw new Error(tokens);
    expect(tokens.length).toBe(2);
    expect(tokens[0]?.launches.map((l) => l.key)).toEqual([
      "MUL_MAT ffn_up",
      "MUL_MAT ffn_up",
      "RMS_NORM",
      "MUL_MAT ffn_down",
      "RMS_NORM",
      "MUL_MAT ffn_down",
    ]);
    expect(tokens[0]?.launches.flatMap((l) => l.kernels.map(deviceOf))).toEqual([0, 1, 0, 0, 1, 1]);
    expect(tokens[0]?.end).toBe(1760);
    expect(tokens[1]?.start).toBe(tokens[0]?.end);
    expect(tokens[1]?.end).toBe(2760);
  });
});
