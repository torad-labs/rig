import { describe, expect, test } from "bun:test";
import {
  kernelFamily,
  ownTimeCensus,
  pairedChange,
  parseCsv,
  readBenchRate,
  readGpuTrace,
  readKldSummary,
  readOpsVerdict,
} from "./lab-readings.ts";

describe("readOpsVerdict", () => {
  test("the count and each failing case, its error with it, colour codes stripped", () => {
    const text = [
      "Backend 2/3: CUDA1",
      "  DSV4_HC_PRE_POST(type_w=bf16,n_embd=4096,n_tokens=1,n_iter=20): \x1b[1;32mOK\x1b[0m",
      "[DSV4_HC_POST] ERR = 0.048931274 > 0.000500000   DSV4_HC_PRE_POST(type_w=bf16,n_embd=4096,n_tokens=8,n_iter=20): \x1b[1;31mFAIL\x1b[0m",
      "  3/5 tests passed",
      "  Backend CUDA1: \x1b[1;31mFAIL\x1b[0m",
      "2/3 backends passed",
    ].join("\n");
    expect(readOpsVerdict(text)).toEqual({
      passed: 3,
      total: 5,
      failures: [
        "[DSV4_HC_POST] ERR = 0.048931274 > 0.000500000   DSV4_HC_PRE_POST(type_w=bf16,n_embd=4096,n_tokens=8,n_iter=20): FAIL",
      ],
    });
  });
  test("a run that died before its count has no verdict", () => {
    expect(readOpsVerdict("Backend 1/3: CUDA0\nSegmentation fault")).toBeNull();
  });
});

describe("readKldSummary", () => {
  test("llama-perplexity's mean KLD, same top p and PPL ratio", () => {
    const text = [
      "Mean PPL(Q)/PPL(base)         :   0.999556 ±   0.001251",
      "Mean    KLD:   0.006462 ±   0.000010",
      "Maximum KLD:   0.010349",
      "Same top p: 79.863 ± 0.443 %",
    ].join("\n");
    expect(readKldSummary(text)).toEqual({
      meanKld: 0.006462,
      meanKldErr: 0.00001,
      sameTopP: 79.863,
      sameTopPErr: 0.443,
      pplRatio: 0.999556,
    });
  });
  test("no mean KLD, no summary", () => {
    expect(readKldSummary("Final estimate: PPL = 344182.1625 +/- 3428.48235")).toBeNull();
  });
});

describe("readBenchRate", () => {
  test("the mean of the repetitions after the first", () => {
    const rate = readBenchRate(`{"n_prompt": 4096, "samples_ts": [ 2000, 2076.0, 2096.0 ]}\n`);
    expect(rate).toEqual({ ok: true, value: { rate: 2086, samples: [2000, 2076, 2096] } });
  });
  test("one repetition is its own rate", () => {
    const rate = readBenchRate(`{"samples_ts": [ 1500 ]}`);
    expect(rate.ok && rate.value.rate).toBe(1500);
  });
  test("two tests in one run are refused: a pair compares one number", () => {
    const rate = readBenchRate(`{"samples_ts": [1, 2]}\n{"samples_ts": [3, 4]}`);
    expect(rate.ok).toBe(false);
  });
});

describe("pairedChange", () => {
  test("the mean change and its interval on t(0.975, n - 1)", () => {
    const change = pairedChange([0.01, 0.02, 0.03]);
    expect(change.n).toBe(3);
    expect(change.mean).toBeCloseTo(0.02, 12);
    // sd 0.01, t(0.975, 2) 4.303: half-width 4.303 * 0.01 / sqrt(3)
    expect(change.interval?.t).toBe(4.303);
    expect(change.interval?.lo).toBeCloseTo(0.02 - (4.303 * 0.01) / Math.sqrt(3), 12);
    expect(change.interval?.hi).toBeCloseTo(0.02 + (4.303 * 0.01) / Math.sqrt(3), 12);
  });
  test("one pair has no interval", () => {
    expect(pairedChange([0.05])).toEqual({ n: 1, mean: 0.05, interval: null });
  });
});

describe("the GPU trace and the own-time census", () => {
  const header =
    "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name";
  const row = (start: number, duration: number, stream: number, name: string, device = "GPU (0)") =>
    `${start},${duration},1,16,1,1,128,1,1,32,0,0,,,,,${device},1,,${stream},"${name.replaceAll('"', '""')}"`;
  test("kernels only, a quoted signature's commas and quotes kept", () => {
    const csv = [
      header,
      "100,800,344,,,,,,,,,,0.000,340.000,Device,,GPU (0),1,,7,[CUDA memset]",
      row(1000, 50, 7, 'void rms_norm_f32<(int)1024, (bool)0>(const float *, "x")'),
    ].join("\n");
    const runs = readGpuTrace(csv);
    expect(runs).toEqual({
      ok: true,
      value: [
        {
          device: "GPU (0)",
          stream: "7",
          start: 1000,
          end: 1050,
          name: 'void rms_norm_f32<(int)1024, (bool)0>(const float *, "x")',
        },
      ],
    });
  });
  test("not nsys's trace is refused", () => {
    expect(readGpuTrace("a,b\n1,2").ok).toBe(false);
  });
  test("a kernel that started under the one before it on its stream is billed from where that one ended", () => {
    const runs = readGpuTrace(
      [
        header,
        row(0, 100, 7, "void a_kernel<1>(int)"),
        // launched early under PDL: starts at 40, waits until 100, ends at 130
        row(40, 90, 7, "void b_kernel(int)"),
        // another stream overlaps freely
        row(50, 20, 8, "void a_kernel<2>(int)"),
        row(0, 10, 7, "void b_kernel(int)", "GPU (1)"),
      ].join("\n"),
    );
    expect(runs.ok).toBe(true);
    const census = ownTimeCensus(runs.ok ? runs.value : []);
    expect(census.kernels).toEqual([
      { kernel: "a_kernel", launches: 2, ownNs: 120, nsysNs: 120, byDevice: { "GPU (0)": 120 } },
      {
        kernel: "b_kernel",
        launches: 2,
        ownNs: 40,
        nsysNs: 100,
        byDevice: { "GPU (0)": 30, "GPU (1)": 10 },
      },
    ]);
    expect(census.ownNs).toBe(160);
    expect(census.nsysNs).toBe(220);
  });
  test("a kernel's family drops its template arguments, parameters and void", () => {
    expect(kernelFamily("void mul_mat_q<(ggml_type)12, (int)64>(const char *, int)")).toBe(
      "mul_mat_q",
    );
    expect(kernelFamily("dsv4_hc_post_f32(const float *)")).toBe("dsv4_hc_post_f32");
  });
  test("parseCsv reads CRLF rows and doubled quotes", () => {
    expect(parseCsv('a,"b,""c"""\r\n1,2')).toEqual([
      ["a", 'b,"c"'],
      ["1", "2"],
    ]);
  });
});
