// What the engine's own tools print, read into numbers: test-backend-ops' verdicts, llama-perplexity's KL-divergence
// summary, llama-bench's JSON lines and nsys's GPU trace as CSV, and the two judgements made of them: a paired A/B's
// change with its interval, and each kernel's own time. Pure: the lab runs the tools and hands their text here.
import { ExitCode, fail, ok, type Result } from "@rig/core";

// biome-ignore lint/suspicious/noControlCharactersInRegex: the colour codes test-backend-ops prints
const ANSI = /\x1b\[[0-9;]*m/g;

export interface OpsVerdict {
  passed: number;
  total: number;
  /** each failing case's line, its error first where the tool printed one ("[OP] ERR = 0.05 > 0.0005  OP(...)") */
  failures: string[];
}

/** test-backend-ops' "N/M tests passed" and its failing cases; null when it printed no count (it died first) */
export function readOpsVerdict(text: string): OpsVerdict | null {
  const plain = text.replace(ANSI, "");
  const count = plain.match(/(\d+)\/(\d+) tests passed/);
  if (!count) return null;
  const failures = plain
    .split("\n")
    .map((line) => line.trim())
    // a case's line ends in its verdict; the backend's own summary line ("Backend CUDA0: FAIL") names no case
    .filter((line) => /\): FAIL$/.test(line));
  return { passed: Number(count[1]), total: Number(count[2]), failures };
}

export interface KldSummary {
  meanKld: number;
  meanKldErr: number;
  /** percent of tokens whose top token is the base's */
  sameTopP: number;
  sameTopPErr: number;
  /** Mean PPL(Q)/PPL(base) */
  pplRatio: number | null;
}

/** llama-perplexity --kl-divergence's summary; null without its mean KLD (the run did not get that far) */
export function readKldSummary(text: string): KldSummary | null {
  const kld = text.match(/Mean\s+KLD:\s+([-\d.e]+)\s+±\s+([-\d.e]+)/);
  const top = text.match(/Same top p:\s+([-\d.e]+)\s+±\s+([-\d.e]+)\s*%/);
  if (!kld || !top) return null;
  const ratio = text.match(/Mean PPL\(Q\)\/PPL\(base\)\s*:\s*([-\d.e]+)/);
  return {
    meanKld: Number(kld[1]),
    meanKldErr: Number(kld[2]),
    sameTopP: Number(top[1]),
    sameTopPErr: Number(top[2]),
    pplRatio: ratio ? Number(ratio[1]) : null,
  };
}

/** one llama-bench test's rate in tokens a second, from its `-o jsonl` output: the mean of its repetitions from the
 *  second on, the first reading slow on a card that was idle (all of them when there is one). One test a run: a run
 *  of several (-p and -n together) is refused, as the pairs compare one number. */
export function readBenchRate(jsonl: string): Result<{ rate: number; samples: number[] }> {
  const tests = jsonl.split("\n").filter((line) => line.startsWith("{"));
  if (tests.length !== 1)
    return fail(
      ExitCode.Failure,
      `llama-bench printed ${tests.length} tests, not one: a pair compares one (-p or -n, one value)`,
    );
  const samples = (JSON.parse(tests[0] as string) as { samples_ts?: number[] }).samples_ts ?? [];
  if (samples.length === 0) return fail(ExitCode.Failure, "llama-bench printed no samples_ts");
  const counted = samples.length > 1 ? samples.slice(1) : samples;
  return ok({ rate: counted.reduce((a, b) => a + b, 0) / counted.length, samples });
}

/** t(0.975, df) for df 1..30; past 30 the normal's 1.96, a little narrow (2.04 at 30) */
const T975 = [
  12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145,
  2.131, 2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048,
  2.045, 2.042,
];

export interface PairedChange {
  n: number;
  /** mean of the pairs' b/a - 1 */
  mean: number;
  /** the 95 % interval on t(0.975, n - 1), from two pairs on */
  interval: { lo: number; hi: number; t: number } | null;
}

/** the pairs' mean change and its 95 % interval, each pair one observation (b/a - 1): back-to-back runs share the
 *  card's state, so the pair, not the run, is the unit */
export function pairedChange(changes: readonly number[]): PairedChange {
  const n = changes.length;
  const mean = changes.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { n, mean, interval: null };
  const sd = Math.sqrt(changes.reduce((a, c) => a + (c - mean) ** 2, 0) / (n - 1));
  const t = T975[n - 2] ?? 1.96;
  const half = (t * sd) / Math.sqrt(n);
  return { n, mean, interval: { lo: mean - half, hi: mean + half, t } };
}

export interface KernelRun {
  device: string;
  stream: string;
  start: number;
  end: number;
  name: string;
}

/** the kernels of `nsys stats --report cuda_gpu_trace --format csv` (memcpys and memsets, which have no grid, left
 *  out), times in ns */
export function readGpuTrace(csv: string): Result<KernelRun[]> {
  const rows = parseCsv(csv);
  const header = rows[0] ?? [];
  const col = (name: string) => header.indexOf(name);
  const [start, duration, grid, device, stream, name] = [
    col("Start (ns)"),
    col("Duration (ns)"),
    col("GrdX"),
    col("Device"),
    col("Strm"),
    col("Name"),
  ];
  if ([start, duration, grid, device, stream, name].some((i) => i < 0))
    return fail(
      ExitCode.Failure,
      `not nsys's cuda_gpu_trace CSV: its header is ${header.join(",")}`,
    );
  const kernels: KernelRun[] = [];
  for (const row of rows.slice(1)) {
    if (!row[grid as number]) continue;
    const s = Number(row[start as number]);
    kernels.push({
      device: row[device as number] ?? "",
      stream: row[stream as number] ?? "",
      start: s,
      end: s + Number(row[duration as number]),
      name: row[name as number] ?? "",
    });
  }
  return ok(kernels);
}

/** RFC 4180 rows: quoted fields hold commas and doubled quotes (a kernel's signature does) */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += c;
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

/** a kernel's family: its name without template arguments, parameters or `void` (every rms_norm_f32<...> one) */
export const kernelFamily = (name: string) =>
  (name.split("(")[0] ?? "")
    .split("<")[0]
    ?.replace(/^void /, "")
    .trim() ?? name;

export interface KernelShare {
  kernel: string;
  launches: number;
  /** ns of its own: from max(its start, the end of the kernel before it on its stream) to its end */
  ownNs: number;
  /** nsys's durations, the wait for the kernel before it included */
  nsysNs: number;
  /** own ns by device */
  byDevice: Record<string, number>;
}

/** each kernel family's own time over every device. Under programmatic dependent launch a kernel starts while the one
 *  before it on its stream runs, and waits for it: nsys's duration bills it that wait, so a kernel is billed only
 *  from where the one before it ended. Largest first. */
export function ownTimeCensus(runs: readonly KernelRun[]): {
  ownNs: number;
  nsysNs: number;
  kernels: KernelShare[];
} {
  const sorted = [...runs].sort(
    (a, b) =>
      a.device.localeCompare(b.device) || a.stream.localeCompare(b.stream) || a.start - b.start,
  );
  const shares = new Map<string, KernelShare>();
  let key = "";
  let prevEnd = 0;
  for (const run of sorted) {
    const at = `${run.device}\u0000${run.stream}`;
    if (at !== key) {
      key = at;
      prevEnd = 0;
    }
    const own = run.end - Math.max(run.start, prevEnd);
    prevEnd = Math.max(prevEnd, run.end);
    const family = kernelFamily(run.name);
    const share = shares.get(family) ?? {
      kernel: family,
      launches: 0,
      ownNs: 0,
      nsysNs: 0,
      byDevice: {},
    };
    share.launches++;
    share.ownNs += own;
    share.nsysNs += run.end - run.start;
    share.byDevice[run.device] = (share.byDevice[run.device] ?? 0) + own;
    shares.set(family, share);
  }
  const kernels = [...shares.values()].sort((a, b) => b.ownNs - a.ownNs);
  return {
    ownNs: kernels.reduce((a, k) => a + k.ownNs, 0),
    nsysNs: kernels.reduce((a, k) => a + k.nsysNs, 0),
    kernels,
  };
}
