// Where a decode token's time goes on this card, against what the card's DRAM allows: llama-bench on
// the served pack under Nsight Systems with the fork's NVTX node ranges (GGML_CUDA_NVTX=1, ggml-cuda's
// nvtx.cuh), CUDA graphs off (a replay has no host ranges; they add nothing to this head's decode:
// engine 02512a3's release, 287.2 tok/s with them off against 286) and PDL on, as served.
// Each kernel belongs to the node range its launch call ran in, and a node's range names the bytes
// it reads and writes; a launch that fused the nodes after its own marks each of them inside its
// range. A token is one graph evaluation of the run the capture ends with, the evaluations with as
// many nodes as the last (a prefill's or a warm-up's differ), each but the first, which only gives
// the next its start. Its kernels are swept in start order and each is charged the time it ends past
// every kernel before it: under PDL a kernel starts early and waits, so its own duration would count
// the wait. nsys traces every CUDA call, which slows the host that enqueues them (the 5070 Ti, depth
// 16,384: 70.0 tok/s captured against 79.9 plain), so a token's wall is a plain llama-bench's, run
// beside the capture, and the host's share is that wall less the kernels' charged time.
// A group is the launches of one shape (the same ops and weight roles), a layer's each; its floor is
// its bytes at the card's peak DRAM rate from the engine's profile line. The report ranks the groups
// by their time over their floor, each launch (a group's nth in a token) at its median over tokens:
// what to optimize for this head on this card. The streaming fits take each matmul kernel's launches
// (a launch goes to the kernel it spent the most time in): time = a fixed cost per launch + bytes at
// a rate, least squares over each launch size's median, for a kernel whose sizes span 2x or more.
// Medians, as a card that drives a display loses one kernel of most tokens to it.
import { Database } from "bun:sqlite";
import { join } from "node:path";
import { benchCache } from "@rig/head";
import type { Probe, ProbeResult } from "./probe.ts";

export interface DeviceProfile {
  name: string;
  sms: number;
  /** the peak the memory clock and bus give, GB/s */
  dramGBs: number;
  line: string;
}

const DEVICE = /^\s*Device 0: ([^,]+), compute capability/m;
const PROFILE = /^\s*profile: (\d+) SMs at .*DRAM (\d+) GB\/s.*$/m;

/** how two cards share a token: layer turns them in turn, tensor streams together */
export type SplitMode = "layer" | "tensor";

const DEVICES =
  /^\s*Device (\d+): ([^,]+), compute capability[\s\S]*?^\s*profile: (\d+) SMs at .*DRAM (\d+) GB\/s.*$/gm;

/** every device profile the engine printed, in printed order; the pinned-fork error when none */
export function parseProfiles(output: string): DeviceProfile[] | string {
  const out: DeviceProfile[] = [];
  for (const m of output.matchAll(DEVICES)) {
    const line = (m[0] ?? "").split("\n").pop()?.trim() ?? "";
    out.push({
      name: m[2] ?? "",
      sms: Number(m[3]),
      dramGBs: Number(m[4]),
      line,
    });
  }
  if (out.length === 0) {
    return "the engine printed no device profile line (ggml_cuda_init's `profile:`): the pinned fork predates it";
  }
  return out;
}

/** the card from the engine's init log: its name line and its profile line */
export function parseProfile(output: string): DeviceProfile | string {
  const device = DEVICE.exec(output);
  const profile = PROFILE.exec(output);
  if (!device || !profile) {
    return "the engine printed no device profile line (ggml_cuda_init's `profile:`): the pinned fork predates it";
  }
  return {
    name: device[1] ?? "",
    sms: Number(profile[1]),
    dramGBs: Number(profile[2]),
    line: profile[0].trim(),
  };
}

/** llama-bench's markdown row for tg<gen>: its tok/s mean; null without the row */
export function parseBenchRate(stdout: string, gen: number): number | null {
  const row = stdout
    .split("\n")
    .find((l) => new RegExp(`\\|\\s*tg${gen}( @ d\\d+)?\\s*\\|`).test(l));
  const m = row ? /\|\s*([\d.]+) ± [\d.]+\s*\|\s*$/.exec(row) : null;
  return m ? Number(m[1]) : null;
}

export interface NodeRange {
  op: string;
  name: string;
  read: number;
  written: number;
  weight: string | null;
}

const NODE = /^(\S+) (.+) r=(\d+) w=(\d+)(?: w0=(\S+))?$/;

/** a node range's name, `<op> <node name> r=<bytes> w=<bytes>[ w0=<weights>]`; null for anything else */
export function parseNode(text: string): NodeRange | null {
  const m = NODE.exec(text);
  if (!m) return null;
  return {
    op: m[1] ?? "",
    name: m[2] ?? "",
    read: Number(m[3]),
    written: Number(m[4]),
    weight: m[5] ?? null,
  };
}

/** blk.12.ffn_up.weight -> ffn_up */
export const weightRole = (name: string) =>
  name.replace(/^blk\.\d+\./, "").replace(/\.weight$/, "");

const nodeDesc = (n: NodeRange) => (n.weight ? `${n.op} ${weightRole(n.weight)}` : n.op);

export interface HostRange {
  start: number;
  end: number;
  tid: number;
  text: string;
}
export interface HostMark {
  at: number;
  tid: number;
  text: string;
}
export interface KernelRun {
  kernel: string;
  start: number;
  end: number;
  /** the host call that launched it: when, on which thread */
  call: number;
  tid: number;
  /** the CUPTI device, present only when read with { withDevice: true } */
  device?: number;
}

/** the device a kernel ran on; kernels read without it are card 0's */
export const deviceOf = (k: KernelRun): number => k.device ?? 0;
export interface Capture {
  ranges: HostRange[];
  marks: HostMark[];
  kernels: KernelRun[];
}

/** one node's launch: its shape (key), the bytes it and the nodes it fused move, its kernels */
export interface Launch {
  key: string;
  bytes: number;
  matmul: boolean;
  kernels: KernelRun[];
}
export interface Token {
  /** the previous token's last kernel end, ns */
  start: number;
  /** this token's last kernel end, ns */
  end: number;
  launches: Launch[];
}

/** index of the first element whose key is >= x, in an array sorted by key */
function lowerBound<T>(xs: T[], key: (t: T) => number, x: number): number {
  let [lo, hi] = [0, xs.length];
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(xs[mid] as T) < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** the items whose time falls in [start, end] on one thread, from an array sorted by time */
function within<T extends { tid: number }>(xs: T[], at: (t: T) => number, r: HostRange): T[] {
  const out: T[] = [];
  for (let i = lowerBound(xs, at, r.start); i < xs.length && at(xs[i] as T) <= r.end; i++) {
    const x = xs[i] as T;
    if (x.tid === r.tid) out.push(x);
  }
  return out;
}

/** the decode tokens the capture ends with, each launch with its kernels; an error without two decode evaluations.
 *  A token is one evaluation on one card, or one evaluation per split on several (layer split evaluates
 *  card 0's split then card 1's, whose node counts differ; tensor split evaluates each step on every card
 *  in turn, twins of equal nodes): the trailing evaluations group by the period of their node-count
 *  sequence whose run of two tokens or more covers the most of them, the shortest on a tie (p = 1 on one
 *  card; under tensor, where twins repeat at 1 too, the token's steps times the cards) */
export function tokensOf(capture: Capture): Token[] | string {
  const graphs = capture.ranges.filter((r) => r.text === "graph").sort((a, b) => a.start - b.start);
  const nodes = capture.ranges.filter((r) => r.text !== "graph").sort((a, b) => a.start - b.start);
  const marks = [...capture.marks].sort((a, b) => a.at - b.at);
  const kernels = [...capture.kernels].sort((a, b) => a.call - b.call);
  const count = (g: HostRange) =>
    within(nodes, (x) => x.start, g).filter((r) => r.end <= g.end).length;
  const last = graphs.at(-1);
  if (!last)
    return "the capture has no graph evaluation with NVTX ranges: is GGML_CUDA_NVTX in this engine?";
  const counts = graphs.map(count);
  const n = counts.length;
  let period = 0;
  let runLen = 0;
  for (let q = 1; q * 2 <= n; q++) {
    let kk = q;
    let i = n - 1 - q;
    while (i >= 0 && counts[i] === counts[i + q]) {
      kk++;
      i--;
    }
    if (kk >= 2 * q && kk > runLen) {
      period = q;
      runLen = kk;
    }
  }
  if (period === 0) {
    return `the capture ends with 1 graph evaluation of ${counts[n - 1]} nodes, and a token needs the one before it`;
  }
  const tail = graphs.slice(n - runLen);
  const ngroups = Math.floor(runLen / period);
  const cut = tail.slice(runLen - ngroups * period);
  const groups: HostRange[][] = [];
  for (let i = 0; i < cut.length; i += period) groups.push(cut.slice(i, i + period));
  const evalLaunches = (g: HostRange): { launches: Launch[]; end: number } => {
    const launches: Launch[] = [];
    const claimed = new Set<KernelRun>();
    for (const r of within(nodes, (x) => x.start, g)) {
      const node = parseNode(r.text);
      if (!node || r.end > g.end) continue;
      const fused = within(marks, (m) => m.at, r)
        .map((m) => (m.text.startsWith("fused ") ? parseNode(m.text.slice(6)) : null))
        .filter((f): f is NodeRange => f !== null);
      const all = [node, ...fused];
      const ks = within(kernels, (k) => k.call, r);
      for (const k of ks) claimed.add(k);
      if (ks.length === 0) continue;
      launches.push({
        key: all.map(nodeDesc).join(" + "),
        bytes: all.reduce((sum, x) => sum + x.read + x.written, 0),
        matmul: all.some((x) => x.op === "MUL_MAT" || x.op === "MUL_MAT_ID"),
        kernels: ks,
      });
    }
    const loose = within(kernels, (k) => k.call, g).filter((k) => !claimed.has(k));
    if (loose.length > 0)
      launches.push({ key: "(outside a node)", bytes: 0, matmul: false, kernels: loose });
    const end = Math.max(...launches.flatMap((l) => l.kernels.map((k) => k.end)));
    return { launches, end };
  };
  let prevEnd = Math.max(...(groups[0] as HostRange[]).map((g) => evalLaunches(g).end));
  if (!Number.isFinite(prevEnd)) return "a graph evaluation launched no kernel";
  const tokens: Token[] = [];
  for (const grp of groups.slice(1)) {
    const launches: Launch[] = [];
    let end = -Infinity;
    for (const g of grp) {
      const e = evalLaunches(g);
      launches.push(...e.launches);
      end = Math.max(end, e.end);
    }
    if (!Number.isFinite(end)) return "a graph evaluation launched no kernel";
    tokens.push({ start: prevEnd, end, launches });
    prevEnd = end;
  }
  return tokens;
}

export interface GroupRow {
  key: string;
  /** per token */
  launches: number;
  us: number;
  /** the rank's measure: each of its launches in a token (the nth, a layer's) at its median over tokens, summed. A card
   *  that drives a desktop loses 0.6-1.6 ms to it at one kernel of most tokens (the 5070 Ti, 2026-09-27), which moves
   *  the mean and a token's total, not this */
  medianUs: number;
  mb: number;
  floorUs: number;
  kernels: string[];
}
export interface Roofline {
  card: DeviceProfile;
  tokens: number;
  /** per token, us: a plain llama-bench's wall; the capture's span from the previous token's last kernel; the
   *  kernels' charged time; the host's share, the wall less the kernels */
  wallUs: number;
  captureUs: number;
  kernelUs: number;
  /** the groups' medianUs, summed: a token of each launch at its median */
  medianKernelUs: number;
  hostUs: number;
  floorUs: number;
  groups: GroupRow[];
  /** per matmul kernel whose launch sizes span 2x or more: time = fixedUs + bytes at gbs */
  fits: Fit[];
}
export interface Fit {
  kernel: string;
  launches: number;
  fixedUs: number;
  gbs: number;
  rmsUs: number;
}

/** each kernel charged the time it ends past every kernel before it in its token */
function charge(t: Token): Map<KernelRun, number> {
  const all = t.launches.flatMap((l) => l.kernels).sort((a, b) => a.start - b.start);
  const out = new Map<KernelRun, number>();
  let reach = t.start;
  for (const k of all) {
    out.set(k, Math.max(0, k.end - Math.max(k.start, reach)));
    reach = Math.max(reach, k.end);
  }
  return out;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? (s[m] as number) : ((s[m - 1] as number) + (s[m] as number)) / 2;
}

export function rooflineOf(tokens: Token[], card: DeviceProfile, wallUs: number): Roofline {
  const nt = tokens.length;
  const groups = new Map<
    string,
    // byNth[n][i]: the group's nth launch in token i, ns (0 where token i has no nth)
    { launches: number; byNth: number[][]; bytes: number; kernels: Set<string> }
  >();
  // a matmul launch and the kernel it spent the most time in (its quantize is the lesser, but for a hole)
  const matmuls: { key: string; bytes: number; us: number; main: string }[] = [];
  let [span, charged] = [0, 0];
  for (const [i, t] of tokens.entries()) {
    const ns = charge(t);
    span += t.end - t.start;
    const nth = new Map<string, number>();
    for (const l of t.launches) {
      const lns = l.kernels.reduce((sum, k) => sum + (ns.get(k) ?? 0), 0);
      charged += lns;
      const g = groups.get(l.key) ?? {
        launches: 0,
        byNth: [],
        bytes: 0,
        kernels: new Set<string>(),
      };
      const n = nth.get(l.key) ?? 0;
      nth.set(l.key, n + 1);
      const row = g.byNth[n] ?? tokens.map(() => 0);
      row[i] = lns;
      g.byNth[n] = row;
      g.launches += 1;
      g.bytes += l.bytes;
      for (const k of l.kernels) g.kernels.add(k.kernel);
      groups.set(l.key, g);
      if (l.matmul) {
        const main = l.kernels.reduce((a, b) => ((ns.get(b) ?? 0) > (ns.get(a) ?? 0) ? b : a));
        matmuls.push({ key: l.key, bytes: l.bytes, us: lns / 1e3, main: main.kernel });
      }
    }
  }
  // each group's points go to the kernel most of its launches spent the most time in: a desktop hole in one launch's
  // quantize would give quantize_q8_1 a fit of its own (the 5070 Ti at depth 245,760: three launches at 196 us)
  const votes = new Map<string, Map<string, number>>();
  for (const m of matmuls) {
    const v = votes.get(m.key) ?? new Map<string, number>();
    v.set(m.main, (v.get(m.main) ?? 0) + 1);
    votes.set(m.key, v);
  }
  const mainOf = new Map(
    [...votes.entries()].map(([key, v]) => [
      key,
      [...v.entries()].reduce((a, b) => (b[1] > a[1] ? b : a))[0],
    ]),
  );
  const points = new Map<string, [number, number][]>();
  for (const m of matmuls) {
    const kernel = mainOf.get(m.key) as string;
    points.set(kernel, [...(points.get(kernel) ?? []), [m.bytes, m.us]]);
  }
  const floorUs = (bytes: number) => bytes / (card.dramGBs * 1e3);
  const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
  const rows = [...groups.entries()]
    .map(([key, g]) => ({
      key,
      launches: g.launches / nt,
      us: sum(g.byNth.flat()) / 1e3 / nt,
      medianUs: sum(g.byNth.map(median)) / 1e3,
      mb: g.bytes / 1e6 / nt,
      floorUs: floorUs(g.bytes) / nt,
      kernels: [...g.kernels].sort(),
    }))
    .sort((a, b) => b.medianUs - b.floorUs - (a.medianUs - a.floorUs));
  return {
    card,
    tokens: nt,
    wallUs,
    captureUs: span / 1e3 / nt,
    kernelUs: charged / 1e3 / nt,
    medianKernelUs: sum(rows.map((r) => r.medianUs)),
    hostUs: wallUs - charged / 1e3 / nt,
    floorUs: rows.reduce((sum, r) => sum + r.floorUs, 0),
    groups: rows,
    fits: [...points.entries()]
      .map(([kernel, ps]) => fitOf(kernel, ps))
      .filter((f): f is Fit => f !== null)
      .sort((a, b) => b.launches - a.launches),
  };
}

/** a token across cards: one rooflineOf per device over that device's kernels, plus the token floor.
 *  Under layer the cards take turns, so the floor is the sum; under tensor they stream together, the max */
export interface TwoCardRoofline {
  devices: DeviceProfile[];
  split: SplitMode;
  perDevice: Roofline[];
  wallUs: number;
  floorUs: number;
}

export function twoCardRoofline(
  devices: DeviceProfile[],
  tokens: Token[],
  wallUs: number,
  split: SplitMode,
): TwoCardRoofline {
  const perDevice = devices.map((card, d) => {
    const sub = tokens.map((t) => ({
      ...t,
      launches: t.launches
        .map((l) => ({ ...l, kernels: l.kernels.filter((k) => deviceOf(k) === d) }))
        .filter((l) => l.kernels.length > 0),
    }));
    return rooflineOf(sub, card, wallUs);
  });
  const floors = perDevice.map((r) => r.floorUs);
  const floorUs = split === "layer" ? floors.reduce((a, b) => a + b, 0) : Math.max(...floors);
  return { devices, split, perDevice, wallUs, floorUs };
}

/** least squares of time (us) on bytes over a kernel's launch sizes, each size's median time weighted by its launches
 *  (a mean fit through the 5070 Ti's desktop holes left a 56 us rms residual); null unless its largest launch moves at
 *  least twice its smallest's bytes: sizes closer than that cannot tell the fixed cost from the rate (Bonsai's q, k, v
 *  group at 19.6 MB against its qkv + gate group at 22.4 MB fit -15 us a launch) */
function fitOf(kernel: string, points: [number, number][]): Fit | null {
  const bySize = new Map<number, number[]>();
  for (const [b, t] of points) bySize.set(b, [...(bySize.get(b) ?? []), t]);
  const sizes = [...bySize.keys()];
  if (Math.max(...sizes) < 2 * Math.min(...sizes)) return null;
  // [bytes, median us, launches]
  const ps = [...bySize.entries()].map(([b, ts]) => [b, median(ts), ts.length] as const);
  const n = points.length;
  const [sx, sy] = [
    ps.reduce((s, [b, , w]) => s + w * b, 0),
    ps.reduce((s, [, t, w]) => s + w * t, 0),
  ];
  const sxx = ps.reduce((s, [b, , w]) => s + w * b * b, 0);
  const sxy = ps.reduce((s, [b, t, w]) => s + w * b * t, 0);
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); // us per byte
  const fixed = (sy - slope * sx) / n;
  const rms = Math.sqrt(ps.reduce((s, [b, t, w]) => s + w * (t - fixed - slope * b) ** 2, 0) / n);
  return { kernel, launches: n, fixedUs: fixed, gbs: 1 / slope / 1e3, rmsUs: rms };
}

const f0 = (x: number) => Math.round(x).toLocaleString("en-US");
const f1 = (x: number) => x.toFixed(1);

const groupRow = (g: GroupRow): string => {
  const gbs = g.medianUs > 0 ? (g.mb * 1e3) / g.medianUs : 0;
  return `${f1(g.launches).padStart(6)} ${f0(g.us).padStart(7)} ${f0(g.medianUs).padStart(7)} ${f1(g.mb).padStart(8)} ${f0(g.floorUs).padStart(6)} ${f0(g.medianUs - g.floorUs).padStart(7)} ${f0(gbs).padStart(6)} | ${g.key} [${g.kernels.join(", ")}]`;
};

/** the two-card report: per-device totals, then the token floor under the split mode */
export function twoCardLines(r: TwoCardRoofline): string[] {
  const lines = r.devices.map((d) => `${d.name}: ${d.line}`);
  const floors = r.perDevice.map((d, i) => `card ${i}: ${f0(d.floorUs)} us`).join(", ");
  const pct = (100 * r.floorUs) / r.wallUs;
  lines.push(
    `a token: ${f0(r.wallUs)} us plain; device floors ${floors}; token floor ${f0(r.floorUs)} us, ${f1(pct)} % of the wall (${r.split})`,
  );
  r.perDevice.forEach((d, i) => {
    lines.push(`card ${i} groups:`);
    for (const g of d.groups) lines.push(groupRow(g));
  });
  return lines;
}

/** the report: the token's split, the streaming fit, then each group, largest time over floor first */
export function rooflineLines(r: Roofline): string[] {
  const pct = (100 * r.floorUs) / r.wallUs;
  const lines = [
    `${r.card.name}: ${r.card.line}`,
    `a token: ${f0(r.wallUs)} us plain (${f1(1e6 / r.wallUs)} tok/s): ${f0(r.kernelUs)} us of kernels (${f0(r.medianKernelUs)} with each launch at its median), ${f0(r.hostUs)} us the host's; its bytes at ${f0(r.card.dramGBs)} GB/s take ${f0(r.floorUs)} us, ${f1(pct)} % of the wall`,
    `the capture: ${r.tokens} tokens of ${f0(r.captureUs)} us (${f1(1e6 / r.captureUs)} tok/s), the host slowed by nsys's tracing`,
  ];
  for (const f of r.fits) {
    lines.push(
      `${f.kernel} (${f0(f.launches)} matmul launches): ${f1(f.fixedUs)} us each + bytes at ${f0(f.gbs)} GB/s (${f1((100 * f.gbs) / r.card.dramGBs)} % of peak), rms residual of its sizes' medians ${f1(f.rmsUs)} us`,
    );
  }
  lines.push(
    "per token: launches, us charged (mean, median), MB moved, floor us, median over floor us, GB/s at the median | group [kernels]",
  );
  for (const g of r.groups) lines.push(groupRow(g));
  return lines;
}

const NVTX_EVENTS = `
select e.start as start, e."end" as "end", e.eventType as type, e.globalTid as tid, coalesce(e.text, s.value) as text
from NVTX_EVENTS e left join StringIds s on s.id = e.textId
where e.eventType in (34, 59)`;

const KERNELS = `
select s.value as kernel, k.start as start, k."end" as "end", r.start as call, r.globalTid as tid
from CUPTI_ACTIVITY_KIND_KERNEL k
join StringIds s on s.id = k.shortName
join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = k.correlationId`;

const KERNELS_DEVICE = `
select s.value as kernel, k.start as start, k."end" as "end", r.start as call, r.globalTid as tid, k.deviceId as device
from CUPTI_ACTIVITY_KIND_KERNEL k
join StringIds s on s.id = k.shortName
join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = k.correlationId`;

/** the NVTX ranges (push/pop, event type 59) and marks (34), and each kernel with its launch call, from
 *  `nsys export -t sqlite` of a capture made with -t cuda,nvtx. With { withDevice: true } each kernel
 *  also carries its CUPTI device; without it the rows read exactly as before */
export function readRooflineCapture(path: string, opts?: { withDevice?: boolean }): Capture {
  const db = new Database(path, { readonly: true });
  try {
    const events = db.query(NVTX_EVENTS).all() as {
      start: number;
      end: number | null;
      type: number;
      tid: number;
      text: string | null;
    }[];
    const rows = db
      .query(opts?.withDevice === true ? KERNELS_DEVICE : KERNELS)
      .all() as (KernelRun & {
      device?: number;
    })[];
    const kernels: KernelRun[] =
      opts?.withDevice === true
        ? rows.map((k) => ({
            kernel: k.kernel,
            start: k.start,
            end: k.end,
            call: k.call,
            tid: k.tid,
            device: k.device as number,
          }))
        : rows.map((k) => ({
            kernel: k.kernel,
            start: k.start,
            end: k.end,
            call: k.call,
            tid: k.tid,
          }));
    return {
      ranges: events
        .filter((e) => e.type === 59 && e.end !== null && e.text !== null)
        .map((e) => ({ start: e.start, end: e.end as number, tid: e.tid, text: e.text as string })),
      marks: events
        .filter((e) => e.type === 34 && e.text !== null)
        .map((e) => ({ at: e.start, tid: e.tid, text: e.text as string })),
      kernels,
    };
  } finally {
    db.close();
  }
}

export const rooflineProbe: Probe = {
  name: "roofline",
  needs: "card",
  async run(ctx) {
    const cfg = ctx.gates.roofline;
    const failed = (summary: string, lines: string[], data: unknown = null): ProbeResult => ({
      name: "roofline",
      pass: false,
      summary,
      lines,
      data,
    });
    if (!(await ctx.shell.which("nsys"))) {
      return failed("nsys is not on PATH", [
        "the roofline attributes kernels from an Nsight Systems capture (apt: cuda-nsight-systems-13-0)",
      ]);
    }
    const cache = benchCache(ctx.cache);
    const lines: string[] = [];
    const depths: (
      | { depth: number; roofline: Roofline }
      | { depth: number; twoCard: TwoCardRoofline }
    )[] = [];
    for (const depth of cfg.depths) {
      const capture = join(ctx.runDir, `roofline-d${depth}`);
      const bench = [
        `${ctx.binDir}/llama-bench`,
        "-m",
        ctx.head.servedPath,
        "-ngl",
        "99",
        "-fa",
        "1",
        ...cache.args,
        "-p",
        "0",
        "-n",
        String(cfg.gen),
        "-d",
        String(depth),
      ];
      const env = {
        CUDA_DEVICE_ORDER: "PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES: String(ctx.gpu),
        LD_LIBRARY_PATH: ctx.binDir,
      };
      const plain = await ctx.shell.run([...bench, "-r", "3"], { env, timeoutMs: 1_800_000 });
      const rate = plain.code === 0 ? parseBenchRate(plain.stdout, cfg.gen) : null;
      if (rate === null) {
        const tail = `${plain.stdout}\n${plain.stderr}`.trim().split("\n").slice(-5);
        return failed(
          `the plain llama-bench at depth ${depth} exited ${plain.code} with no tg${cfg.gen} rate`,
          tail,
          {
            cmd: [...bench, "-r", "3"],
            cache: cache.ran,
          },
        );
      }
      const profile = [
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
      ];
      const cmd = [...profile, "-o", capture, ...bench, "-r", "1"];
      const profiled = await ctx.shell.run(cmd, {
        env: { ...env, GGML_CUDA_NVTX: "1", GGML_CUDA_DISABLE_GRAPHS: "1" },
        timeoutMs: 1_800_000,
      });
      if (profiled.code !== 0) {
        const tail = profiled.stderr.trim().split("\n").slice(-5);
        return failed(`nsys profile exited ${profiled.code} at depth ${depth}`, tail, {
          cmd,
          cache: cache.ran,
        });
      }
      const profiles = parseProfiles(`${profiled.stdout}\n${profiled.stderr}`);
      if (typeof profiles === "string") return failed(profiles, [], { cmd, cache: cache.ran });
      const exported = await ctx.shell.run(
        [
          "nsys",
          "export",
          "-t",
          "sqlite",
          "-f",
          "true",
          "-o",
          `${capture}.sqlite`,
          `${capture}.nsys-rep`,
        ],
        { timeoutMs: 600_000 },
      );
      if (exported.code !== 0) {
        const tail = exported.stderr.trim().split("\n").slice(-5);
        return failed(`nsys export exited ${exported.code}`, tail, { cmd, cache: cache.ran });
      }
      const tokens = tokensOf(
        readRooflineCapture(
          `${capture}.sqlite`,
          profiles.length > 1 ? { withDevice: true } : undefined,
        ),
      );
      if (typeof tokens === "string") return failed(tokens, [], { cmd, cache: cache.ran });
      if (profiles.length === 1) {
        const card = profiles[0] as DeviceProfile;
        const roofline = rooflineOf(tokens, card, 1e6 / rate);
        depths.push({ depth, roofline });
        lines.push(`depth ${depth}:`, ...rooflineLines(roofline));
      } else {
        const split = cfg.split ?? "layer";
        const two = twoCardRoofline(profiles, tokens, 1e6 / rate, split);
        depths.push({ depth, twoCard: two });
        lines.push(`depth ${depth}:`, ...twoCardLines(two));
      }
    }
    const summary = depths
      .map(({ depth, ...rest }) => {
        if (!("roofline" in rest)) {
          const two = (rest as { twoCard: TwoCardRoofline }).twoCard;
          return `depth ${depth}: ${f0(two.wallUs)} us a token against a ${f0(two.floorUs)} us floor (${two.split})`;
        }
        const r = rest.roofline;
        const top = r.groups[0];
        const over = top
          ? `; most over its floor: ${top.key}, +${f0(top.medianUs - top.floorUs)} us`
          : "";
        return `depth ${depth}: ${f0(r.wallUs)} us a token against a ${f0(r.floorUs)} us floor${over}`;
      })
      .join("; ");
    return {
      name: "roofline",
      pass: "measured",
      summary,
      lines,
      data: { cache: cache.ran, gen: cfg.gen, depths },
    };
  },
};
