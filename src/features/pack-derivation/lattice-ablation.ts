// Bake a refusal-direction ablation into a PQ2_0 pack by flipping ternary digits on the lattice.
//
// Standard abliteration projects the direction out of the residual writers, W <- W - r (r^T W).
// On a ternary pack that projection rounds straight back (a ~1.4% nudge against a 100% lattice
// step), so the edit is done as BoldingBuilds did for Bonsai 2: the quantity to remove is the row
// sum c_k = sum_n r_n W_nk, one value per input column; a single flipped digit moves it by
// r_n * s_{n,g} (a whole lattice step), so digits are spent on the rows with the largest |r_n|,
// greedily, one pass, clipped to {-1, 0, +1}, block scales untouched. --row-cap bounds the share
// of one row's digits the pass may flip: uncapped it piles the edit on the few most r-aligned
// rows (67% of one row in blk.20.ffn_down) and that pack doubled words; capped at 10% the same
// component removal spreads over more rows and the artifact is gone.
//
// ARITHMETIC ORDER IS THE CONTRACT. The served pack is pinned by sha256 and a box must reproduce
// it byte for byte, so this is a transcription of the numpy reference (ablate-pq2_0.py, 2026-09-19)
// down to the precision of each operation: products r_n * s * d in float32, the column sums
// accumulated in float64 in row order, the per-row step in float32, comparisons in float64
// against a float32 half-step. The head's derive test proves it: the reference sha or nothing.
import { pack, QK, unpack } from "./pq2.ts";

export interface AblateParams {
  rows: number;
  lambda: number;
  rowCap: number;
}
export interface AblateStats {
  flipped: number;
  digits: number;
  removed: number;
}

export function ablateTensor(
  raw: Uint8Array,
  N: number,
  K: number,
  r: Float32Array,
  params: AblateParams,
): { out: Uint8Array; stats: AblateStats } {
  if (r.length !== N) throw new Error(`direction has ${r.length} rows, tensor has ${N}`);
  const { digits, scales, nb } = unpack(raw, N, K);
  // c_k = sum_n (r_n * s_{n,g(k)}) * d_nk: the f32 product of the f32 row factor, summed in f64 row by row
  const c = new Float64Array(K);
  for (let n = 0; n < N; n++) {
    const rn = r[n]!;
    const srow = n * nb;
    const drow = n * K;
    for (let b = 0; b < nb; b++) {
      const f = Math.fround(rn * scales[srow + b]!);
      const base = drow + b * QK;
      const kb = b * QK;
      for (let j = 0; j < QK; j++) {
        const d = digits[base + j]!;
        if (d !== 0) c[kb + j]! += Math.fround(f * d);
      }
    }
  }
  const c0 = Float64Array.from(c);
  const oneMinusLam = 1 - params.lambda;
  const target = new Float64Array(K);
  for (let k = 0; k < K; k++) target[k] = oneMinusLam * c0[k]!;
  // rows by |r| descending, ties by index (numpy's stable argsort of -|r|)
  const order = Array.from({ length: N }, (_, i) => i)
    .sort((a, b) => Math.abs(r[b]!) - Math.abs(r[a]!) || a - b)
    .slice(0, params.rows);
  const cap = Math.trunc(params.rowCap * K);
  const step = new Float32Array(K);
  const want = new Int8Array(K);
  const ok = new Uint8Array(K);
  const gain = new Float64Array(K);
  let flipped = 0;
  for (const n of order) {
    const rn = r[n]!;
    const srow = n * nb;
    const drow = n * K;
    for (let b = 0; b < nb; b++) {
      const f = Math.fround(rn * scales[srow + b]!);
      for (let j = 0; j < QK; j++) step[b * QK + j] = f;
    }
    let count = 0;
    for (let k = 0; k < K; k++) {
      const resid = c[k]! - target[k]!;
      const st = step[k]!;
      const w = -Math.sign(resid) * Math.sign(st); // the digit move that shrinks |resid|
      const nxt = digits[drow + k]! + w;
      const good = Math.abs(resid) > Math.abs(st) / 2 && w !== 0 && nxt >= -1 && nxt <= 1;
      want[k] = w;
      ok[k] = good ? 1 : 0;
      if (good) count++;
    }
    if (count === 0) continue;
    if (count > cap) {
      // keep the `cap` columns with the most to gain (numpy: argpartition on -gain; ties by index here)
      for (let k = 0; k < K; k++) gain[k] = ok[k] ? Math.abs(c[k]! - target[k]!) : -1;
      const idx = Array.from({ length: K }, (_, k) => k)
        .filter((k) => ok[k])
        .sort((a, b) => gain[b]! - gain[a]! || a - b);
      ok.fill(0);
      for (let i = 0; i < cap; i++) ok[idx[i]!] = 1;
      count = cap;
    }
    for (let k = 0; k < K; k++)
      if (ok[k]) {
        digits[drow + k] = (digits[drow + k]! + want[k]!) as number;
        c[k]! += Math.fround(step[k]! * want[k]!);
      }
    flipped += count;
  }
  // component removed: |c - target| after the pass over |c0 - target| before it
  let remaining = 0;
  let original = 0;
  for (let k = 0; k < K; k++) {
    const after = c[k]! - target[k]!;
    const before = c0[k]! - target[k]!;
    remaining += after * after;
    original += before * before;
  }
  return {
    out: pack(digits, raw, N, K),
    stats: { flipped, digits: N * K, removed: 1 - Math.sqrt(remaining) / Math.sqrt(original) },
  };
}

/** The unit direction from a rank-1 adapter: every lora_b column must be one direction. */
export function directionFromLoraB(vectors: Float32Array[]): Float32Array {
  const first = vectors[0];
  if (!first) throw new Error("the adapter has no lora_b tensors");
  const sameLength = vectors.filter((vector) => vector.length === first.length);
  const ref = normalize(first);
  let minCos = 1;
  for (const vector of sameLength) {
    const unit = normalize(vector);
    let dot = 0;
    for (let i = 0; i < unit.length; i++) dot += unit[i]! * ref[i]!;
    minCos = Math.min(minCos, Math.abs(dot));
  }
  if (minCos < 0.9999)
    throw new Error(`lora_b vectors are not one direction (min |cos| ${minCos.toFixed(6)})`);
  return ref;
}

/** b / ||b|| in float32, the norm accumulated in float32 like numpy's dot on a float32 vector. */
function normalize(vector: Float32Array): Float32Array {
  let acc = 0;
  for (let i = 0; i < vector.length; i++) {
    acc = Math.fround(acc + Math.fround(vector[i]! * vector[i]!));
  }
  const norm = Math.fround(Math.sqrt(acc));
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = Math.fround(vector[i]! / norm);
  return out;
}
