import { describe, expect, test } from "bun:test";
import { ablateTensor } from "./lattice-ablation.ts";
import { BLOCK_BYTES, pack, QK, unpack } from "./pq2.ts";

function block(scaleF16: number, digits: number[]): Uint8Array {
  const b = new Uint8Array(BLOCK_BYTES);
  b[0] = scaleF16 & 0xff;
  b[1] = scaleF16 >> 8;
  for (let j = 0; j < QK; j++) {
    const i = 2 + (j >> 2);
    b[i] = (b[i] ?? 0) | (((digits[j] ?? 0) + 1) << (2 * (j & 3)));
  }
  return b;
}

describe("pq2_0", () => {
  test("unpack/pack round-trips every digit and keeps the scale bytes", () => {
    const d = Array.from({ length: QK }, (_, j) => (j % 3) - 1);
    const raw = new Uint8Array([
      ...block(0x3c00, d),
      ...block(
        0xbc00,
        d.map((x) => -x),
      ),
    ]); // scales 1.0 and -1.0
    const u = unpack(raw, 1, 2 * QK);
    expect(Array.from(u.digits.subarray(0, QK))).toEqual(d);
    expect(Array.from(u.scales)).toEqual([1, -1]);
    expect(pack(u.digits, raw, 1, 2 * QK)).toEqual(raw);
  });
  test("pack refuses a digit off the lattice", () => {
    const raw = new Uint8Array(block(0x3c00, []));
    const d = new Int8Array(QK);
    d[5] = 2;
    expect(() => pack(d, raw, 1, QK)).toThrow("lattice");
  });
});

describe("ablateTensor", () => {
  test("removes the direction's component from the column sums and touches only the rows spent", () => {
    // 4 rows x 128 cols, all digits +1, scale 1.0; direction concentrated on row 0
    const N = 4,
      K = QK;
    const raw = new Uint8Array(N * BLOCK_BYTES);
    for (let n = 0; n < N; n++) raw.set(block(0x3c00, Array(K).fill(1)), n * BLOCK_BYTES);
    const r = new Float32Array([1, 0, 0, 0]);
    const { out, stats } = ablateTensor(raw, N, K, r, { rows: 1, lambda: 1, rowCap: 1 });
    // c_k = 1 for every column; one flip of row 0 from +1 to 0 brings it to 0
    expect(stats.flipped).toBe(K);
    expect(stats.removed).toBeCloseTo(1, 6);
    const u = unpack(out, N, K);
    expect(Array.from(u.digits.subarray(0, K)).every((x) => x === 0)).toBe(true);
    expect(Array.from(u.digits.subarray(K)).every((x) => x === 1)).toBe(true);
    expect(out.subarray(0, 2)).toEqual(raw.subarray(0, 2)); // scale bytes untouched
  });
  test("row_cap bounds the share of a row's digits one pass may flip", () => {
    const N = 2,
      K = QK;
    const raw = new Uint8Array(N * BLOCK_BYTES);
    for (let n = 0; n < N; n++) raw.set(block(0x3c00, Array(K).fill(1)), n * BLOCK_BYTES);
    const { stats } = ablateTensor(raw, N, K, new Float32Array([1, 0.5]), {
      rows: 2,
      lambda: 1,
      rowCap: 0.25,
    });
    expect(stats.flipped).toBe(2 * 32); // 25% of 128 per row, two rows
  });
});
