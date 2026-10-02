// PQ2_0: 128 ternary digits per block, packed 4 per byte (2 bits each, code = digit + 1), behind
// one fp16 block scale. Element j of a block lives in byte j>>2 at bit 2*(j&3).
import { f16ToF32 } from "./gguf.ts";

export const QK = 128;
export const BLOCK_BYTES = 34;

export interface Unpacked {
  digits: Int8Array;
  scales: Float32Array;
  N: number;
  K: number;
  nb: number;
}

export function unpack(raw: Uint8Array, N: number, K: number): Unpacked {
  const nb = K / QK;
  const digits = new Int8Array(N * K);
  const scales = new Float32Array(N * nb);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (let n = 0; n < N; n++)
    for (let b = 0; b < nb; b++) {
      const base = (n * nb + b) * BLOCK_BYTES;
      scales[n * nb + b] = f16ToF32(view.getUint16(base, true));
      const row = n * K + b * QK;
      for (let j = 0; j < 32; j++) {
        const byte = raw[base + 2 + j]!;
        digits[row + j * 4] = (byte & 3) - 1;
        digits[row + j * 4 + 1] = ((byte >> 2) & 3) - 1;
        digits[row + j * 4 + 2] = ((byte >> 4) & 3) - 1;
        digits[row + j * 4 + 3] = ((byte >> 6) & 3) - 1;
      }
    }
  return { digits, scales, N, K, nb };
}

/** Write digits back over a copy of raw; the scale bytes are kept byte for byte. */
export function pack(digits: Int8Array, raw: Uint8Array, N: number, K: number): Uint8Array {
  const nb = K / QK;
  const out = new Uint8Array(raw);
  for (let n = 0; n < N; n++)
    for (let b = 0; b < nb; b++) {
      const base = (n * nb + b) * BLOCK_BYTES;
      const row = n * K + b * QK;
      for (let j = 0; j < 32; j++) {
        const d0 = (digits[row + j * 4] ?? 0) + 1;
        const d1 = (digits[row + j * 4 + 1] ?? 0) + 1;
        const d2 = (digits[row + j * 4 + 2] ?? 0) + 1;
        const d3 = (digits[row + j * 4 + 3] ?? 0) + 1;
        if (d0 > 2 || d1 > 2 || d2 > 2 || d3 > 2 || d0 < 0 || d1 < 0 || d2 < 0 || d3 < 0)
          throw new Error(`digit left the ternary lattice at row ${n}`);
        out[base + 2 + j] = d0 | (d1 << 2) | (d2 << 4) | (d3 << 6);
      }
    }
  return out;
}
