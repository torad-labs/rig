// A GGUF v3 writer for fixtures: enough of the format to build a pack the derive slice can edit.
// Tensor data is aligned to 32 like real files; offsets in the table are relative to the data
// section, as the spec says and as the reader (src/features/pack-derivation/gguf.ts) expects.
export interface FixtureTensor {
  name: string;
  ne: number[];
  type: number;
  data: Uint8Array;
}

export function writeGguf(
  tensors: FixtureTensor[],
  kv: Record<string, string | number> = {},
): Uint8Array {
  const parts: Uint8Array[] = [];
  const enc = new TextEncoder();
  const u32 = (n: number) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, n, true);
    parts.push(b);
  };
  const u64 = (n: number) => {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setBigUint64(0, BigInt(n), true);
    parts.push(b);
  };
  const str = (s: string) => {
    const b = enc.encode(s);
    u64(b.length);
    parts.push(b);
  };
  u32(0x46554747);
  u32(3);
  u64(tensors.length);
  u64(Object.keys(kv).length);
  for (const [k, val] of Object.entries(kv)) {
    str(k);
    if (typeof val === "string") {
      u32(8);
      str(val);
    } else {
      u32(4);
      u32(val);
    }
  }
  const align = 32;
  let off = 0;
  for (const t of tensors) {
    str(t.name);
    u32(t.ne.length);
    for (const n of t.ne) u64(n);
    u32(t.type);
    u64(off);
    off += Math.ceil(t.data.length / align) * align;
  }
  const headerLen = parts.reduce((a, p) => a + p.length, 0);
  const dataStart = Math.ceil(headerLen / align) * align;
  const out = new Uint8Array(dataStart + off);
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  pos = dataStart;
  for (const t of tensors) {
    out.set(t.data, pos);
    pos += Math.ceil(t.data.length / align) * align;
  }
  return out;
}

/** One PQ2_0 block: an fp16 scale and 128 ternary digits (code = digit + 1). */
export function pq2Block(scaleF16: number, digits: number[]): Uint8Array {
  const b = new Uint8Array(34);
  b[0] = scaleF16 & 0xff;
  b[1] = scaleF16 >> 8;
  for (let j = 0; j < 128; j++) {
    const i = 2 + (j >> 2);
    b[i] = (b[i] ?? 0) | (((digits[j] ?? 0) + 1) << (2 * (j & 3)));
  }
  return b;
}

export const f32Bytes = (xs: number[]) => new Uint8Array(Float32Array.from(xs).buffer);
