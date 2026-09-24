// A GGUF reader sufficient for editing: the tensor table (name, shape, type, absolute byte offset,
// byte length) and the metadata needed to place the data section. It reads the header from a
// growing prefix of the file, so a 7 GB pack costs a few MB of reads, and it never loads tensor
// data itself — callers read ranges by offset.
import type { FileSystem } from "../../shared/ports/index.ts";

export const GGUF_MAGIC = 0x46554747; // "GGUF" little-endian
export enum GgmlType {
  F32 = 0,
  F16 = 1,
  Q8_0 = 8,
  PQ2_0 = 142,
}

/** bytes per block and elements per block, for the types this code touches */
export function typeLayout(type: number): { blockBytes: number; blockElems: number } {
  switch (type) {
    case GgmlType.F32:
      return { blockBytes: 4, blockElems: 1 };
    case GgmlType.F16:
      return { blockBytes: 2, blockElems: 1 };
    case GgmlType.Q8_0:
      return { blockBytes: 34, blockElems: 32 }; // fp16 scale + 32 int8
    case GgmlType.PQ2_0:
      return { blockBytes: 34, blockElems: 128 }; // fp16 scale + 128 x 2-bit codes
    default:
      throw new Error(`unsupported ggml type ${type}`);
  }
}

export interface TensorInfo {
  name: string;
  ne: number[];
  type: number;
  offset: number;
}

/** byte length of a tensor, for the types this code reads (others are never touched) */
export function tensorBytes(tensor: TensorInfo): number {
  const { blockBytes, blockElems } = typeLayout(tensor.type);
  const columns = tensor.ne[0] ?? 0;
  if (columns % blockElems !== 0) {
    throw new Error(
      `${tensor.name}: ne[0]=${columns} is not a multiple of the block size ${blockElems}`,
    );
  }
  return (elementCount(tensor) / blockElems) * blockBytes;
}

const elementCount = (tensor: TensorInfo) => tensor.ne.reduce((product, dim) => product * dim, 1);
export interface GgufFile {
  path: string;
  version: number;
  alignment: number;
  dataOffset: number;
  tensors: TensorInfo[];
  kv: Map<string, unknown>;
}

class Cursor {
  pos = 0;
  constructor(
    readonly view: DataView,
    readonly bytes: Uint8Array,
  ) {}
  need(bytes: number) {
    if (this.pos + bytes > this.bytes.length) throw new NeedMore();
  }
  u8() {
    this.need(1);
    return this.view.getUint8(this.pos++);
  }
  u16() {
    this.need(2);
    const value = this.view.getUint16(this.pos, true);
    this.pos += 2;
    return value;
  }
  u32() {
    this.need(4);
    const value = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return value;
  }
  i32() {
    this.need(4);
    const value = this.view.getInt32(this.pos, true);
    this.pos += 4;
    return value;
  }
  f32() {
    this.need(4);
    const value = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return value;
  }
  u64() {
    this.need(8);
    const value = this.view.getBigUint64(this.pos, true);
    this.pos += 8;
    return Number(value);
  }
  i64() {
    this.need(8);
    const value = this.view.getBigInt64(this.pos, true);
    this.pos += 8;
    return Number(value);
  }
  f64() {
    this.need(8);
    const value = this.view.getFloat64(this.pos, true);
    this.pos += 8;
    return value;
  }
  str() {
    const n = this.u64();
    this.need(n);
    const text = new TextDecoder().decode(this.bytes.subarray(this.pos, this.pos + n));
    this.pos += n;
    return text;
  }
  value(type: number): unknown {
    switch (type) {
      case 0:
        return this.u8();
      case 1:
        this.need(1);
        return this.view.getInt8(this.pos++);
      case 2:
        return this.u16();
      case 3:
        this.need(2);
        {
          const value = this.view.getInt16(this.pos, true);
          this.pos += 2;
          return value;
        }
      case 4:
        return this.u32();
      case 5:
        return this.i32();
      case 6:
        return this.f32();
      case 7:
        return this.u8() !== 0;
      case 8:
        return this.str();
      case 9: {
        const elementType = this.u32();
        const count = this.u64();
        const out: unknown[] = [];
        for (let i = 0; i < count; i++) out.push(this.value(elementType));
        return out;
      }
      case 10:
        return this.u64();
      case 11:
        return this.i64();
      case 12:
        return this.f64();
      default:
        throw new Error(`unknown gguf value type ${type}`);
    }
  }
}
class NeedMore extends Error {}

export function parseGgufHeader(path: string, bytes: Uint8Array): GgufFile {
  const cursor = new Cursor(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), bytes);
  if (cursor.u32() !== GGUF_MAGIC) throw new Error(`${path} is not a GGUF file`);
  const version = cursor.u32();
  if (version !== 3 && version !== 2) throw new Error(`gguf version ${version} is not supported`);
  const nTensors = cursor.u64();
  const nKv = cursor.u64();
  const kv = new Map<string, unknown>();
  for (let i = 0; i < nKv; i++) {
    const key = cursor.str();
    const valueType = cursor.u32();
    kv.set(key, cursor.value(valueType));
  }
  const alignment = Number(kv.get("general.alignment") ?? 32);
  const tensors: TensorInfo[] = [];
  for (let i = 0; i < nTensors; i++) {
    const name = cursor.str();
    const dims = cursor.u32();
    const ne: number[] = [];
    for (let dim = 0; dim < dims; dim++) ne.push(cursor.u64());
    const type = cursor.u32();
    const offset = cursor.u64();
    tensors.push({ name, ne, type, offset });
  }
  const dataOffset = Math.ceil(cursor.pos / alignment) * alignment;
  for (const tensor of tensors) tensor.offset += dataOffset; // absolute in the file
  return { path, version, alignment, dataOffset, tensors, kv };
}

/** Read the header from a growing prefix: start at 8 MB, double until the tensor table fits. */
export async function readGguf(fs: FileSystem, path: string): Promise<GgufFile> {
  const file = await fs.stat(path);
  if (!file) throw new Error(`${path} does not exist`);
  for (let prefix = 8 << 20; ; prefix *= 2) {
    const bytes = await fs.readRange(path, 0, Math.min(prefix, file.size));
    try {
      return parseGgufHeader(path, bytes);
    } catch (error) {
      if (!(error instanceof NeedMore) || prefix >= file.size) throw error;
    }
  }
}

export async function readTensorF32(
  fs: FileSystem,
  file: GgufFile,
  tensor: TensorInfo,
): Promise<Float32Array> {
  const raw = await fs.readRange(file.path, tensor.offset, tensorBytes(tensor));
  const count = elementCount(tensor);
  if (tensor.type === GgmlType.F32)
    return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  if (tensor.type === GgmlType.F16) {
    const out = new Float32Array(count);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    for (let i = 0; i < count; i++) out[i] = f16ToF32(view.getUint16(i * 2, true));
    return out;
  }
  throw new Error(`${tensor.name}: cannot read type ${tensor.type} as f32`);
}

/** IEEE half → single, exact (every f16 is representable in f32). */
export function f16ToF32(half: number): number {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >> 10) & 0x1f;
  const mantissa = half & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24; // zero / subnormal
  if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}
