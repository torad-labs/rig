import { describe, expect, test } from "bun:test";
import { f32Bytes, pq2Block, writeGguf } from "../../../test/fakes/gguf.ts";
import { fakePorts } from "../../../test/fakes/index.ts";
import { loadHead } from "../../shared/head/head.ts";
import { deriveAsset } from "../../shared/head/head-config.ts";
import { layoutAt } from "../../shared/layout.ts";
import { GgmlType, parseGgufHeader, readGguf, relayoutGguf } from "./gguf.ts";
import { DerivePack } from "./pack-derivation.service.ts";
import { unpack } from "./pq2.ts";

// a two-block pack: blk.0 and blk.1 each with an ffn_down (target) and an attn_q (never touched),
// 4 rows x 128 cols of +1 digits at scale 1.0, and a rank-1 adapter whose direction is row 0
const N = 4,
  K = 128;
const rows = () =>
  new Uint8Array(Array.from({ length: N }, () => [...pq2Block(0x3c00, Array(K).fill(1))]).flat());
const packBytes = () =>
  writeGguf(
    [
      { name: "blk.0.ffn_down.weight", ne: [K, N], type: GgmlType.PQ2_0, data: rows() },
      { name: "blk.0.attn_q.weight", ne: [K, N], type: GgmlType.PQ2_0, data: rows() },
      { name: "blk.1.ffn_down.weight", ne: [K, N], type: GgmlType.PQ2_0, data: rows() },
      { name: "blk.2.ffn_down.weight", ne: [K, N], type: GgmlType.PQ2_0, data: rows() }, // outside blocks = "0-1"
      ...headTensors(0x11, 1),
    ],
    { "general.architecture": "test" },
  );
// a draft head block (blk.3, the MTP layer's place): one Q8_0 matrix, 2 rows x 32, and an F32 norm
function headTensors(fill: number, norm: number) {
  return [
    {
      name: "blk.3.nextn.eh_proj.weight",
      ne: [32, 2],
      type: GgmlType.Q8_0,
      data: new Uint8Array(2 * 34).fill(fill),
    },
    {
      name: "blk.3.nextn.enorm.weight",
      ne: [4],
      type: GgmlType.F32,
      data: f32Bytes([norm, norm, norm, norm]),
    },
  ];
}
const draftHeadBytes = () => writeGguf(headTensors(0x22, 2));
const loraBytes = () =>
  writeGguf([
    {
      name: "blk.0.ffn_down.weight.lora_a",
      ne: [K, 1],
      type: GgmlType.F32,
      data: f32Bytes(Array(K).fill(0.5)),
    },
    {
      name: "blk.0.ffn_down.weight.lora_b",
      ne: [1, N],
      type: GgmlType.F32,
      data: f32Bytes([2, 0, 0, 0]),
    },
    {
      name: "blk.1.ffn_down.weight.lora_b",
      ne: [1, N],
      type: GgmlType.F32,
      data: f32Bytes([-3, 0, 0, 0]),
    },
  ]);
const sha = (b: Uint8Array) => new Bun.CryptoHasher("sha256").update(b).digest("hex");
const BF16 = 30; // a type the reader cannot size, as the real pack's ssm_alpha and ssm_beta are

function headToml(o: {
  servedSha: string;
  sourceSha: string;
  loraSha: string;
  derive: boolean;
  splice?: string;
  /** a public draft-head-splice step (a url) ahead of the ablation, and [public] pinning its output */
  publicSplice?: { headSha: string; publicSha: string };
}) {
  return `
name = "tiny"
title = "tiny"
port = 9000
[source]
repo = "x/y"
rev = "${"a".repeat(40)}"
file = "tiny.gguf"
sha256 = "${o.sourceSha}"
bytes = 1
[served]
file = "${o.derive ? "tiny-ablated.gguf" : "tiny.gguf"}"
sha256 = "${o.servedSha}"
bytes = 1
${
  o.publicSplice
    ? `[public]
file = "tiny-public.gguf"
sha256 = "${o.publicSplice.publicSha}"
bytes = 1
[[derive]]
kind = "draft-head-splice"
head = "draft-head.gguf"
head_sha256 = "${o.publicSplice.headSha}"
head_bytes = 1
url = "https://example.com/draft-head.gguf"`
    : ""
}
${
  o.derive
    ? `[[derive]]
kind = "pq2-lattice-ablation"
lora = "assets/lora.gguf"
lora_sha256 = "${o.loraSha}"
blocks = "0-1"
rows = 1
lambda = 1.0
row_cap = 1.0`
    : ""
}
${
  o.splice
    ? `[[derive]]
kind = "draft-head-splice"
head = "assets/draft-head.gguf"
head_sha256 = "${o.splice}"`
    : ""
}
[context]
model = 1024
advertise = 1024
[geometry]
kv_bytes_per_token = 16
compute_bytes_per_token = 0
weights_mib = 1
state_per_slot_mib = 0
compute_mib = 0
compute_per_output_row_mib = 0
tiers = [{ min_vram_mib = 100, slots = 1, ctx = 1024 }]
[runtime]
args = []
[client]
rejects_reasoning_effort = true
slot_pinning = true
any_model_id = true
`;
}

async function setup(
  o: {
    servedSha?: string;
    sourceSha?: string;
    derive?: boolean;
    source?: boolean;
    lora?: boolean;
    /** a draft-head-splice step after the ablation, its asset on disk unless false */
    splice?: boolean;
    /** the public splice ahead of the ablation, its asset in local/packs unless draftHead is false */
    publicSplice?: boolean;
    publicSha?: string;
    draftHead?: Uint8Array | false;
    ports?: ReturnType<typeof fakePorts>;
  } = {},
) {
  const p = o.ports ?? fakePorts();
  const layout = layoutAt("/r");
  const source = packBytes(),
    lora = loraBytes();
  p.fs.put(
    "/r/heads/tiny/head.toml",
    headToml({
      servedSha: o.servedSha ?? "0".repeat(64),
      sourceSha: o.sourceSha ?? sha(source),
      loraSha: sha(lora),
      derive: o.derive ?? true,
      ...(o.splice ? { splice: sha(draftHeadBytes()) } : {}),
      ...(o.publicSplice
        ? {
            publicSplice: {
              headSha: sha(draftHeadBytes()),
              publicSha: o.publicSha ?? "1".repeat(64),
            },
          }
        : {}),
    }),
  );
  if (o.source ?? true) await p.fs.writeBytes("/r/local/packs/tiny/tiny.gguf", source);
  if (o.lora ?? true) await p.fs.writeBytes("/r/heads/tiny/assets/lora.gguf", lora);
  if (o.splice && o.draftHead !== false)
    await p.fs.writeBytes("/r/heads/tiny/assets/draft-head.gguf", o.draftHead ?? draftHeadBytes());
  if (o.publicSplice && o.draftHead !== false)
    await p.fs.writeBytes("/r/local/packs/tiny/draft-head.gguf", o.draftHead ?? draftHeadBytes());
  const head = await loadHead(p.fs, layout, "tiny");
  if (!head.ok) throw new Error(head.message);
  return { p, head: head.value, uc: new DerivePack(p) };
}

/** what the bake must produce for this fixture: row 0 of every target flipped +1 -> 0, the rest untouched */
async function expectedOutput(
  o: { splice?: boolean; publicSplice?: boolean; lora?: boolean } = {},
) {
  const { p, head, uc } = await setup(o);
  p.hasher.pinned.set(`${head.servedPath}.deriving`, head.served.sha256); // let publish accept whatever came out
  const r = await uc.run(head);
  if (!r.ok) throw new Error(r.message);
  return p.fs.files.get(head.servedPath)!;
}

describe("derive", () => {
  test("a head without [derive] serves its source (state none)", async () => {
    const { head, uc } = await setup({ derive: false, servedSha: sha(packBytes()) });
    expect(await uc.run(head)).toEqual({
      ok: true,
      value: { path: head.servedPath, state: "none" },
    });
  });
  test("a served pack already at the pinned sha is left alone (state present)", async () => {
    const { p, head, uc } = await setup({ servedSha: sha(new TextEncoder().encode("served")) });
    p.fs.put(head.servedPath, "served");
    const r = await uc.run(head);
    expect(r.ok && r.value.state).toBe("present");
    expect(p.fs.renames).toEqual([]);
  });
  test("a served pack that is there but unreadable is refused by its errno, never derived over", async () => {
    const { p, head, uc } = await setup();
    p.fs.put(head.servedPath, "served");
    p.fs.deny(head.servedPath, "EIO");
    const r = await uc.run(head);
    expect(!r.ok && r.message).toContain("unreadable (EIO) — refusing to derive over it");
    expect(await p.fs.exists(`${head.servedPath}.deriving`)).toBe(false);
    expect(p.fs.renames).toEqual([]);
  });
  test("a missing or wrong source is refused before anything is written", async () => {
    const { head, uc } = await setup({ source: false });
    let r = await uc.run(head);
    expect(!r.ok && r.message).toContain("source pack is missing");
    const w = await setup({ sourceSha: "1".repeat(64) });
    r = await w.uc.run(w.head);
    expect(!r.ok && r.message).toContain("source pack is not the pinned bytes (sha256 differs)");
  });
  test("the edit flips only the target tensors in the block range, and only the direction's rows", async () => {
    const out = await expectedOutput();
    const f = parseGgufHeader("out", out);
    const digitsOf = (name: string) => {
      const t = f.tensors.find((x) => x.name === name)!;
      return Array.from(unpack(out.subarray(t.offset, t.offset + N * 34), N, K).digits);
    };
    for (const target of ["blk.0.ffn_down.weight", "blk.1.ffn_down.weight"]) {
      const d = digitsOf(target);
      expect(d.slice(0, K).every((x) => x === 0)).toBe(true); // row 0 spent: +1 -> 0 removes c_k = 1
      expect(d.slice(K).every((x) => x === 1)).toBe(true); // rows 1..3 untouched
    }
    for (const untouched of ["blk.0.attn_q.weight", "blk.2.ffn_down.weight"])
      expect(digitsOf(untouched).every((x) => x === 1)).toBe(true);
  });
  test("the output is published only when its sha256 is the pinned one; a different edit is removed", async () => {
    const expected = await expectedOutput();
    const good = await setup({ servedSha: sha(expected) });
    let r = await good.uc.run(good.head);
    expect(r.ok && r.value.state).toBe("derived");
    expect(r.ok && r.value.flipped).toBe(2 * K);
    expect(good.p.fs.renames).toEqual([[`${good.head.servedPath}.deriving`, good.head.servedPath]]);
    const bad = await setup({ servedSha: "f".repeat(64) });
    r = await bad.uc.run(bad.head);
    expect(!r.ok && r.message).toContain("DIFFERENT edit");
    expect(bad.p.fs.files.has(bad.head.servedPath)).toBe(false);
    expect(bad.p.fs.files.has(`${bad.head.servedPath}.deriving`)).toBe(false);
  });
  test("a wrong adapter is refused and the staged copy removed", async () => {
    const { p, head, uc } = await setup();
    await p.fs.writeBytes(head.path("assets/lora.gguf"), writeGguf([]));
    const r = await uc.run(head);
    expect(!r.ok && r.message).toContain("adapter is not the pinned bytes (sha256 differs)");
    expect(p.fs.files.has(`${head.servedPath}.deriving`)).toBe(false);
  });
  test("no adapter and no derived pack on the machine: the source pack is served, undrived", async () => {
    const { head, uc } = await setup({ lora: false });
    const reason = head.undrived;
    expect(reason).toContain("assets/lora.gguf");
    if (!reason) throw new Error("fixture: head is not undrived");
    expect(head.derive).toBeUndefined();
    expect(head.servedPath).toBe(head.sourcePath);
    expect(head.served).toEqual(head.source);
    expect(await uc.run(head)).toEqual({
      ok: true,
      value: { path: "/r/local/packs/tiny/tiny.gguf", state: "undrived", reason },
    });
  });
  test("an adapter gone missing keeps a pack already derived: never a swap to the source", async () => {
    const p = fakePorts();
    await p.fs.writeBytes("/r/local/packs/tiny/tiny-ablated.gguf", packBytes());
    const loaded = await setup({ lora: false, ports: p });
    expect(loaded.head.undrived).toBeUndefined();
    expect(loaded.head.servedPath).toBe("/r/local/packs/tiny/tiny-ablated.gguf");
    p.hasher.pinned.set(loaded.head.servedPath, "0".repeat(64));
    expect(await loaded.uc.run(loaded.head)).toEqual({
      ok: true,
      value: { path: "/r/local/packs/tiny/tiny-ablated.gguf", state: "present" },
    });
  });
  test("lora_b vectors that disagree on the direction are refused", async () => {
    const { p, head, uc } = await setup();
    const lora = writeGguf([
      { name: "a.lora_b", ne: [1, N], type: GgmlType.F32, data: f32Bytes([1, 0, 0, 0]) },
      { name: "b.lora_b", ne: [1, N], type: GgmlType.F32, data: f32Bytes([0, 1, 0, 0]) },
    ]);
    await p.fs.writeBytes(head.path("assets/lora.gguf"), lora);
    p.hasher.pinned.set(head.path("assets/lora.gguf"), deriveAsset(head.derive![0]!).sha256);
    await expect(uc.run(head)).rejects.toThrow("not one direction");
  });
});

describe("derive: draft-head splice", () => {
  const tensorBytesOf = (file: Uint8Array, name: string, length: number) => {
    const t = parseGgufHeader("f", file).tensors.find((x) => x.name === name)!;
    return Array.from(file.subarray(t.offset, t.offset + length));
  };
  test("after the ablation, the draft head's tensors replace the pack's byte for byte and nothing else moves", async () => {
    const ablatedOnly = await expectedOutput();
    const out = await expectedOutput({ splice: true });
    expect(tensorBytesOf(out, "blk.3.nextn.eh_proj.weight", 68)).toEqual(Array(68).fill(0x22));
    expect(tensorBytesOf(out, "blk.3.nextn.enorm.weight", 16)).toEqual(
      Array.from(f32Bytes([2, 2, 2, 2])),
    );
    for (const other of [
      "blk.0.ffn_down.weight",
      "blk.0.attn_q.weight",
      "blk.1.ffn_down.weight",
      "blk.2.ffn_down.weight",
    ])
      expect(tensorBytesOf(out, other, N * 34)).toEqual(tensorBytesOf(ablatedOnly, other, N * 34));
    const good = await setup({ splice: true, servedSha: sha(out) });
    const r = await good.uc.run(good.head);
    expect(r.ok && r.value).toMatchObject({ state: "derived", flipped: 2 * K, spliced: 2 });
  });
  test("a head tensor of another type re-lays the pack out: that tensor takes the head's type and bytes, the rest keep theirs, the metadata stays", async () => {
    const q4 = writeGguf([
      {
        name: "blk.3.nextn.eh_proj.weight",
        ne: [32, 2],
        type: GgmlType.Q4_0,
        data: new Uint8Array(2 * 18).fill(0x44),
      },
      ...headTensors(0x22, 2).slice(1),
    ]);
    const { p, head, uc } = await setup({ splice: true, draftHead: q4 });
    p.hasher.pinned.set(head.path("assets/draft-head.gguf"), deriveAsset(head.derive![1]!).sha256);
    p.hasher.pinned.set(`${head.servedPath}.deriving`, head.served.sha256);
    const r = await uc.run(head);
    expect(r.ok && r.value).toMatchObject({ state: "derived", flipped: 2 * K, spliced: 2 });
    const out = p.fs.files.get(head.servedPath)!;
    const ablatedOnly = await expectedOutput();
    const [after, before] = [parseGgufHeader("out", out), parseGgufHeader("in", ablatedOnly)];
    expect(after.kv).toEqual(before.kv);
    expect(after.tensors.map((t) => [t.name, t.type])).toEqual(
      before.tensors.map((t) => [
        t.name,
        t.name === "blk.3.nextn.eh_proj.weight" ? GgmlType.Q4_0 : t.type,
      ]),
    );
    expect(after.tensors.every((t) => (t.offset - after.dataOffset) % 32 === 0)).toBe(true);
    expect(tensorBytesOf(out, "blk.3.nextn.eh_proj.weight", 36)).toEqual(Array(36).fill(0x44));
    expect(tensorBytesOf(out, "blk.3.nextn.enorm.weight", 16)).toEqual(
      Array.from(f32Bytes([2, 2, 2, 2])),
    ); // after the retyped tensor, so it moved
    for (const other of [
      "blk.0.ffn_down.weight",
      "blk.0.attn_q.weight",
      "blk.1.ffn_down.weight",
      "blk.2.ffn_down.weight",
    ])
      expect(tensorBytesOf(out, other, N * 34)).toEqual(tensorBytesOf(ablatedOnly, other, N * 34));
    expect(out.length).toBe(ablatedOnly.length - 96 + 64); // 68 bytes padded to 96, now 36 padded to 64
    expect([...p.fs.files.keys()].some((path) => path.endsWith(".relayout"))).toBe(false);
  });
  test("a head tensor the pack lacks, of another shape, or of a type rig cannot size, is refused and the staged copy removed", async () => {
    const cases: [Uint8Array, string][] = [
      [
        writeGguf([
          {
            name: "blk.9.nextn.eh_proj.weight",
            ne: [32, 2],
            type: GgmlType.Q8_0,
            data: new Uint8Array(68),
          },
        ]),
        "is not a tensor of the pack",
      ],
      [
        writeGguf([
          {
            name: "blk.3.nextn.eh_proj.weight",
            ne: [32, 1],
            type: GgmlType.Q8_0,
            data: new Uint8Array(34),
          },
        ]),
        "never its shape",
      ],
      [
        writeGguf([
          { name: "blk.3.nextn.enorm.weight", ne: [4], type: BF16, data: new Uint8Array(8) },
        ]),
        "unsupported ggml type 30",
      ],
    ];
    for (const [draftHead, message] of cases) {
      const { p, head, uc } = await setup({ splice: true, draftHead });
      p.hasher.pinned.set(
        head.path("assets/draft-head.gguf"),
        deriveAsset(head.derive![1]!).sha256,
      );
      const r = await uc.run(head);
      expect(!r.ok && r.message).toContain(message);
      expect(p.fs.files.has(`${head.servedPath}.deriving`)).toBe(false);
    }
  });
  test("a draft head that is not the pinned bytes is refused", async () => {
    const { head, uc } = await setup({ splice: true, draftHead: writeGguf(headTensors(0x33, 3)) });
    const r = await uc.run(head);
    expect(!r.ok && r.message).toContain("draft head is not the pinned bytes (sha256 differs)");
  });
  test("every step or none: the adapter without the draft head, and no derived pack, serves the source", async () => {
    const { head } = await setup({ splice: true, draftHead: false });
    expect(head.undrived).toContain("assets/draft-head.gguf");
    expect(head.servedPath).toBe(head.sourcePath);
    const both = await setup({ splice: true });
    expect(both.head.undrived).toBeUndefined();
  });
});

describe("derive: a public draft head", () => {
  const tensorBytesOf = (file: Uint8Array, name: string, length: number) => {
    const t = parseGgufHeader("f", file).tensors.find((x) => x.name === name)!;
    return Array.from(file.subarray(t.offset, t.offset + length));
  };
  test("without the adapter, the public pack is derived: the source with the public head, the ablation's tensors untouched, the reason carried", async () => {
    const out = await expectedOutput({ publicSplice: true, lora: false });
    const source = packBytes();
    expect(tensorBytesOf(out, "blk.3.nextn.eh_proj.weight", 68)).toEqual(Array(68).fill(0x22));
    for (const other of ["blk.0.ffn_down.weight", "blk.1.ffn_down.weight", "blk.0.attn_q.weight"])
      expect(tensorBytesOf(out, other, N * 34)).toEqual(tensorBytesOf(source, other, N * 34));
    const { head, uc } = await setup({ publicSplice: true, lora: false, publicSha: sha(out) });
    const reason = head.undrived;
    if (!reason) throw new Error("fixture: head is not undrived");
    expect(reason).toContain("assets/lora.gguf");
    expect(reason).toContain("serving the public pack tiny-public.gguf");
    expect(head.servedPath).toBe("/r/local/packs/tiny/tiny-public.gguf");
    expect(head.derive?.map((step) => step.kind)).toEqual(["draft-head-splice"]);
    const r = await uc.run(head);
    expect(r.ok && r.value).toEqual({
      path: "/r/local/packs/tiny/tiny-public.gguf",
      state: "derived",
      spliced: 2,
      reason,
    });
  });
  test("with the adapter, the public splice then the ablation are the bytes of the ablation then a private splice: the steps write disjoint tensors", async () => {
    const privateOrder = await expectedOutput({ splice: true });
    const publicFirst = await expectedOutput({ publicSplice: true });
    expect(Array.from(publicFirst)).toEqual(Array.from(privateOrder));
    const { head, uc } = await setup({ publicSplice: true, servedSha: sha(publicFirst) });
    expect(head.undrived).toBeUndefined();
    const r = await uc.run(head);
    expect(r.ok && r.value).toMatchObject({ state: "derived", flipped: 2 * K, spliced: 2 });
  });
  test("a public asset not fetched yet is no reason to go undrived; derive then names it", async () => {
    const { head, uc } = await setup({ publicSplice: true, draftHead: false });
    expect(head.undrived).toBeUndefined(); // `rig fetch` gets it; only a private asset decides
    const r = await uc.run(head);
    expect(!r.ok && r.message).toContain(
      "the draft head is missing: /r/local/packs/tiny/draft-head.gguf",
    );
  });
});

describe("gguf relayout", () => {
  test("a tensor of a type the reader cannot size is copied through by its span, and the tensors after a retyped one move up", async () => {
    const p = fakePorts();
    const source = writeGguf(
      [
        { name: "a", ne: [4], type: BF16, data: Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]) },
        { name: "b", ne: [32, 2], type: GgmlType.Q8_0, data: new Uint8Array(68).fill(9) },
        { name: "c", ne: [4], type: BF16, data: Uint8Array.from([8, 7, 6, 5, 4, 3, 2, 1]) },
      ],
      { "general.architecture": "test" },
    );
    await p.fs.writeBytes("/in.gguf", source);
    const retype = new Map([["b", { type: GgmlType.Q4_0, bytes: new Uint8Array(36).fill(7) }]]);
    await relayoutGguf(p.fs, await readGguf(p.fs, "/in.gguf"), retype, "/out.gguf");
    const out = p.fs.files.get("/out.gguf")!;
    const parsed = parseGgufHeader("out", out);
    const read = (name: string, length: number) => {
      const t = parsed.tensors.find((x) => x.name === name)!;
      return [t.type, Array.from(out.subarray(t.offset, t.offset + length))];
    };
    expect(read("a", 8)).toEqual([BF16, [1, 2, 3, 4, 5, 6, 7, 8]]);
    expect(read("b", 36)).toEqual([GgmlType.Q4_0, Array(36).fill(7)]);
    expect(read("c", 8)).toEqual([BF16, [8, 7, 6, 5, 4, 3, 2, 1]]);
    expect(parsed.tensors.map((t) => t.offset - parsed.dataOffset)).toEqual([0, 32, 96]); // b: 68 bytes in 96, now 36 in 64
    expect(parsed.kv.get("general.architecture")).toBe("test");
    expect(out.length).toBe(source.length - 32);
  });
});
