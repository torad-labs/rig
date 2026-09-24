// Produces a head's served pack from its source pack, reproducibly: the derive step is data in
// head.toml, the output is pinned by sha256, and "it produced something" is not a result. The
// edit IS the model: different bytes are a different edit, so a machine whose bake does not
// match the pin refuses to serve it (fatal, never a warning). The bake writes a sibling of the
// served path and renames it in only after the hash matched: a running llama-server has the
// served file mmapped, and an in-place write would be a SIGBUS in the live head.
import { basename } from "node:path";
import {
  artifactProblem,
  checkArtifact,
  publishArtifact,
  stagingPath,
} from "../../shared/artifact.ts";
import type { Head } from "../../shared/head/head.ts";
import type { Derive } from "../../shared/head/head-config.ts";
import type { FileSystem, Hasher, Log } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import {
  GgmlType,
  type GgufFile,
  readGguf,
  readTensorF32,
  type TensorInfo,
  tensorBytes,
} from "./gguf.ts";
import { ablateTensor, directionFromLoraB } from "./lattice-ablation.ts";

export interface DerivePackReport {
  path: string;
  state: "none" | "undrived" | "present" | "derived";
  /** why this machine serves less than the declared served pack (the public pack, "present" or
   *  "derived", or the source, "undrived"): a caller with only this report (a rented box's --json,
   *  past the console this machine's own log went to) still gets the reason, not just the state */
  reason?: string;
  flipped?: number;
  digits?: number;
  /** tensors a draft-head splice wrote */
  spliced?: number;
}

export interface DerivePackDeps {
  fs: FileSystem;
  hasher: Hasher;
  log: Log;
}

type LatticeAblation = Extract<Derive, { kind: "pq2-lattice-ablation" }>;
type DraftHeadSplice = Extract<Derive, { kind: "draft-head-splice" }>;

interface EditStats {
  flipped?: number;
  digits?: number;
  spliced?: number;
}

/** the tensors that write into the residual stream, the ones an ablation edits */
const RESIDUAL_WRITER = /^blk\.(\d+)\.(ffn_down|ssm_out|attn_output)\.weight$/;

export class DerivePack {
  constructor(private readonly deps: DerivePackDeps) {}

  async run(head: Head): Promise<Result<DerivePackReport>> {
    const reason = head.undrived ? { reason: head.undrived } : {};
    if (head.undrived) this.deps.log.info(head.undrived);
    if (!head.derive) {
      return ok({ path: head.servedPath, state: head.undrived ? "undrived" : "none", ...reason });
    }
    const served = { path: head.servedPath, sha256: head.served.sha256 };
    const current = await checkArtifact(this.deps.fs, this.deps.hasher, served);
    if (current === "ok") return ok({ path: served.path, state: "present", ...reason });
    if (typeof current === "object") {
      const message = `${served.path} is ${artifactProblem(current)} — refusing to derive over it`;
      return fail(ExitCode.Failure, message);
    }
    const source = await checkArtifact(this.deps.fs, this.deps.hasher, {
      path: head.sourcePath,
      sha256: head.source.sha256,
    });
    if (source !== "ok") {
      const message = `the source pack is ${artifactProblem(source)} at ${head.sourcePath} — run fetch first`;
      return fail(ExitCode.Failure, message);
    }

    const staged = stagingPath(served.path, "deriving");
    await this.deps.fs.remove(staged);
    const kinds = head.derive.map((step) => step.kind).join(", then ");
    this.deps.log.info(
      `deriving ${basename(served.path)} from ${basename(head.sourcePath)} (${kinds})`,
    );
    await this.deps.fs.copy(head.sourcePath, staged);
    const stats: EditStats = {};
    for (const step of head.derive) {
      const done = await this.apply(head, step, staged);
      if (!done.ok) {
        await this.deps.fs.remove(staged);
        return done;
      }
      for (const [key, count] of Object.entries(done.value) as [keyof EditStats, number][]) {
        stats[key] = (stats[key] ?? 0) + count;
      }
    }

    const state = await publishArtifact(this.deps.fs, this.deps.hasher, staged, served);
    if (typeof state === "object" && "unreadable" in state) {
      await this.deps.fs.remove(staged);
      return fail(
        ExitCode.Failure,
        `the derived pack could not be read back (${state.unreadable}): ${staged}`,
      );
    }
    if (state !== "ok") {
      const produced = state === "missing" ? "nothing" : state.mismatch;
      const message = `the derive step produced a DIFFERENT edit than the pinned one (sha256 ${produced}, pinned ${served.sha256}) — removed, refusing to serve it`;
      return fail(ExitCode.Failure, message);
    }
    return ok({ path: served.path, state: "derived", ...stats, ...reason });
  }

  private apply(head: Head, derive: Derive, staged: string): Promise<Result<EditStats>> {
    switch (derive.kind) {
      case "pq2-lattice-ablation":
        return this.ablate(head, derive, staged);
      case "draft-head-splice":
        return this.splice(head, derive, staged);
    }
  }

  /** the draft head's tensors written over the pack's own, in place in the staged copy: each one
   *  must already be in the pack with the same type and shape, so a splice replaces bytes and never
   *  a layout (a new tensor or shape would be a different pack format, not an edit) */
  private async splice(
    head: Head,
    step: DraftHeadSplice,
    staged: string,
  ): Promise<Result<EditStats>> {
    const asset = { path: head.assetPath(step), sha256: step.head_sha256 };
    const state = await checkArtifact(this.deps.fs, this.deps.hasher, asset);
    if (state !== "ok") {
      return fail(ExitCode.Failure, `the draft head is ${artifactProblem(state)}: ${asset.path}`);
    }
    const donor = await readGguf(this.deps.fs, asset.path);
    if (donor.tensors.length === 0) {
      return fail(ExitCode.Failure, `the draft head ${step.head} holds no tensors`);
    }
    const pack = new Map((await readGguf(this.deps.fs, staged)).tensors.map((t) => [t.name, t]));
    for (const tensor of donor.tensors) {
      const into = pack.get(tensor.name);
      if (!into) {
        return fail(
          ExitCode.Failure,
          `${tensor.name} (in ${step.head}) is not a tensor of the pack`,
        );
      }
      if (into.type !== tensor.type || into.ne.join("x") !== tensor.ne.join("x")) {
        const message = `${tensor.name}: the draft head's is type ${tensor.type} [${tensor.ne.join(", ")}], the pack's type ${into.type} [${into.ne.join(", ")}] — a splice replaces bytes, never a layout`;
        return fail(ExitCode.Failure, message);
      }
    }
    for (const tensor of donor.tensors) {
      const bytes = await this.deps.fs.readRange(asset.path, tensor.offset, tensorBytes(tensor));
      await this.deps.fs.writeAt(staged, (pack.get(tensor.name) as TensorInfo).offset, bytes);
    }
    this.deps.log.info(`spliced ${donor.tensors.length} draft-head tensors from ${step.head}`);
    return ok({ spliced: donor.tensors.length });
  }

  /** the refusal direction from the adapter's B matrices, removed from every residual writer
   *  in the named blocks, in place in the staged copy */
  private async ablate(
    head: Head,
    derive: LatticeAblation,
    staged: string,
  ): Promise<Result<EditStats>> {
    const direction = await this.refusalDirection(head, derive);
    if (!direction.ok) return direction;

    const pack = await readGguf(this.deps.fs, staged);
    const writers = residualWriters(pack, derive.blocks);
    if (writers.length === 0) {
      return fail(ExitCode.Failure, `no residual-writer tensors in blocks ${derive.blocks}`);
    }

    const params = { rows: derive.rows, lambda: derive.lambda, rowCap: derive.row_cap };
    const total = { flipped: 0, digits: 0 };
    for (const tensor of writers) {
      if (tensor.type !== GgmlType.PQ2_0) {
        return fail(ExitCode.Failure, `${tensor.name} is ggml type ${tensor.type}, not PQ2_0`);
      }
      const columns = tensor.ne[0] ?? 0;
      const rows = tensor.ne[1] ?? 0;
      const raw = await this.deps.fs.readRange(staged, tensor.offset, tensorBytes(tensor));
      const { out, stats } = ablateTensor(raw, rows, columns, direction.value, params);
      await this.deps.fs.writeAt(staged, tensor.offset, out);
      total.flipped += stats.flipped;
      total.digits += stats.digits;
      this.deps.log.info(
        `  ${tensor.name.padEnd(28)} flipped ${String(stats.flipped).padStart(8)} (${percent(stats.flipped, stats.digits)}%)  component removed ${(100 * stats.removed).toFixed(1)}%`,
      );
    }
    this.deps.log.info(
      `total digits flipped ${total.flipped} of ${total.digits} (${percent(total.flipped, total.digits)}%)`,
    );
    return ok(total);
  }

  /** the adapter the head pins, read, its lora_b tensors folded into one direction */
  private async refusalDirection(
    head: Head,
    derive: LatticeAblation,
  ): Promise<Result<Float32Array>> {
    const loraPath = head.assetPath(derive);
    const lora = await checkArtifact(this.deps.fs, this.deps.hasher, {
      path: loraPath,
      sha256: derive.lora_sha256,
    });
    if (lora !== "ok") {
      return fail(ExitCode.Failure, `the adapter is ${artifactProblem(lora)}: ${loraPath}`);
    }
    const adapter = await readGguf(this.deps.fs, loraPath);
    const bMatrices: Float32Array[] = [];
    for (const tensor of adapter.tensors) {
      if (tensor.name.endsWith(".lora_b")) {
        bMatrices.push(await readTensorF32(this.deps.fs, adapter, tensor));
      }
    }
    return ok(directionFromLoraB(bMatrices));
  }
}

/** the residual writers in an inclusive block range "lo-hi" */
function residualWriters(pack: GgufFile, blocks: string): TensorInfo[] {
  const [lo, hi] = blocks.split("-").map(Number) as [number, number];
  return pack.tensors.filter((tensor) => {
    const match = RESIDUAL_WRITER.exec(tensor.name);
    if (!match) return false;
    const block = Number(match[1]);
    return block >= lo && block <= hi;
  });
}

const percent = (part: number, whole: number) => ((100 * part) / whole).toFixed(3);
