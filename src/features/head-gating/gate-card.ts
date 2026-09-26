// The card the gate legs run on, claimed before anything starts. Never the card the head is
// serving from while it serves (on a workstation gates.toml declares another card; on a rented
// box with one card --gpu 0 is right once the head is stopped); a CUDA card at that index; a
// complete engine build for its sm; and both pinned packs intact, since the probes compare them.
import { artifactProblem, checkArtifact } from "../../shared/artifact.ts";
import { type Engine, isBuilt } from "../../shared/engine/engine.ts";
import { type CacheFormats, cacheRefusal, tierCache } from "../../shared/head/cache-formats.ts";
import type { Head } from "../../shared/head/head.ts";
import { pickTier } from "../../shared/head/tier.ts";
import type { FileSystem, Gpu, Hasher, Http } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";
import { LlamaClient } from "./llama-client.ts";

export interface GateCard {
  gpu: number;
  binDir: string;
  /** the card's compute capability, 120 for sm_120 */
  cap: string;
  /** the cache formats of the tier serve would give this card (the head's own when it is below every tier): the
   *  legs run what serve runs there */
  cache: CacheFormats;
  /** that tier's min_vram_mib, or null when the card is below every tier */
  tier: number | null;
}

export interface GateCardDeps {
  fs: FileSystem;
  gpu: Gpu;
  hasher: Hasher;
  http: Http;
}

export async function claimGateCard(
  deps: GateCardDeps,
  engine: Engine,
  head: Head,
  gpu: number,
): Promise<Result<GateCard>> {
  if (gpu === head.gpu) {
    const servingHead = new LlamaClient(deps.http, `http://127.0.0.1:${head.port}`);
    if (await servingHead.healthy()) {
      const message =
        `GPU ${gpu} is the card ${head.name} is serving from (:${head.port} answers) — ` +
        "gates run on another card, or stop the head first";
      return fail(ExitCode.Busy, message);
    }
  }

  const card = await deps.gpu.query(gpu);
  if (!card) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${gpu}`);

  const tier = pickTier(head, card.memoryMiB - card.usedMiB);
  const cache = tierCache(head, tier.ok ? tier.value : {});
  // the gate's drafted legs load the head's draft whatever the tier says
  const refusal = cacheRefusal(engine, cache, head.speculative?.cache);
  if (refusal) return fail(ExitCode.Unsupported, `REFUSING on GPU ${gpu}'s tier: ${refusal}`);

  const binDir = engine.binDir(card.computeCap);
  if (!(await isBuilt(deps.fs, binDir))) {
    const message = `no complete build at ${binDir} for sm_${card.computeCap} (run: rig build --gpu ${gpu})`;
    return fail(ExitCode.Failure, message);
  }

  const pinnedPacks = [
    { path: head.sourcePath, sha256: head.source.sha256 },
    { path: head.servedPath, sha256: head.served.sha256 },
  ];
  for (const pack of pinnedPacks) {
    const state = await checkArtifact(deps.fs, deps.hasher, pack);
    if (state !== "ok") {
      const message = `${pack.path} is ${artifactProblem(state)} — the gates compare the pinned packs (run: rig fetch / rig derive)`;
      return fail(ExitCode.Failure, message);
    }
  }

  return ok({
    gpu,
    binDir,
    cap: card.computeCap,
    cache,
    tier: tier.ok ? tier.value.min_vram_mib : null,
  });
}
