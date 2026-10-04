// The card the gate legs run on, claimed before anything starts. Never a card the head is
// serving from while it serves (on a workstation gates.toml declares another card; on a rented
// box with one card --gpu 0 is right once the head is stopped); a CUDA card at that index; a
// complete engine build for its sm; and both pinned packs intact, since the probes compare them.
// The legs run on one card, so a head whose every profile spans several is refused here.

import type { FileSystem, Gpu, Hasher, Host } from "@rig/core";
import { artifactProblem, checkArtifact, ExitCode, fail, ok, type Result } from "@rig/core";
import { type Engine, isBuilt } from "@rig/engine";
import type { Head } from "@rig/head";
import {
  type CacheFormats,
  cacheRefusal,
  describeProfile,
  pickProfile,
  profileCache,
  profileEngineEnv,
} from "@rig/head";

export interface GateCard {
  gpu: number;
  binDir: string;
  /** the card's compute capability, 120 for sm_120 */
  cap: string;
  /** the cache formats of the profile serve would give this card (the head's own when it fits none): the legs run
   *  what serve runs there */
  cache: CacheFormats;
  /** that profile's min_vram_mib, or null when the card fits none */
  profile: number | null;
  /** the engine switches of that profile as serve sets them (profileEngineEnv), its CUDA graph cap among them; with no
   *  profile, the defaults */
  engineEnv: Record<string, string>;
}

export interface GateCardDeps {
  fs: FileSystem;
  gpu: Gpu;
  hasher: Hasher;
  host: Host;
}

export async function claimGateCard(
  deps: GateCardDeps,
  engine: Engine,
  head: Head,
  gpu: number,
): Promise<Result<GateCard>> {
  if (head.profiles.every((profile) => profile.devices > 1)) {
    const spans = head.profiles.map(describeProfile).join("; ");
    return fail(
      ExitCode.Unsupported,
      `the gate legs run on one card, and every profile of ${head.name} spans several (${spans})`,
    );
  }
  const serving = await deps.host.listeningPid(head.port);
  if (serving !== null && (await deps.gpu.processMiB(gpu, serving)) > 0) {
    const message =
      `GPU ${gpu} is a card ${head.name} is serving from (pid ${serving} holds :${head.port} and memory on it) — ` +
      "gates run on another card, or stop the head first";
    return fail(ExitCode.Busy, message);
  }

  const card = await deps.gpu.query(gpu);
  if (!card) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${gpu}`);

  const room = { index: gpu, cap: card.computeCap, vramMiB: card.memoryMiB - card.usedMiB };
  const placed = pickProfile(head, [room]);
  const cache = profileCache(head, placed.ok ? placed.value.profile : {});
  // the gate's drafted legs load the head's draft whatever the profile says
  const refusal = cacheRefusal(engine, cache, head.speculative?.cache);
  if (refusal) return fail(ExitCode.Unsupported, `REFUSING on GPU ${gpu}'s profile: ${refusal}`);

  const binDir = engine.binDir(card.computeCap);
  if (!(await isBuilt(deps.fs, binDir))) {
    const message = `no complete build at ${binDir} for sm_${card.computeCap} (run: rig build --gpu ${gpu})`;
    return fail(ExitCode.Failure, message);
  }

  for (const pack of [...head.sourceFiles, ...(head.declaredServed ? head.servedFiles : [])]) {
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
    profile: placed.ok ? placed.value.profile.min_vram_mib : null,
    engineEnv: profileEngineEnv(placed.ok ? placed.value.profile : {}),
  });
}
