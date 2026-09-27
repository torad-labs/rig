// -np / -c for the card, from the head's tiers (largest first) and the VRAM the head can have on it. The tiers
// are measurements, not a formula — utilization runs 77–96% across the four measured cards — and
// the head's constants are what each tier is checked against at load (kernel/head/schema).
import type { Gpu, GpuInfo, Host } from "../ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../result.ts";
import type { Head } from "./head.ts";
import type { Tier } from "./head-config.ts";

export function pickTier(head: Head, vramMiB: number): Result<Tier> {
  const tier = head.geometry.tiers.find((tier) => vramMiB >= tier.min_vram_mib);
  if (!tier)
    return fail(
      ExitCode.Unsupported,
      `${vramMiB} MiB of VRAM for it is below the smallest tier this head declares (${head.geometry.tiers.at(-1)!.min_vram_mib} MiB)`,
    );
  return ok(tier);
}

/** The VRAM a head can have on `card`: its total less what every other process holds, since a
 *  card that also drives a desktop keeps its compositor's and browser's share (2.3 GB of a 16 GB
 *  card here, where the 16 GB tier then failed to allocate). What holds the head's port is the head
 *  itself, already serving, and its share is its own. */
export async function headVramMiB(
  deps: { gpu: Gpu; host: Host },
  card: GpuInfo,
  port: number,
): Promise<number> {
  const pid = await deps.host.listeningPid(port);
  const own = pid === null ? 0 : await deps.gpu.processMiB(card.index, pid);
  return card.memoryMiB - Math.max(0, card.usedMiB - own);
}
