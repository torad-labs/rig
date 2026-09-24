// -np / -c for the card, from the head's tiers (largest first) and the card's VRAM. The tiers
// are measurements, not a formula — utilization runs 77–96% across the four measured cards — and
// the head's constants are what each tier is checked against at load (kernel/head/schema).
import type { Head } from "../../shared/head/head.ts";
import type { Tier } from "../../shared/head/head-config.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";

export function pickTier(head: Head, vramMiB: number): Result<Tier> {
  const tier = head.geometry.tiers.find((tier) => vramMiB >= tier.min_vram_mib);
  if (!tier)
    return fail(
      ExitCode.Unsupported,
      `${vramMiB} MiB of VRAM is below the smallest tier this head declares (${head.geometry.tiers.at(-1)!.min_vram_mib} MiB)`,
    );
  return ok(tier);
}

/** --cache-ram: a quarter of the RAM this process may use, capped at 32 GiB — the rule for a
 *  box that is ours alone. A shared host sets its own bound (the unit keeps it). */
export const CACHE_RAM_CAP_MIB = 32768;
export const defaultCacheRam = (ramMiB: number) =>
  Math.min(Math.floor(ramMiB / 4), CACHE_RAM_CAP_MIB);
