// -np / -c for the card, from the head's tiers (largest first) and the card's VRAM. The tiers
// are measurements, not a formula — utilization runs 77–96% across the four measured cards — and
// the head's constants are what each tier is checked against at load (kernel/head/schema).
import { ExitCode, fail, ok, type Result } from "../result.ts";
import type { Head } from "./head.ts";
import type { Tier } from "./head-config.ts";

export function pickTier(head: Head, vramMiB: number): Result<Tier> {
  const tier = head.geometry.tiers.find((tier) => vramMiB >= tier.min_vram_mib);
  if (!tier)
    return fail(
      ExitCode.Unsupported,
      `${vramMiB} MiB of VRAM is below the smallest tier this head declares (${head.geometry.tiers.at(-1)!.min_vram_mib} MiB)`,
    );
  return ok(tier);
}
