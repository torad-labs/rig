/** --cache-ram: a quarter of the RAM this process may use, capped at 32 GiB — the rule for a
 *  box that is ours alone. A shared host sets its own bound (the unit keeps it). */
export const CACHE_RAM_CAP_MIB = 32768;
export const defaultCacheRam = (ramMiB: number) =>
  Math.min(Math.floor(ramMiB / 4), CACHE_RAM_CAP_MIB);
