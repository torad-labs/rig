// What a host's download measured on a box rig rented from it, kept past the box so the next rental is ranked on it. A
// host declares its download speed (vast's inet_down), and the declaration says nothing of the path from where it stands
// to where the pack is served. 2026-10-03, rentals 2b and 2c: box 54020392 (Japan) declared 7,754 Mb/s and its own
// sampler read the pack's pull at 51-64 MB/s; box 53980196 (Texas) declared 5,422 and read 437.5 MB/s.
import { dirname } from "node:path";
import type { FileSystem, Offer } from "@rig/core";

/** one box's best pull, as its sampler read it */
export interface MeasuredPull {
  instanceId: number;
  /** absent for a box created before rig kept its host's machine */
  machineId?: number;
  geo: string;
  mbps: number;
  /** when the best window was read, epoch ms */
  at: number;
}

/** the download rate an offer is ranked on, and where it came from */
export interface DownloadRate {
  mbps: number;
  source: "host" | "region" | "declared";
}

/** A pull is a measurement only when it was a sustained bulk download: 2 GiB, which rig's own payload (the binary and
 *  the pin, a few hundred MB) never reaches, over at least a minute, which the engine's runtime from NVIDIA's CDN (about
 *  2 GB, seconds on a fast host) does not last. A host that never pulls that much in a window is ranked on its
 *  declaration. */
export const MEASURED_MIN_BYTES = 2 * 2 ** 30;
export const MEASURED_MIN_SECONDS = 60;

export class DownloadRates {
  constructor(
    private readonly fs: FileSystem,
    readonly file: string,
  ) {}

  async all(): Promise<MeasuredPull[]> {
    if (!(await this.fs.exists(this.file))) return [];
    return JSON.parse(await this.fs.readText(this.file)) as MeasuredPull[];
  }

  /** A window's pull (box-sampler.ts: the rate over its ticks that pulled, and how long they ran), kept as the box's
   *  when it was a sustained bulk download and beat the box's best. The rate in Mb/s when it was kept, else null. */
  async record(
    box: { instanceId: number; machineId?: number | undefined; geo: string },
    pull: { kibPerSecond: number; seconds: number },
    at: number,
  ): Promise<number | null> {
    const bytes = pull.kibPerSecond * 1024 * pull.seconds;
    if (bytes < MEASURED_MIN_BYTES || pull.seconds < MEASURED_MIN_SECONDS) return null;
    const mbps = Math.round((pull.kibPerSecond * 1024 * 8) / 1e6);
    const pulls = await this.all();
    const best = pulls.find((each) => each.instanceId === box.instanceId);
    if (best && best.mbps >= mbps) return null;
    const kept: MeasuredPull = {
      instanceId: box.instanceId,
      ...(box.machineId ? { machineId: box.machineId } : {}),
      geo: box.geo,
      mbps,
      at,
    };
    const next = best ? pulls.map((each) => (each === best ? kept : each)) : [...pulls, kept];
    await this.fs.mkdirp(dirname(this.file));
    await this.fs.replaceText(this.file, JSON.stringify(next, null, 2));
    return mbps;
  }
}

/** the host's own latest pull, else the median of its region's, else what the host declares */
export function downloadRate(offer: Offer, pulls: readonly MeasuredPull[]): DownloadRate {
  const own = offer.machineId
    ? pulls.filter((pull) => pull.machineId === offer.machineId).sort((a, b) => b.at - a.at)[0]
    : undefined;
  if (own) return { mbps: own.mbps, source: "host" };
  const region = pulls
    .filter((pull) => pull.geo === offer.geo)
    .map((pull) => pull.mbps)
    .sort((a, b) => a - b);
  if (region.length > 0) {
    const mid = region.length >> 1;
    const median = region.length % 2 ? region[mid]! : (region[mid - 1]! + region[mid]!) / 2;
    return { mbps: Math.round(median), source: "region" };
  }
  return { mbps: offer.downMbps, source: "declared" };
}

/** a rate as the log shows it: "447 Mb/s measured on its host", "447 Mb/s measured in Japan, JP", "7754 Mb/s declared" */
export function describeRate(rate: DownloadRate, offer: Offer): string {
  const mbps = `${rate.mbps.toFixed(0)} Mb/s`;
  if (rate.source === "host") return `${mbps} measured on its host`;
  if (rate.source === "region") return `${mbps} measured in ${offer.geo}`;
  return `${mbps} declared`;
}
