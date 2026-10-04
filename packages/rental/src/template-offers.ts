// The offers a head's template can be rented on, and what each would cost all in. The template's own filters (the head's
// first profile's cards and memory, the image's sm, the engine's CUDA runtime, the pack's disk) plus a host that keeps
// its boxes (reliability) and downloads fast enough for the pack. vast applies a template's filters only to a search in
// its console; a rent by offer id ignores them, so rig asks the market with them itself.
// The all-in cost is pick-offer.sh's (box3, 2026-09-29), moved here: the hours at the offer's price with its disk, the
// boot included (the image, the pack's download at the rate rig measured on the host or its region, else the one it
// declares (download-rates.ts), its hash and load), and the pack's gigabytes
// at the host's download price, which is 17 times apart between hosts (0.003 to 0.051 $/GB): box 53422109 billed 135
// GB at 0.051, $6.86, before it served anything. The cheapest by the hour is not the cheapest box.
import type { Offer } from "@rig/core";
import type { Head } from "@rig/head";
import type { DownloadRate } from "./download-rates.ts";

/** a host whose boxes stay up; the rentals so far all came from 0.97 and over */
export const MIN_RELIABILITY = 0.97;
/** 134.3 GB at 800 Mbit/s is 22 minutes; below it the download is the session */
export const MIN_DOWN_MBPS = 800;
/** decode reads every weight each token, so a card's memory bandwidth is its speed: vast.toml's floor for the PRO 6000
 *  classes, all of which measure about 1,400 GB/s. A "PRO 6000 WS" listed at 636 GB/s on 2026-09-30 serves at half */
export const MIN_GPU_MEM_BW = 1300;
/** the image's pull and the box's start (97 s and 108 s on the template's boxes) */
const START_SECONDS = 180;
/** the pack hashed on a core a shard (about 3 GB/s together) and mapped onto the cards */
const HASH_BYTES_PER_SECOND = 3e9;
const LOAD_SECONDS = 60;
/** a host whose RAM cannot hold the pack beside the box reads it from disk again to load it, after the hash */
const DISK_BYTES_PER_SECOND = 2e9;

export interface OfferEstimate {
  offer: Offer;
  /** from the rent to the first token, in minutes */
  minutesToServe: number;
  /** the session of `hours` after it, the boot and the pack's download, in dollars */
  dollars: number;
  /** the download rate the boot was priced at: the host's measured pull where rig has one, else its declaration */
  rate: DownloadRate;
}

/** what the host declares, the rate an offer is ranked on when rig has measured nothing there */
export const declaredRate = (offer: Offer): DownloadRate => ({
  mbps: offer.downMbps,
  source: "declared",
});

/** the market query for the head's template: its filters as vast's search spells them */
export function templateQuery(
  head: Head,
  options: { cap: string; cuda?: string | undefined; diskGb: number; maxDph?: number | undefined },
): string {
  const profile = head.profiles[0]!;
  const parts = [
    `num_gpus=${profile.devices}`,
    // the search reads GB of card memory, the profile MiB
    `gpu_ram>=${Math.floor(profile.min_vram_mib / 1024)}`,
    `compute_cap=${Number(options.cap) * 10}`,
    ...(options.cuda ? [`cuda_max_good>=${options.cuda}`] : []),
    `disk_space>=${options.diskGb}`,
    `gpu_mem_bw>=${MIN_GPU_MEM_BW}`,
    `reliability>${MIN_RELIABILITY}`,
    `inet_down>=${MIN_DOWN_MBPS}`,
    "rentable=true",
    ...(options.maxDph !== undefined ? [`dph_total<=${options.maxDph}`] : []),
  ];
  return parts.join(" ");
}

/** what one offer costs to serve the pack for `hours`, and how long it takes to first serve, the pack pulled at `rate` */
export function estimateOffer(
  offer: Offer,
  packBytes: number,
  hours: number,
  rate: DownloadRate = declaredRate(offer),
): OfferEstimate {
  const downloadSeconds = (packBytes * 8) / (Math.max(rate.mbps, 1) * 1e6);
  const reread = offer.ramGiB * 2 ** 30 < packBytes * 1.1 ? packBytes / DISK_BYTES_PER_SECOND : 0;
  const bootSeconds =
    START_SECONDS + downloadSeconds + packBytes / HASH_BYTES_PER_SECOND + LOAD_SECONDS + reread;
  const billedHours = hours + bootSeconds / 3600;
  // dph is priced with the box's disk already (Rental.searchOffers)
  const dollars = offer.dph * billedHours + (offer.downCostPerGb * packBytes) / 1e9;
  return { offer, minutesToServe: bootSeconds / 60, dollars, rate };
}

/** every offer, the cheapest all in first; a tie goes to the one that serves sooner */
export function rankOffers(
  offers: Offer[],
  packBytes: number,
  hours: number,
  rateOf: (offer: Offer) => DownloadRate = declaredRate,
): OfferEstimate[] {
  return offers
    .map((offer) => estimateOffer(offer, packBytes, hours, rateOf(offer)))
    .sort((a, b) => a.dollars - b.dollars || a.minutesToServe - b.minutesToServe);
}
