// Rental: one seam between rig and the machine. A port names a capability, never a tool.
export interface Offer {
  id: number;
  gpu: string;
  gpus: number;
  gpuRamMiB: number;
  computeCap: string;
  dph: number;
  geo: string;
  cpu: string;
  ramGiB: number;
  bandwidth: number;
  cudaMaxGood: number;
  reliability: number;
  downMbps: number;
  /** what the host bills per GB downloaded, in dollars: 17 times apart between hosts (0.003 to 0.051, 2026-09-29) */
  downCostPerGb: number;
  /** the disk's part of dph, in dollars an hour: what the box keeps billing once stopped */
  storagePerHour: number;
}
/** a box from a stock image that rig ships itself to, or from a template whose image and on-start bring the head up */
export type CreateOptions = { diskGb: number; label: string } & (
  | { image: string }
  | { templateHash: string }
);
export interface Instance {
  id: number;
  status: string;
  label: string;
  dph: number;
  sshHost?: string;
  sshPort?: number;
  /** the card's utilization in percent as the market last sampled it; absent while it has none */
  gpuUtil?: number;
  /** the image it runs, as created: name:tag, or a digest */
  image?: string;
}
/** what a box rented from a template runs: the image, the login its registry needs, the on-start,
 *  the disk, and the offers it fits (the market's filters, e.g. { num_gpus: { eq: 2 } }) */
export interface TemplateSpec {
  name: string;
  image: string;
  tag: string;
  onstart: string;
  diskGb: number;
  login: { registry: string; user: string; password: string };
  filters: Record<string, Record<string, number | string>>;
  description: string;
}
/** A GPU rental market (vast.ai): offers, one box at a time, its ssh endpoint, and its funds. */
export interface Rental {
  funds(): Promise<number>;
  hasSshKey(pubkey: string): Promise<boolean>;
  registerSshKey(pubkey: string): Promise<void>;
  /** the offers matching `query`, each priced with a disk of `diskGb` (vast prices 5 GB unless told) */
  searchOffers(query: string, diskGb: number): Promise<Offer[]>;
  create(offerId: number, o: CreateOptions): Promise<number>;
  /** the instance, or null when the market lists no such instance; throws when the market cannot
   *  be read, which is no evidence the instance is gone */
  show(id: number): Promise<Instance | null>;
  list(): Promise<Instance[]>;
  destroy(id: number): Promise<void>;
  /** a private template, created, or edited in place when `hashId` names the one saved before */
  saveTemplate(spec: TemplateSpec, hashId?: string): Promise<{ id: number; hashId: string }>;
}
/** A remote shell over ssh with a per-box known_hosts file. */
