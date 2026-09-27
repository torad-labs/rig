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
}
export interface Instance {
  id: number;
  status: string;
  label: string;
  dph: number;
  sshHost?: string;
  sshPort?: number;
  /** the card's utilization in percent as the market last sampled it; absent while it has none */
  gpuUtil?: number;
}
/** A GPU rental market (vast.ai): offers, one box at a time, its ssh endpoint, and its funds. */
export interface Rental {
  funds(): Promise<number>;
  hasSshKey(pubkey: string): Promise<boolean>;
  registerSshKey(pubkey: string): Promise<void>;
  searchOffers(query: string): Promise<Offer[]>;
  create(offerId: number, o: { image: string; diskGb: number; label: string }): Promise<number>;
  /** the instance, or null when the market lists no such instance; throws when the market cannot
   *  be read, which is no evidence the instance is gone */
  show(id: number): Promise<Instance | null>;
  list(): Promise<Instance[]>;
  destroy(id: number): Promise<void>;
}
/** A remote shell over ssh with a per-box known_hosts file. */
