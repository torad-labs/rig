// vast.ai through its CLI (`vastai`, pip). Every call is --raw JSON; the DEPRECATED banner the CLI
// prints goes to stderr and is ignored. Offers are normalized to the Rental shape (compute_cap
// comes as cap*100: 900 -> "90", 1200 -> "120").
import type { Instance, Offer, Rental, Shell } from "../ports/index.ts";

export class VastAiRental implements Rental {
  constructor(private readonly shell: Shell) {}
  private async json<T>(...args: string[]): Promise<T> {
    const result = await this.shell.run(["vastai", ...args, "--raw"], { timeoutMs: 120_000 });
    if (result.code !== 0)
      throw new Error(`vastai ${args.join(" ")}: ${result.stderr.trim() || result.stdout.trim()}`);
    const start = result.stdout.search(/[[{]/);
    if (start < 0)
      throw new Error(
        `vastai ${args.join(" ")}: no JSON in ${JSON.stringify(result.stdout.slice(0, 120))}`,
      );
    return JSON.parse(result.stdout.slice(start)) as T;
  }
  async funds() {
    const account = await this.json<{ credit?: number; balance?: number }>("show", "user");
    return Math.round((Number(account.credit ?? 0) + Number(account.balance ?? 0)) * 100) / 100;
  }
  async hasSshKey(pubkey: string) {
    const keys = await this.json<Array<{ public_key?: string }>>("show", "ssh-keys");
    const body = pubkey.split(" ")[1];
    return keys.some((key) => body !== undefined && (key.public_key ?? "").includes(body));
  }
  async registerSshKey(pubkey: string) {
    const result = await this.shell.run(["vastai", "create", "ssh-key", pubkey]);
    if (result.code !== 0) throw new Error(`vastai create ssh-key: ${result.stderr.trim()}`);
  }
  async searchOffers(query: string) {
    const rows = await this.json<Array<Record<string, unknown>>>(
      "search",
      "offers",
      query,
      "-o",
      "dph_total",
    );
    return rows
      .filter((row) => row.gpu_ram)
      .map(
        (row): Offer => ({
          id: Number(row.id),
          gpu: String(row.gpu_name),
          gpus: Number(row.num_gpus ?? 1),
          gpuRamMiB: Number(row.gpu_ram),
          computeCap: String(Math.floor(Number(row.compute_cap ?? 0) / 10)),
          dph: Number(row.dph_total),
          geo: String(row.geolocation ?? ""),
          cpu: String(row.cpu_name ?? ""),
          ramGiB: Math.round(Number(row.cpu_ram ?? 0) / 1024),
          bandwidth: Number(row.gpu_mem_bw ?? 0),
          cudaMaxGood: Number(row.cuda_max_good ?? 0),
          reliability: Number(row.reliability2 ?? 0),
          downMbps: Number(row.inet_down ?? 0),
        }),
      );
  }
  async create(offerId: number, options: { image: string; diskGb: number; label: string }) {
    const created = await this.json<{ success?: boolean; new_contract?: number }>(
      "create",
      "instance",
      String(offerId),
      "--image",
      options.image,
      "--disk",
      String(options.diskGb),
      "--ssh",
      "--direct",
      "--label",
      options.label,
      "--cancel-unavail",
    );
    if (!created.success || !created.new_contract)
      throw new Error(`vastai create instance: ${JSON.stringify(created)}`);
    return created.new_contract;
  }
  private toInstance(raw: Record<string, unknown>): Instance {
    return {
      id: Number(raw.id),
      status: String(raw.actual_status ?? ""),
      label: String(raw.label ?? ""),
      dph: Number(raw.dph_total ?? 0),
      ...(raw.ssh_host ? { sshHost: String(raw.ssh_host), sshPort: Number(raw.ssh_port) } : {}),
      ...(typeof raw.gpu_util === "number" ? { gpuUtil: raw.gpu_util } : {}),
    };
  }
  /** a `show` that fails (a 429) falls back to the listing: its row, card reading included, when
   *  it lists the instance, null only when it confirms the instance is gone, and a listing that
   *  fails too (an expired key, no CLI) throws */
  async show(id: number) {
    try {
      return this.toInstance(
        await this.json<Record<string, unknown>>("show", "instance", String(id)),
      );
    } catch {
      return (await this.list()).find((instance) => instance.id === id) ?? null;
    }
  }
  async list() {
    return (await this.json<Array<Record<string, unknown>>>("show", "instances")).map((raw) =>
      this.toInstance(raw),
    );
  }
  async destroy(id: number) {
    const result = await this.shell.run(["vastai", "destroy", "instance", String(id), "-y"], {
      timeoutMs: 60_000,
    });
    if (result.code !== 0)
      throw new Error(
        `vastai destroy instance ${id}: ${result.stderr.trim() || result.stdout.trim()}`,
      );
  }
}
