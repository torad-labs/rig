// vast.ai through its CLI (`vastai`, pip). Every call is --raw JSON; the DEPRECATED banner the CLI
// prints goes to stderr and is ignored. Offers are normalized to the Rental shape (compute_cap
// comes as cap*100: 900 -> "90", 1200 -> "120").
import { homedir } from "node:os";
import { join } from "node:path";
import type { CreateOptions, Instance, Offer, Rental, Shell, TemplateSpec } from "@rig/core";

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
  /** dph_total includes the disk's storage_total_cost at --storage GiB, 5 unless told: priced with the box's own disk,
   *  a host at 0.333 $/GB/month bills a 160 GB box 0.074 $/h more than one at 0.027 (2026-09-30) */
  async searchOffers(query: string, diskGb: number) {
    const rows = await this.json<Array<Record<string, unknown>>>(
      "search",
      "offers",
      query,
      "--storage",
      String(diskGb),
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
          downCostPerGb: Number(row.inet_down_cost ?? 0),
          storagePerHour: Number(row.storage_total_cost ?? 0),
        }),
      );
  }
  /** from a template, its image, login, on-start and ssh launch are the template's */
  async create(offerId: number, options: CreateOptions) {
    const source =
      "templateHash" in options
        ? ["--template_hash", options.templateHash]
        : [
            "--image",
            options.image,
            "--ssh",
            "--direct",
            ...(options.onstart ? ["--onstart-cmd", options.onstart] : []),
          ];
    const created = await this.json<{ success?: boolean; new_contract?: number }>(
      "create",
      "instance",
      String(offerId),
      ...source,
      "--disk",
      String(options.diskGb),
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
      ...directSsh(raw),
      ...(typeof raw.gpu_util === "number" ? { gpuUtil: raw.gpu_util } : {}),
      ...(raw.image_uuid ? { image: String(raw.image_uuid) } : {}),
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
  /** over the REST API rather than the CLI: the registry login is a body field there, where the
   *  CLI would take it as an argument any process list shows. The key is the one the CLI uses. */
  async saveTemplate(spec: TemplateSpec, hashId?: string) {
    const key = (
      await Bun.file(join(homedir(), ".config", "vastai", "vast_api_key")).text()
    ).trim();
    const body = {
      ...(hashId ? { hash_id: hashId } : {}),
      name: spec.name,
      image: spec.image,
      tag: spec.tag,
      onstart: spec.onstart,
      runtype: "ssh",
      ssh_direct: true,
      use_ssh: true,
      private: true,
      recommended_disk_space: spec.diskGb,
      docker_login_repo: spec.login.registry,
      docker_login_user: spec.login.user,
      docker_login_pass: spec.login.password,
      extra_filters: spec.filters,
      desc: spec.description,
    };
    const response = await fetch("https://console.vast.ai/api/v0/template/", {
      method: hashId ? "PUT" : "POST",
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60_000),
    });
    const reply = (await response.json().catch(() => ({}))) as {
      success?: boolean;
      msg?: string;
      template?: { id?: number; hash_id?: string };
    };
    const saved = reply.template;
    if (!response.ok || !saved?.id || !saved.hash_id)
      throw new Error(
        `vast template ${spec.name}: HTTP ${response.status} ${reply.msg ?? ""}`.trim(),
      );
    return { id: saved.id, hashId: saved.hash_id };
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

/** the host's own ssh endpoint: the instance's public address and the host port mapped to its port 22,
 *  both of which vast's listing carries beside the proxy's ssh_host/ssh_port. Absent when either is. */
function directSsh(raw: Record<string, unknown>): Pick<Instance, "directSsh"> {
  const mapped = (raw.ports as Record<string, Array<{ HostPort?: string }>> | undefined)?.[
    "22/tcp"
  ];
  const port = Number(mapped?.[0]?.HostPort);
  const host = raw.public_ipaddr ? String(raw.public_ipaddr).trim() : "";
  return host && Number.isInteger(port) && port > 0 ? { directSsh: { host, port } } : {};
}
