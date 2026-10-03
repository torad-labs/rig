import type { FileSystem, Layout } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import * as v from "valibot";

const card = v.strictObject({
  min_bandwidth: v.pipe(v.number(), v.minValue(0)),
  max_dph: v.pipe(v.number(), v.minValue(0)),
});
export const VastSchema = v.strictObject({
  rental: v.strictObject({
    image: v.string(),
    disk_gb: v.pipe(v.number(), v.integer(), v.minValue(10)),
    label: v.string(),
    local_port: v.pipe(v.number(), v.integer(), v.minValue(1024), v.maxValue(65535)),
    idle_minutes: v.pipe(v.number(), v.integer(), v.minValue(1)),
    stopped_hours: v.pipe(v.number(), v.integer(), v.minValue(1)),
    remote_dir: v.pipe(v.string(), v.startsWith("/")),
    query: v.string(),
  }),
  cards: v.pipe(
    v.record(v.string(), card),
    v.check((cards) => "default" in cards, "cards.default is required"),
  ),
});
export type VastConfig = v.InferOutput<typeof VastSchema>;

export async function loadVastConfig(fs: FileSystem, layout: Layout): Promise<Result<VastConfig>> {
  const file = layout.vastConfig;
  if (!(await fs.exists(file))) return fail(ExitCode.Failure, `${file} is missing`);
  let data: unknown;
  try {
    data = Bun.TOML.parse(await fs.readText(file));
  } catch (error) {
    return fail(ExitCode.Failure, `vast.toml does not parse: ${(error as Error).message}`);
  }
  const parsed = v.safeParse(VastSchema, data);
  if (!parsed.success)
    return fail(
      ExitCode.Failure,
      `vast.toml is invalid: ${parsed.issues.map((issue) => `${v.getDotPath(issue) ?? "(root)"}: ${issue.message}`).join("; ")}`,
    );
  return ok(parsed.output);
}

/** the market query for a card class: the shared filters plus the class's floor and ceiling. The
 * class ceiling is per card and scales with the count; an explicit maxDph is the box price as given. */
export function offerQuery(
  config: VastConfig,
  gpu: string,
  options: {
    maxDph?: number | undefined;
    geo?: string | undefined;
    diskGb: number;
    gpus?: number | undefined;
    /** only hosts that rent a full virtual machine: a vast CONTAINER cannot run docker (no NET_ADMIN for its
     *  iptables chain, no mount privilege for its overlayfs), which the driver-only gate and `rig image` both need */
    vm?: boolean | undefined;
    /** only hosts that download at this speed or faster, in Mb/s; never lower than the shared query's own floor */
    minDownMbps?: number | undefined;
  },
): string {
  const cardClass = config.cards[gpu] ?? config.cards.default;
  if (!cardClass) throw new Error("vast.toml: cards.default is required");
  const gpus = options.gpus ?? 1;
  const maxDph = options.maxDph ?? Math.round(cardClass.max_dph * gpus * 100) / 100;
  const parts = [
    `gpu_name=${gpu}`,
    `num_gpus=${gpus}`,
    withDownFloor(config.rental.query, options.minDownMbps),
    `disk_space>=${options.diskGb}`,
    `gpu_mem_bw>=${cardClass.min_bandwidth}`,
    `dph_total<=${maxDph}`,
  ];
  if (options.geo) parts.push(`geolocation=${options.geo}`);
  if (options.vm) parts.push("vms_enabled=true");
  return parts.join(" ");
}

/** the shared query with its inet_down term raised to `floor`; a floor under the term's own leaves it, and a query with
 *  no such term gets one. The term is replaced rather than repeated so the query never carries two answers for one field. */
function withDownFloor(query: string, floor: number | undefined): string {
  if (floor === undefined) return query;
  const term = /inet_down>=(\d+(?:\.\d+)?)/.exec(query);
  if (!term) return `${query} inet_down>=${floor}`;
  return Number(term[1]) >= floor ? query : query.replace(term[0], `inet_down>=${floor}`);
}
