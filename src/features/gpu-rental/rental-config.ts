import * as v from "valibot";
import type { Layout } from "../../shared/layout.ts";
import type { FileSystem } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";

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
  },
): string {
  const cardClass = config.cards[gpu] ?? config.cards.default;
  if (!cardClass) throw new Error("vast.toml: cards.default is required");
  const gpus = options.gpus ?? 1;
  const maxDph = options.maxDph ?? Math.round(cardClass.max_dph * gpus * 100) / 100;
  const parts = [
    `gpu_name=${gpu}`,
    `num_gpus=${gpus}`,
    config.rental.query,
    `disk_space>=${options.diskGb}`,
    `gpu_mem_bw>=${cardClass.min_bandwidth}`,
    `dph_total<=${maxDph}`,
  ];
  if (options.geo) parts.push(`geolocation=${options.geo}`);
  return parts.join(" ");
}
