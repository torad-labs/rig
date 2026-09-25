// gates.toml — a head's probes as data. Every number a probe uses comes from here, so a second
// head declares its own prompts, depths and markers without a code path of its own.
import * as v from "valibot";
import type { Head } from "../../shared/head/head.ts";
import type { FileSystem } from "../../shared/ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../../shared/result.ts";

const posInt = v.pipe(v.number(), v.integer(), v.minValue(1));
const prompts = v.pipe(v.array(v.string()), v.minLength(1));
const depth = v.pipe(v.number(), v.minValue(0), v.maxValue(1));
// how far below the base pack's own top-1 the served token may sit at the first divergence, in
// nats, for the divergence to still count as a tie. Both cards declare 0.15 against measurements
// of 0.021 (capability) and 0.008-0.053 (speculative); the ceiling is not a tuning of that, it
// stops the probe going vacuous. At 1 nat the served token is under 37% as likely as the token
// the base pack would have chosen, and a gate that calls that a tie passes every pack.
const tieGap = v.pipe(v.number(), v.minValue(0), v.maxValue(1));

export const GatesSchema = v.strictObject({
  gate: v.strictObject({
    gpu: v.pipe(v.number(), v.integer(), v.minValue(0)),
    port: v.pipe(posInt, v.maxValue(65535)),
  }),
  refusal: v.strictObject({ pattern: v.string(), max_tokens: posInt, prompts }),
  capability: v.strictObject({
    max_tokens: posInt,
    prompts,
    tie_gap: tieGap,
  }),
  fluency: v.strictObject({ max_tokens: posInt, prompts }),
  humaneval: v.strictObject({
    url: v.pipe(v.string(), v.url()),
    workers: posInt,
    n_predict: posInt,
    stop: v.array(v.string()),
    alpha: v.pipe(v.number(), v.minValue(0), v.maxValue(1)),
  }),
  decode: v.strictObject({
    prompt: v.string(),
    n_predict: posInt,
    tolerance_pct: v.pipe(v.number(), v.minValue(0)),
  }),
  speculative: v.strictObject({
    max_tokens: posInt,
    prompts,
    min_gain: v.pipe(v.number(), v.minValue(0)),
    tie_gap: tieGap,
  }),
  depth: v.strictObject({
    prompt: posInt,
    gen: posInt,
    depths: v.pipe(v.array(v.pipe(v.number(), v.integer(), v.minValue(0))), v.minLength(1)),
    repeats: posInt,
  }),
  corpus: v.strictObject({ chars: posInt }), // the long text the needle and sessions probes are built from: kernel/corpus.ts, from the pinned engine tree
  needle: v.strictObject({
    tokens: posInt,
    ctx: posInt,
    needles: v.pipe(
      v.array(v.strictObject({ depth, secret: v.string(), question: v.string() })),
      v.minLength(1),
    ),
  }),
  sessions: v.strictObject({
    tokens_each: posInt,
    sessions: v.pipe(
      v.array(
        v.strictObject({ name: v.string(), marker: v.string(), depth, question: v.string() }),
      ),
      v.minLength(1),
    ),
  }),
  concurrency: v.strictObject({
    prompt: v.string(),
    max_tokens: posInt,
    batches: v.pipe(v.array(posInt), v.minLength(1)),
    min_gain: v.pipe(v.number(), v.minValue(1)),
  }),
});
export type Gates = v.InferOutput<typeof GatesSchema>;

export async function loadGates(fs: FileSystem, head: Head): Promise<Result<Gates>> {
  const file = head.path("gates.toml");
  if (!(await fs.exists(file)))
    return fail(ExitCode.Failure, `${head.name} declares no gates: ${file} is missing`);
  let data: unknown;
  try {
    data = Bun.TOML.parse(await fs.readText(file));
  } catch (e) {
    return fail(ExitCode.Failure, `gates.toml does not parse: ${(e as Error).message}`);
  }
  const parsed = v.safeParse(GatesSchema, data);
  if (!parsed.success)
    return fail(
      ExitCode.Failure,
      `gates.toml is invalid: ${parsed.issues.map((i) => `${v.getDotPath(i) ?? "(root)"}: ${i.message}`).join("; ")}`,
    );
  if (parsed.output.gate.gpu === head.gpu)
    return fail(
      ExitCode.Failure,
      `gates.toml puts the gates on GPU ${head.gpu}, the card this head serves from — gates run on another card`,
    );
  return ok(parsed.output);
}
