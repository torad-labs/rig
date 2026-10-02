// `rig vast template <head>`: the head as a vast template, so a box rented from it comes up with
// no script and no ssh session of ours. The template names the image `rig image --push` last
// published for the head (local/images/<head>-sm<cap>/image.json), the registry login its pull key
// needs, the disk the head's packs fill, offer filters from the head's first profile and the
// engine's CUDA runtime, and an on-start that runs `rig up <head> --foreground` beside `rig vast
// guard <head>`, the idle rule, each under a shell loop that starts it again: a failed fetch, a
// server that dies or an OOM at load is tried again, and five quick failures in a row give up,
// writing the file at which the guard stops the box at once rather than bill an idle hour. A run
// that lasted a quarter of an hour (it served) does not count against the five. vast's SSH launch mode replaces an image's entrypoint, so the
// on-start is what runs; the server binds loopback and ssh is the only way in. The template is
// created once and edited in place after (its hash changes with its content; its id does not),
// its ids kept beside the image record.
import { join } from "node:path";
import type { FileSystem, Http, Layout, Log, Rental, TemplateSpec } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import type { Engine } from "@rig/engine";
import type { Head } from "@rig/head";
import { draftSidecar } from "@rig/head";
import { loadRegistryConfig, publishedImage, servedDigest } from "@rig/image";

/** where the image keeps rig (image-build's IMAGE_ROOT) and where the on-start logs go */
const RIG = "/opt/rig/dist/rig";
const LOGS = "/var/log/rig";

export interface TemplateDeps {
  fs: FileSystem;
  /** the registry asked, as a box's pull will ask it, before a template names the image */
  http: Http;
  rental: Rental;
  log: Log;
  /** the key a pull presents, read when a template is saved (registryCredentials: the environment, else the keyring):
   *  never written but into the template */
  pullKey: () => Promise<string | undefined>;
}
export interface TemplateOptions {
  idleMinutes: number;
  /** the guard stops the box after this long whatever it reads */
  maxHours: number;
  diskGb?: number | undefined;
  dryRun: boolean;
}
export interface TemplateReport {
  name: string;
  image: string;
  diskGb: number;
  /** vast's ids for it, absent on a dry run */
  id?: number;
  hashId?: string;
  onstart: string;
}

export class PublishTemplate {
  constructor(
    private readonly deps: TemplateDeps,
    private readonly layout: Layout,
    private readonly engine: Engine,
  ) {}

  async run(head: Head, options: TemplateOptions): Promise<Result<TemplateReport>> {
    const { fs, log } = this.deps;
    const registry = await loadRegistryConfig(fs, this.layout.root);
    if (!registry.ok) return registry;
    if (!registry.value)
      return fail(
        ExitCode.Failure,
        "no registry.toml in this checkout: no image to point a template at",
      );
    const found = await publishedImage(fs, this.layout, head.name);
    if (!found.ok) return found;
    const { record, dir } = found.value;
    const pullKey = options.dryRun ? undefined : await this.deps.pullKey();
    if (!options.dryRun && !pullKey)
      return fail(
        ExitCode.Failure,
        "the template logs its boxes into the registry: the keyring's rig-registry pull, or RIG_REGISTRY_PULL_KEY in the environment",
      );
    const profile = head.profiles[0]!;
    const onstart = renderOnstart(head.name, options.idleMinutes, options.maxHours);
    const diskGb = options.diskGb ?? headDiskGb(head);
    const name = `rig-${head.name}-sm${record.cap}`;
    const [image, tag] = splitRef(record.image);
    const spec: TemplateSpec = {
      name,
      image,
      tag,
      onstart,
      diskGb,
      login: { registry: registry.value.host, user: "rig", password: pullKey ?? "" },
      filters: {
        num_gpus: { eq: profile.devices },
        gpu_ram: { gte: profile.min_vram_mib },
        compute_cap: { eq: Number(record.cap) * 10 },
        ...(this.engine.cuda ? { cuda_max_good: { gte: Number(this.engine.cuda.version) } } : {}),
        disk_space: { gte: diskGb },
      },
      description: `${head.title}: rig up --foreground on ${profile.devices} × sm_${record.cap} (≥ ${profile.min_vram_mib} MiB each), restarted on failure, stopped after ${options.idleMinutes} min idle or ${options.maxHours} h; reach it with ssh -L ${head.port}:127.0.0.1:${head.port}`,
    };
    const report: TemplateReport = { name, image: record.image, diskGb, onstart };
    if (options.dryRun) return ok(report);

    // a template whose login the registry refuses, or whose tag it serves as another image, fails every box rented from
    // it at its pull, after the box bills: asked here, as the box will ask
    const served = await servedDigest(this.deps.http, registry.value, pullKey ?? "", tag);
    if (!served.ok) return served;
    const pushed = record.digest?.split("@")[1];
    if (pushed && served.value !== pushed)
      return fail(
        ExitCode.Failure,
        `${registry.value.host} serves ${tag} as ${served.value}, not the ${pushed} rig image pushed: push it again`,
      );

    const recordFile = join(dir, "template.json");
    const previous = (await fs.exists(recordFile))
      ? (JSON.parse(await fs.readText(recordFile)) as { hashId?: string })
      : {};
    const saved = await this.deps.rental.saveTemplate(spec, previous.hashId);
    await fs.writeText(
      recordFile,
      `${JSON.stringify({ ...saved, name, image: record.image }, null, 2)}\n`,
    );
    log.info(
      `template ${name} ${previous.hashId ? "updated" : "created"} (id ${saved.id}, hash ${saved.hashId}): ${record.image}, ${diskGb} GB`,
    );
    return ok({ ...report, ...saved });
  }
}

/** what the box runs at every start, a resume included: the guard, then the head in the foreground, each started again
 *  when it exits; the head's supervisor gives up after five tries that each ended within 15 minutes */
export function renderOnstart(head: string, idleMinutes: number, maxHours: number): string {
  const failed = `${LOGS}/FAILED`;
  const guard = `until ${RIG} vast guard ${head} --idle-minutes ${idleMinutes} --max-hours ${maxHours} --stop-when ${failed}; do sleep 30; done`;
  const up = [
    "n=0; while :; do t=$(date +%s)",
    `${RIG} up ${head} --foreground`,
    "[ $(($(date +%s) - t)) -ge 900 ] && n=0; n=$((n + 1)); [ $n -ge 5 ] && break",
    `echo "rig: rig up exited, try $((n + 1)) of 5 in $((30 * n)) s"; sleep $((30 * n)); done; touch ${failed}`,
  ].join("; ");
  return [
    `# rig: ${head}, supervised, and the idle rule (rig vast template)`,
    `mkdir -p ${LOGS} && rm -f ${failed}`,
    `nohup sh -c '${guard}' >> ${LOGS}/guard.log 2>&1 &`,
    `nohup sh -c '${up}' >> ${LOGS}/up.log 2>&1 &`,
  ].join("\n");
}

/** the bytes the head puts on a box's disk: every pinned file it fetches or derives */
export function headPackBytes(head: Head): number {
  const sidecar = head.speculative ? draftSidecar(head.speculative) : undefined;
  return (
    head.sourceFiles.reduce((sum, file) => sum + file.bytes, 0) +
    (head.derive ? head.servedFiles.reduce((sum, file) => sum + file.bytes, 0) : 0) +
    (sidecar?.bytes ?? 0)
  );
}

/** the disk the head fills on a box, in GB: its pack, a tenth more, and 10 GB for the image and the logs, rounded up
 *  to 10 */
export function headDiskGb(head: Head): number {
  return Math.ceil(((headPackBytes(head) * 1.1) / 1e9 + 10) / 10) * 10;
}

/** "host/repo:tag" as ["host/repo", "tag"] */
function splitRef(ref: string): [string, string] {
  const at = ref.lastIndexOf(":");
  return [ref.slice(0, at), ref.slice(at + 1)];
}
