// What a head IS, as one JSON object, for whatever sits in front of it (a splice wizard maps
// these facts to its own provider row; the head never learns the proxy's file shape). Facts about
// the machine — supported, built, pack_verified, unit_installed, serving — are read live; facts
// about the server — port, windows, what the client must not send — are the head's data. Never
// fails: the JSON says what is missing.

import type { Clock, Devices, FileSystem, Gpu, Hasher, Host, Http, Layout } from "@rig/core";
import { checkArtifact } from "@rig/core";
import type { Engine } from "@rig/engine";
import { isBuilt } from "@rig/engine";
import type { Head } from "@rig/head";
import { describeProfile, HeadEndpoint, placeHead } from "@rig/head";

/** what describe needs from unit (injected; describe never imports unit) */
export interface UnitReader {
  status(head: Head): Promise<{ installed: boolean; active: boolean }>;
  /** the cards the installed unit serves on, when there is one */
  devices(head: Head): Promise<Devices | undefined>;
}
export interface DescribeHeadDeps {
  fs: FileSystem;
  gpu: Gpu;
  host: Host;
  hasher: Hasher;
  http: Http;
  clock: Clock;
  unit: UnitReader;
}

export interface Description {
  name: string;
  title: string;
  dialect: "openai-chat";
  port: number;
  base_url: string;
  served_file: string;
  /** set when this machine serves the source pack in place of the derived one, and why */
  undrived: string | null;
  engine_commit: string;
  speculative: { type: string; file: string | null; n_max: number } | null;
  model_ctx: number;
  advertise_ctx: number;
  supported_archs: string[];
  /** the first of this_gpus: what a reader of one card reads */
  this_gpu: number;
  /** the cards this machine serves the head on (the installed unit's, --gpu's, or the ones its profile takes), or the
   *  cards named when none of the head's profiles fits them */
  this_gpus: number[];
  this_arch: string | null;
  /** the profile the cards serve, null when they fit none */
  profile: string | null;
  supported: boolean;
  built: boolean;
  pack_verified: boolean;
  unit_installed: boolean;
  unit_active: boolean;
  serving: boolean;
  server_facts: Head["client"];
  sampling: string[];
  repo: string;
  head_dir: string;
}

export class DescribeHead {
  constructor(
    private readonly deps: DescribeHeadDeps,
    private readonly layout: Layout,
    private readonly engine: Engine,
  ) {}

  async run(head: Head, given?: Devices): Promise<Description> {
    const devices = given ?? (await this.deps.unit.devices(head)) ?? "auto";
    const placed = await placeHead(this.deps, head, devices);
    const gpus = placed.ok
      ? placed.value.cards.map((card) => card.index)
      : devices === "auto"
        ? (await this.deps.gpu.list()).map((card) => card.index)
        : [...devices];
    const card = gpus[0] === undefined ? null : await this.deps.gpu.query(gpus[0]);
    const cap = card?.computeCap ?? null;
    const built = cap ? await isBuilt(this.deps.fs, this.engine.binDir(cap)) : false;
    let pack = true;
    for (const file of head.servedFiles)
      pack &&= (await checkArtifact(this.deps.fs, this.deps.hasher, file)) === "ok";
    const unit = await this.deps.unit.status(head);
    const endpoint = new HeadEndpoint(
      this.deps.http,
      this.deps.clock,
      `http://127.0.0.1:${head.port}`,
    );
    const serving = await endpoint.healthy();
    return {
      name: head.name,
      title: head.title,
      dialect: "openai-chat",
      port: head.port,
      base_url: `http://127.0.0.1:${head.port}/v1`,
      served_file: head.servedFiles[0]!.file,
      undrived: head.undrived ?? null,
      engine_commit: this.engine.sha7,
      speculative: head.speculative
        ? {
            type: head.speculative.type,
            file: "file" in head.speculative ? head.speculative.file : null,
            n_max: head.speculative.n_max,
          }
        : null,
      model_ctx: head.context.model,
      advertise_ctx: head.context.advertise,
      supported_archs: this.engine.archs.map((arch) => `sm_${arch.cap}`),
      this_gpu: gpus[0] ?? 0,
      this_gpus: gpus,
      this_arch: cap ? `sm_${cap}` : null,
      profile: placed.ok ? describeProfile(placed.value.profile) : null,
      supported: cap ? this.engine.supports(cap) : false,
      built,
      pack_verified: pack,
      unit_installed: unit.installed,
      unit_active: unit.active,
      serving,
      server_facts: head.client,
      sampling: head.runtime.sampling,
      repo: this.layout.root,
      head_dir: head.dir,
    };
  }
}
