// What a head IS, as one JSON object, for whatever sits in front of it (a splice wizard maps
// these facts to its own provider row; the head never learns the proxy's file shape). Facts about
// the machine — supported, built, pack_verified, unit_installed, serving — are read live; facts
// about the server — port, windows, what the client must not send — are the head's data. Never
// fails: the JSON says what is missing.

import { checkArtifact } from "../../shared/artifact.ts";
import type { Engine } from "../../shared/engine/engine.ts";
import { isBuilt } from "../../shared/engine/engine.ts";
import type { Head } from "../../shared/head/head.ts";
import { HeadEndpoint } from "../../shared/head/head-endpoint.ts";
import type { Layout } from "../../shared/layout.ts";
import type { Clock, FileSystem, Gpu, Hasher, Http } from "../../shared/ports/index.ts";

/** what describe needs from unit (injected; describe never imports unit) */
export interface UnitReader {
  status(head: Head): Promise<{ installed: boolean; active: boolean }>;
}
export interface DescribeHeadDeps {
  fs: FileSystem;
  gpu: Gpu;
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
  this_gpu: number;
  this_arch: string | null;
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

  async run(head: Head, gpu: number): Promise<Description> {
    const card = await this.deps.gpu.query(gpu);
    const cap = card?.computeCap ?? null;
    const built = cap ? await isBuilt(this.deps.fs, this.engine.binDir(cap)) : false;
    const pack =
      (await checkArtifact(this.deps.fs, this.deps.hasher, {
        path: head.servedPath,
        sha256: head.served.sha256,
      })) === "ok";
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
      served_file: head.served.file,
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
      this_gpu: gpu,
      this_arch: cap ? `sm_${cap}` : null,
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
