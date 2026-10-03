// `rig image <head>`: the head as a container image a rented box runs with no script of its own —
// the rig CLI, the head, the engine pin and the engine for one sm baked in, the pack fetched at the
// first start by `rig up --foreground`. No source goes in: the CLI is compiled here from the
// committed tree and only the binary is copied. What goes in is what the public repo publishes: the
// head and the pin through public-export.toml's rules and gitleaks, the CLI's and the engine's bytes
// through the same deny patterns, the head's private [derive] assets never. The image is proven on a card here before
// it can be pushed: prepare finds the engine installed, build finds it built for the card, and the
// engine decodes on it. The tag names the head, the sm, the rig commit and the context's hash, so a
// changed input is a new tag and no host serves a stale layer under an old name. --push writes it
// into registry.toml's bucket (oci-push.ts), where the registry serves it behind the pull key.
//
// Three machines each hold one of the three things an image needs, so the work also runs in three
// parts, each refusing to take the next on trust: --build where the public rules live (this repo, no
// card), --prove where the card is (any checkout, no rules, no credential), --push-proven where the
// registry credential is. What passes between them is the image as a tarball and its receipt, image.json.
import { basename, dirname, join, relative } from "node:path";
import type {
  Clock,
  Containers,
  FileSystem,
  Git,
  Gpu,
  Hasher,
  Http,
  Layout,
  Log,
  ObjectStores,
  Shell,
} from "@rig/core";
import { ExitCode, fail, fetchPinned, ok, type Result } from "@rig/core";
import { type Engine, engineTarballName } from "@rig/engine";
import { type Head, privateAssets } from "@rig/head";
import { IMAGE_ROOT, renderDockerfile } from "./dockerfile.ts";
import { pushImage } from "./oci-push.ts";
import {
  excludedBy,
  loadPublicRules,
  type PublicRules,
  publicFile,
  rawHits,
} from "./public-rules.ts";
import { imageDir } from "./published.ts";
import {
  loadRegistryConfig,
  type RegistryConfig,
  type RegistryCredentials,
} from "./registry-config.ts";
import { readLayout, saveImage } from "./saved-image.ts";

export const IMAGE_SOURCE = "https://github.com/torad-labs/rig";

/** what the CLI is compiled from, repo-relative, as committed: the workspace root and every
 *  workspace its lockfile names (a frozen install refuses a lockfile whose workspaces are missing) */
const CLI_SOURCE = [
  "package.json",
  "bun.lock",
  "bunfig.toml",
  "apps",
  "packages",
  "tools",
] as const;
/** the CLI's entry, under the exported workspace root */
const CLI_ENTRY = "apps/cli/src/main.ts";

/** a 135M model any card decodes in a second: the smoke that the engine in the image runs */
export const SMOKE_MODEL = {
  url: "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf",
  sha256: "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d",
  file: "SmolLM2-135M-Instruct-Q4_K_M.gguf",
} as const;

export interface BuildImageDeps {
  clock: Clock;
  shell: Shell;
  fs: FileSystem;
  git: Git;
  gpu: Gpu;
  hasher: Hasher;
  containers: Containers;
  objectStores: ObjectStores;
  http: Http;
  log: Log;
}

/** `rig image <head>`: built, proven on a card of this machine, and with --push published, in one run. The
 *  push's credentials come from the environment (main.ts): never from a file. */
export interface BuildImageOptions {
  /** the card the image is proven on, by nvidia-smi index; its sm is the image's */
  gpu: number;
  /** the engine tarball to bake, instead of the one found under local/ */
  fromTarball?: string | undefined;
  push: boolean;
}

/** `rig image <head> --cap N --build FILE`: the image for an sm, with no card on this machine */
export interface BuildOptions {
  /** the sm the image is for, as `rig build --cap` names it */
  cap: string;
  /** the engine tarball to bake, instead of the one found under local/ */
  fromTarball?: string | undefined;
  /** where the image is kept, as `docker save` writes it; the ID in the receipt is read out of that
   *  very file, so the tarball carried to another machine and the receipt cannot disagree */
  out: string;
}

/** `rig image <head> --prove RECEIPT`: the card proof of an image a `docker load` put on this machine */
export interface ProveOptions {
  /** the image.json a --build run wrote, carried here beside the tarball; the proof is written back to it */
  receipt: string;
  /** the card the image is proven on, by nvidia-smi index */
  gpu: number;
}

/** `rig image <head> --push-proven`: a push of an image proven on another machine's card */
export interface PushProvenOptions {
  /** the image.json a --prove run wrote, carried back beside the tarball */
  receipt: string;
}

/** which card ran the image and when: written by the run that proved it, required by the one that pushes */
export interface Proof {
  card: string;
  /** the card's compute capability: the image's sm */
  sm: string;
  /** ISO time of the proof */
  at: string;
}

export interface ImageReport {
  image: string;
  /** host/repository@sha256:… once pushed */
  digest: string | null;
  head: string;
  cap: string;
  commit: string;
  engine: string;
  pushed: boolean;
  /** the config digest of the image — docker's image ID, preserved by save and load. Written by --build;
   *  what --prove and --push-proven require the loaded image to be. */
  id?: string;
  /** set by the run that proved the image on a card; --push-proven refuses a receipt without it */
  proven?: Proof;
}

export class BuildImage {
  constructor(
    private readonly deps: BuildImageDeps,
    private readonly layout: Layout,
    private readonly engine: Engine,
    /** what a machine whose engine is installed needs from apt: prepare's own list */
    private readonly packages: readonly string[],
    /** read when a push needs them, not before: the keyring is asked only then */
    private readonly credentials: () => Promise<RegistryCredentials>,
  ) {}

  async run(head: Head, options: BuildImageOptions): Promise<Result<ImageReport>> {
    const { deps, engine } = this;
    const registry = await loadRegistryConfig(deps.fs, this.layout.root);
    if (!registry.ok) return registry;
    const { accessKeyId, secretAccessKey, pullKey } = options.push ? await this.credentials() : {};
    if (options.push && !registry.value)
      return fail(
        ExitCode.Failure,
        "no registry.toml in this checkout: an image has nowhere to be pushed",
      );
    if (options.push && !(accessKeyId && secretAccessKey && pullKey))
      return fail(
        ExitCode.Failure,
        "--push writes the registry's bucket and reads the image back: the keyring's rig-registry entries (r2-access-key-id, r2-secret-access-key, pull), or RIG_R2_ACCESS_KEY_ID, RIG_R2_SECRET_ACCESS_KEY and RIG_REGISTRY_PULL_KEY in the environment",
      );
    const card = await deps.gpu.query(options.gpu);
    if (!card)
      return fail(
        ExitCode.Failure,
        `no CUDA card at nvidia-smi index ${options.gpu}: an image is proven on a card of its sm before it is pushed`,
      );
    const cap = card.computeCap;
    if (!engine.supports(cap))
      return fail(
        ExitCode.Unsupported,
        `sm_${cap} (${card.name}) is not a card the engine's kernels are measured on`,
      );
    const assembled = await this.assemble(head, cap, options.fromTarball, registry.value);
    if (!assembled.ok) return assembled;
    const { image, tag, name, dir, commit } = assembled.value;

    const proven = await this.proveOnCard(image, head, cap, options.gpu);
    if (!proven.ok) return proven;

    const report: ImageReport = {
      image,
      digest: null,
      head: head.name,
      cap,
      commit,
      engine: engine.fork.sha,
      pushed: false,
      proven: this.proof(card),
    };
    if (options.push && registry.value && accessKeyId && secretAccessKey && pullKey) {
      const { endpoint, bucket } = registry.value;
      const store = deps.objectStores.open({ endpoint, bucket, accessKeyId, secretAccessKey });
      deps.log.info(`pushing ${image} into ${bucket}`);
      const pushed = await pushImage(
        deps,
        store,
        registry.value,
        pullKey,
        image,
        tag,
        join(dir, "push"),
      );
      if (!pushed.ok) return pushed;
      report.pushed = true;
      report.digest = `${name}@${pushed.value}`;
      deps.log.info(`pushed ${report.digest}, served back by ${registry.value.host}`);
    }
    await deps.fs.writeText(join(dir, "image.json"), `${JSON.stringify(report, null, 2)}\n`);
    return ok(report);
  }

  /** The build half, for the machine that has the public rules and no card: the image for an sm, kept
   *  as a tarball with the ID read out of that file and the private commit it was built from. The
   *  receipt it writes carries no proof; `prove` adds that on a machine with a card of this sm. */
  async build(head: Head, options: BuildOptions): Promise<Result<ImageReport>> {
    const { deps, engine } = this;
    const { cap } = options;
    if (!engine.supports(cap))
      return fail(
        ExitCode.Unsupported,
        `sm_${cap} is not a card the engine's kernels are measured on, so there is no image to build for it`,
      );
    const registry = await loadRegistryConfig(deps.fs, this.layout.root);
    if (!registry.ok) return registry;
    const assembled = await this.assemble(head, cap, options.fromTarball, registry.value);
    if (!assembled.ok) return assembled;
    const { image, dir, commit } = assembled.value;
    const kept = await this.keep(image, options.out, join(dir, "save"));
    if (!kept.ok) return kept;
    const report: ImageReport = {
      image,
      digest: null,
      head: head.name,
      cap,
      commit,
      engine: engine.fork.sha,
      pushed: false,
      id: kept.value,
    };
    await deps.fs.writeText(join(dir, "image.json"), `${JSON.stringify(report, null, 2)}\n`);
    deps.log.info(
      `${options.out} holds ${image}, image ID ${kept.value}: carry it and ${join(dir, "image.json")} to a machine with an sm_${cap} card and run --prove there`,
    );
    return ok(report);
  }

  /** The proof half, for the machine with the card: any checkout of the head, no export rules and no
   *  credential. The image must be the one the receipt names, a `docker load` of the tarball --build
   *  kept, and the card must be of its sm; then it runs as a rented box will run it, and the card and
   *  the time are written into the receipt. The cheap refusals come before the image is read. */
  async prove(head: Head, options: ProveOptions): Promise<Result<ImageReport>> {
    const { deps, engine } = this;
    const { receipt: path } = options;
    const read = await this.readReceipt(path, head);
    if (!read.ok) return read;
    const { image, id, cap } = read.value;
    if (!id)
      return fail(
        ExitCode.Failure,
        `${path} records no image ID, so nothing here can be shown to be the image it names: run rig image --cap N --build FILE where the public rules are`,
      );
    if (read.value.engine !== engine.fork.sha)
      return fail(
        ExitCode.Failure,
        `${image} bakes torad-labs/llama.cpp @ ${read.value.engine?.slice(0, 7)} and this checkout pins ${engine.sha7}: prove it from a checkout of the same pin`,
      );
    const card = await deps.gpu.query(options.gpu);
    if (!card)
      return fail(
        ExitCode.Failure,
        `no CUDA card at nvidia-smi index ${options.gpu}: an image is proven on a card of its sm`,
      );
    if (card.computeCap !== cap)
      return fail(
        ExitCode.Failure,
        `${image} is the sm_${cap} image and ${card.name} is sm_${card.computeCap}: a proof on another sm says nothing about it`,
      );

    const dir = imageDir(this.layout, head.name, cap);
    const loaded = await this.keep(image, join(dir, "loaded.tar"), join(dir, "proven"));
    await deps.fs.remove(join(dir, "loaded.tar"));
    if (!loaded.ok) return loaded;
    if (loaded.value !== id)
      return fail(
        ExitCode.Failure,
        `${image} here is ${loaded.value}, which is not the image the receipt names: ${path} says ${id}. Load the tarball --build kept.`,
      );

    const proven = await this.proveOnCard(image, head, cap, options.gpu);
    if (!proven.ok) return proven;
    const report: ImageReport = { ...read.value, proven: this.proof(card) };
    await deps.fs.writeText(path, `${JSON.stringify(report, null, 2)}\n`);
    deps.log.info(`${path} records ${image} proven on ${card.name} (sm_${cap})`);
    return ok(report);
  }

  private proof(card: { name: string; computeCap: string }): Proof {
    return {
      card: card.name,
      sm: card.computeCap,
      at: new Date(this.deps.clock.now()).toISOString(),
    };
  }

  /** the receipt at `path`, parsed, and known to be this head's and to name an image and an sm */
  private async readReceipt(path: string, head: Head): Promise<Result<ImageReport>> {
    const { fs } = this.deps;
    if (!(await fs.exists(path)))
      return fail(
        ExitCode.Failure,
        `${path} does not exist: it is the image.json a --build run wrote, carried here beside the tarball`,
      );
    let receipt: Partial<ImageReport>;
    try {
      receipt = JSON.parse(await fs.readText(path)) as Partial<ImageReport>;
    } catch (error) {
      return fail(ExitCode.Failure, `${path} does not parse: ${(error as Error).message}`);
    }
    if (!receipt.image || !receipt.cap)
      return fail(ExitCode.Failure, `${path} names no image and sm`);
    if (receipt.head !== head.name)
      return fail(ExitCode.Failure, `${path} is ${receipt.head}'s image, not ${head.name}'s`);
    return ok(receipt as ImageReport);
  }

  /** the image for `cap` built under the public rules from this checkout's HEAD: the context staged and
   *  scanned, the CLI compiled, the engine baked, `docker build`. Nothing is run and nothing is pushed. */
  private async assemble(
    head: Head,
    cap: string,
    fromTarball: string | undefined,
    registry: RegistryConfig | null,
  ): Promise<Result<{ image: string; tag: string; name: string; dir: string; commit: string }>> {
    const { deps, engine } = this;
    const commit = await deps.git.revParse(this.layout.root, "HEAD");
    if (!commit) return fail(ExitCode.Failure, `${this.layout.root} has no commit to build from`);
    const tarball = await this.engineTarball(cap, fromTarball);
    if (!tarball.ok) return tarball;

    const rules = await loadPublicRules(deps.fs, this.layout.root);
    if (!rules.ok) return rules;
    const dir = imageDir(this.layout, head.name, cap);
    const context = join(dir, "context");
    // context/rig/ is the image's /opt/rig: the head and the pin, then the CLI in dist/
    const tree = join(context, "rig");
    await deps.fs.remove(context);
    await deps.fs.mkdirp(tree);
    const staged = await this.stageHead(head, rules.value, tree, dir);
    if (!staged.ok) return staged;
    const cli = await this.compileCli(dir, join(tree, "dist", "rig"));
    if (!cli.ok) return cli;
    const cliHits = rawHits(
      rules.value,
      "dist/rig",
      await deps.fs.readBytes(join(tree, "dist", "rig")),
    );
    if (cliHits.length > 0) return fail(ExitCode.Failure, refusal(cliHits));
    if (tarball.value) {
      const hits = rawHits(
        rules.value,
        basename(tarball.value),
        Bun.gunzipSync(new Uint8Array(await deps.fs.readBytes(tarball.value))),
      );
      if (hits.length > 0) return fail(ExitCode.Failure, refusal(hits));
      await deps.fs.copy(tarball.value, join(context, "engine", basename(tarball.value)));
    }
    const dockerfile = renderDockerfile({
      head: head.name,
      cap,
      commit: commit.slice(0, 7),
      ...(tarball.value ? { engineTarball: basename(tarball.value) } : {}),
      packages: this.packages,
    });
    await deps.fs.writeText(join(context, "Dockerfile"), dockerfile);

    const hash = await this.contextHash(context);
    const tag = `${head.name}-sm${cap}-${commit.slice(0, 7)}-${hash.slice(0, 8)}`;
    const name = registry ? `${registry.host}/${registry.repository}` : "rig";
    const image = `${name}:${tag}`;
    const log = join(this.layout.logsDir, `image-${head.name}-sm${cap}.log`);
    await deps.fs.mkdirp(this.layout.logsDir);
    deps.log.info(`building ${image} (rig ${commit.slice(0, 7)}, engine ${engine.sha7}) → ${log}`);
    const built = await deps.containers.build(
      context,
      image,
      {
        "org.opencontainers.image.source": IMAGE_SOURCE,
        "org.opencontainers.image.revision": commit,
        "org.opencontainers.image.description": `${head.name} on sm_${cap}: rig and torad-labs/llama.cpp @ ${engine.sha7}; the pack is fetched at the first start`,
      },
      log,
    );
    if (built.code !== 0)
      return fail(ExitCode.Failure, `the image did not build: ${lastLines(built.stderr)} (${log})`);
    return ok({ image, tag, name, dir, commit });
  }

  /** The push half of a proof made on another machine's card. The card rule sends `rig image` to a
   *  rented box and a registry credential never goes to one, so the halves run apart: there --prove
   *  checks the loaded image against the receipt and runs it, here `docker load` puts it back and this
   *  pushes it. The prove-before-push gate holds across the gap because what goes out is required to
   *  BE the image a card proved — same ID, same head, same rig commit, an sm a card of that sm ran.
   *  No card is read and nothing is built: this machine has neither to offer. */
  async pushProven(head: Head, options: PushProvenOptions): Promise<Result<ImageReport>> {
    const { deps, engine } = this;
    const registry = await loadRegistryConfig(deps.fs, this.layout.root);
    if (!registry.ok) return registry;
    if (!registry.value)
      return fail(
        ExitCode.Failure,
        "no registry.toml in this checkout: an image has nowhere to be pushed",
      );
    const read = await this.readReceipt(options.receipt, head);
    if (!read.ok) return read;
    const receipt = read.value;
    const { image, id, cap, proven } = receipt;
    if (!id)
      return fail(
        ExitCode.Failure,
        `${options.receipt} records no image ID, so nothing here can be shown to be the image that was proven: re-run rig image --cap N --build FILE where the public rules are, then --prove on the card`,
      );
    if (!proven)
      return fail(
        ExitCode.Failure,
        `${options.receipt} records no proof: no card has proved ${image}. Carry it and its tarball to a machine with an sm_${cap} card and run --prove there.`,
      );
    if (proven.sm !== cap)
      return fail(
        ExitCode.Failure,
        `${options.receipt} records ${image} (sm_${cap}) proven on ${proven.card}, which is sm_${proven.sm}: a proof on another sm says nothing about it`,
      );
    const commit = await deps.git.revParse(this.layout.root, "HEAD");
    if (!commit) return fail(ExitCode.Failure, `${this.layout.root} has no commit to push from`);
    if (receipt.commit !== commit)
      return fail(
        ExitCode.Failure,
        `${image} was built from rig ${receipt.commit?.slice(0, 7)} and this checkout is ${commit.slice(0, 7)}: push from the commit the image was built from`,
      );
    if (!engine.supports(cap))
      return fail(
        ExitCode.Unsupported,
        `sm_${cap} is not a card the engine's kernels are measured on, so its image is not one to publish`,
      );

    const { accessKeyId, secretAccessKey, pullKey } = await this.credentials();
    if (!(accessKeyId && secretAccessKey && pullKey))
      return fail(
        ExitCode.Failure,
        "a push writes the registry's bucket and reads the image back: the keyring's rig-registry entries (r2-access-key-id, r2-secret-access-key, pull), or RIG_R2_ACCESS_KEY_ID, RIG_R2_SECRET_ACCESS_KEY and RIG_REGISTRY_PULL_KEY in the environment",
      );

    const dir = imageDir(this.layout, head.name, cap);
    const here = await this.keep(image, join(dir, "loaded.tar"), join(dir, "proven"));
    await deps.fs.remove(join(dir, "loaded.tar"));
    if (!here.ok) return here;
    if (here.value !== id)
      return fail(
        ExitCode.Failure,
        `${image} here is ${here.value}, which is not the image proven on a card: ${options.receipt} names ${id}. Load the tarball --build kept.`,
      );

    const tag = image.slice(image.lastIndexOf(":") + 1);
    const { endpoint, bucket } = registry.value;
    const store = deps.objectStores.open({ endpoint, bucket, accessKeyId, secretAccessKey });
    deps.log.info(`pushing ${image}, proven on sm_${cap} as ${id}, into ${bucket}`);
    const pushed = await pushImage(
      deps,
      store,
      registry.value,
      pullKey,
      image,
      tag,
      join(dir, "push"),
    );
    if (!pushed.ok) return pushed;
    const name = `${registry.value.host}/${registry.value.repository}`;
    const report: ImageReport = {
      image,
      digest: `${name}@${pushed.value}`,
      head: head.name,
      cap,
      commit,
      engine: receipt.engine ?? engine.fork.sha,
      pushed: true,
      id,
      proven,
    };
    await deps.fs.writeText(options.receipt, `${JSON.stringify(report, null, 2)}\n`);
    deps.log.info(`pushed ${report.digest}, served back by ${registry.value.host}`);
    return ok(report);
  }

  /** `image` saved at `tarball`, and the ID read out of that file: the config digest, which covers the
   *  layer diff_ids, the entrypoint and the environment, and which save and load both preserve */
  private async keep(image: string, tarball: string, work: string): Promise<Result<string>> {
    const { fs } = this.deps;
    await fs.mkdirp(dirname(tarball));
    const saved = await saveImage(this.deps, image, tarball);
    if (!saved.ok) return saved;
    await fs.remove(work);
    try {
      const read = await readLayout(this.deps, tarball, work);
      return read.ok ? ok(read.value.manifest.config.digest) : read;
    } finally {
      await fs.remove(work);
    }
  }

  /** the tarball to bake: the one named, else a build of this pin for this sm under local/ (the
   *  release floor's first), else none when the pin publishes a prebuilt the image fetches itself */
  private async engineTarball(
    cap: string,
    named: string | undefined,
  ): Promise<Result<string | undefined>> {
    const name = engineTarballName(this.engine.sha7, cap);
    if (named) {
      if (basename(named) !== name)
        return fail(
          ExitCode.Failure,
          `${basename(named)} is not the build of torad-labs/llama.cpp @ ${this.engine.sha7} for sm_${cap} (expected ${name})`,
        );
      if (!(await this.deps.fs.exists(named)))
        return fail(ExitCode.Failure, `${named} does not exist`);
      return ok(named);
    }
    const { releaseBuildsDir, engineBuildsDir, cachedBuildsDir } = this.layout;
    for (const dir of [releaseBuildsDir, engineBuildsDir, cachedBuildsDir]) {
      if (await this.deps.fs.exists(join(dir, name))) return ok(join(dir, name));
    }
    if (this.engine.prebuiltFor(cap)) return ok(undefined);
    return fail(
      ExitCode.Failure,
      `no build of torad-labs/llama.cpp @ ${this.engine.sha7} for sm_${cap} to bake, and the pin publishes none: rig build --prebuilt --sha ${this.engine.fork.sha} makes ${name} on the release floor, or --from-tarball names one`,
    );
  }

  /** the CLI compiled from the committed tree (not this checkout's edits) into `out`: the tree
   *  exported beside the context, its lockfile's dependencies, one `bun build --compile` */
  private async compileCli(dir: string, out: string): Promise<Result<void>> {
    const { fs, shell } = this.deps;
    if (!(await shell.which("bun")))
      return fail(
        ExitCode.Failure,
        "bun is not on this machine: it compiles the CLI the image runs",
      );
    const source = join(dir, "cli-source");
    await fs.remove(source);
    await fs.mkdirp(source);
    await this.deps.git.exportTree(this.layout.root, "HEAD", CLI_SOURCE, source);
    for (const step of [
      ["bun", "install", "--frozen-lockfile", "--production"],
      ["bun", "build", "--compile", "--target=bun-linux-x64", "--outfile", out, CLI_ENTRY],
    ]) {
      const run = await shell.run(step, { cwd: source, timeoutMs: 600_000 });
      if (run.code !== 0)
        return fail(ExitCode.Failure, `${step.slice(0, 2).join(" ")}: ${lastLines(run.stderr)}`);
    }
    await fs.remove(source);
    return ok(undefined);
  }

  /** the head and the engine pin as committed, under `into`, each file as the public repo would
   *  publish it; refused by name when a rule or gitleaks does not clear */
  private async stageHead(
    head: Head,
    rules: PublicRules,
    into: string,
    dir: string,
  ): Promise<Result<void>> {
    const { fs } = this.deps;
    const headDir = `heads/${head.name}`;
    await this.deps.git.exportTree(this.layout.root, "HEAD", ["engine/engine.toml", headDir], into);
    const held = new Set(privateAssets(head).map((path) => `${headDir}/${path}`));
    const hits: string[] = [];
    for (const path of await walk(fs, into)) {
      const file = join(into, path);
      if (held.has(path) || excludedBy(rules, path) !== undefined) {
        await fs.remove(file);
        continue;
      }
      if ((await fs.stat(file))?.isSymlink) {
        hits.push(`${path}: a symlink ships as its target path, which the deny scan never reads`);
        continue;
      }
      const published = publicFile(rules, path, await fs.readBytes(file));
      hits.push(...published.hits);
      await fs.writeBytes(file, published.bytes);
    }
    hits.push(...(await this.gitleaks(into, join(dir, "gitleaks.json"))));
    return hits.length > 0 ? fail(ExitCode.Failure, refusal(hits)) : ok(undefined);
  }

  /** gitleaks over the staged source with this repo's config; its own report tells a finding from
   *  a scanner that could not run, both of which refuse */
  private async gitleaks(dir: string, report: string): Promise<string[]> {
    if (!(await this.deps.shell.which("gitleaks")))
      return ["gitleaks is not on this machine: an image is published only after it scans clean"];
    await this.deps.fs.remove(report); // a report an earlier scan left is never read as this one's
    const scan = await this.deps.shell.run(
      [
        "gitleaks",
        "detect",
        "--no-git",
        "--source",
        dir,
        "--config",
        join(this.layout.root, ".gitleaks.toml"),
        "--report-format",
        "json",
        "--report-path",
        report,
        "--redact",
        "--no-banner",
      ],
      { timeoutMs: 600_000 },
    );
    if (scan.code === 0) return [];
    const found = await this.deps.fs
      .readText(report)
      .then((text) => JSON.parse(text) as { File: string; StartLine: number; RuleID: string }[])
      .catch(() => null);
    if (Array.isArray(found) && found.length > 0)
      return found.map(
        (leak) => `${relative(dir, leak.File)}:${leak.StartLine}: gitleaks ${leak.RuleID}`,
      );
    return [`gitleaks could not run: ${lastLines(scan.stderr)}`];
  }

  /** every file of the context by path and sha256: the tag's hash */
  private async contextHash(context: string): Promise<string> {
    const lines: string[] = [];
    for (const path of await walk(this.deps.fs, context))
      lines.push(`${path} ${await this.deps.hasher.sha256File(join(context, path))}`);
    return new Bun.CryptoHasher("sha256").update(lines.join("\n")).digest("hex");
  }

  /** the image on the card, as a rented box will run it: prepare needs nothing it lacks, the
   *  engine is built for the card's sm, and it decodes there */
  private async proveOnCard(
    image: string,
    head: Head,
    cap: string,
    gpu: number,
  ): Promise<Result<void>> {
    const { containers, log } = this.deps;
    const on = { gpu };
    const prepare = await containers.run(image, ["rig", "prepare", head.name, "--json"], on);
    const prepared = parseJson<{ built?: boolean }>(prepare.stdout);
    if (prepare.code !== 0 || prepared?.built !== true)
      return fail(
        ExitCode.Failure,
        `${image}: prepare on the card does not find the engine installed: ${lastLines(prepare.stderr)}`,
      );
    const build = await containers.run(image, ["rig", "build", head.name, "--json"], on);
    if (
      build.code !== 0 ||
      parseJson<{ alreadyBuilt?: boolean }>(build.stdout)?.alreadyBuilt !== true
    )
      return fail(
        ExitCode.Failure,
        `${image}: build on the card is not the baked sm_${cap} engine: ${lastLines(build.stderr)}`,
      );
    const smoke = {
      path: join(this.layout.downloadsDir, SMOKE_MODEL.file),
      sha256: SMOKE_MODEL.sha256,
    };
    if (!(await this.deps.fs.exists(smoke.path))) {
      const fetched = await fetchPinned(this.deps, SMOKE_MODEL.url, smoke, "the smoke model");
      if (!fetched.ok) return fetched;
    }
    const bench = join(
      IMAGE_ROOT,
      "local",
      "engine-builds",
      basename(this.engine.binDir(cap)),
      "llama-bench",
    );
    // CUDA that fails to initialize leaves no device and llama-bench decodes on the CPU with exit 0:
    // the card must be listed first
    const devices = await containers.run(image, [bench, "--list-devices"], on);
    if (devices.code !== 0 || !/^\s*CUDA0: /m.test(devices.stdout))
      return fail(
        ExitCode.Failure,
        `${image}: the engine sees no CUDA device on the card: ${lastLines(`${devices.stdout}\n${devices.stderr}`)}`,
      );
    const decode = await containers.run(
      image,
      [bench, "-m", "/smoke.gguf", "-ngl", "99", "-p", "64", "-n", "16", "-r", "1"],
      { ...on, mounts: { [smoke.path]: "/smoke.gguf" } },
    );
    if (decode.code !== 0)
      return fail(
        ExitCode.Failure,
        `${image}: the engine did not decode on the card: ${lastLines(decode.stderr)}`,
      );
    log.info(
      `proven on the card: prepare needs nothing, the sm_${cap} engine is built, and it decodes`,
    );
    return ok(undefined);
  }
}

/** every file under `dir`, relative to it, sorted */
async function walk(fs: FileSystem, dir: string, prefix = ""): Promise<string[]> {
  const out: string[] = [];
  for (const name of await fs.list(join(dir, prefix))) {
    const path = prefix ? `${prefix}/${name}` : name;
    const stat = await fs.stat(join(dir, path));
    if (stat?.isDirectory && !stat.isSymlink) out.push(...(await walk(fs, dir, path)));
    else out.push(path);
  }
  return out.sort();
}

function parseJson<T>(text: string): T | null {
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

const lastLines = (text: string) => text.trim().split("\n").slice(-3).join(" | ");
const refusal = (hits: string[]) =>
  `REFUSED: ${hits.length} check(s) did not clear, so nothing was built:\n  ${hits.join("\n  ")}`;
