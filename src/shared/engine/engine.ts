// The engine is torad-labs/llama.cpp at one commit. The pin is engine/engine.toml (data a box
// without git can carry) and the submodule at engine/llama.cpp (the source a developer edits);
// a test holds the two equal. A build is identified by that commit and the card's compute
// capability: local/engine-builds/<sha7>-sm<cap>/, published by build in one rename with a BUILD marker
// written last, so a directory either is a complete build or does not exist.
import { basename, join } from "node:path";
import * as v from "valibot";
import type { Layout } from "../layout.ts";
import type { FileSystem, Git } from "../ports/index.ts";
import { ExitCode, fail, ok, type Result } from "../result.ts";

const cap = v.pipe(
  v.string(),
  v.regex(/^\d{2,3}$/, "a compute capability with the dot removed, like 120"),
);
const sha256 = v.pipe(v.string(), v.regex(/^[0-9a-f]{64}$/));

export const EngineSchema = v.strictObject({
  fork: v.strictObject({
    repo: v.pipe(v.string(), v.url()),
    sha: v.pipe(v.string(), v.regex(/^[0-9a-f]{40}$/)),
    base: v.string(),
  }),
  archs: v.pipe(
    v.array(v.strictObject({ cap, cards: v.string(), evidence: v.string() })),
    v.minLength(1),
  ),
  // nvcc builds that compile the pin's kernels wrong for the listed cards, each with its evidence
  miscompilers: v.optional(
    v.array(
      v.strictObject({
        nvcc: v.pipe(
          v.string(),
          v.regex(/^\d+\.\d+\.\d+$/, "an nvcc version as --version prints it, like 13.2.78"),
        ),
        caps: v.pipe(v.array(cap), v.minLength(1)),
        why: v.pipe(v.string(), v.minLength(1)),
      }),
    ),
  ),
  // The CUDA runtime every build links, as NVIDIA publishes it for redistribution: a prebuilt
  // engine carries its own copy of these libraries, so a machine needs only the driver.
  cuda: v.optional(
    v.strictObject({
      version: v.pipe(v.string(), v.regex(/^\d+\.\d+$/, "a CUDA major.minor, like 13.3")),
      runtime: v.pipe(
        v.array(
          v.strictObject({
            url: v.pipe(v.string(), v.url()),
            sha256,
            libs: v.pipe(
              v.array(
                v.pipe(
                  v.string(),
                  v.regex(/^lib[\w+-]+\.so\.\d+$/, "a soname a build links, like libcudart.so.13"),
                ),
              ),
              v.minLength(1),
            ),
          }),
        ),
        v.minLength(1),
      ),
    }),
  ),
  // A portable build of the pin for one card, published with the fork's release.
  // glibc: the oldest C library it loads on, the one of the image it was built in (a machine below
  // it compiles the engine instead of installing a build that would fail its ldd -r check)
  prebuilt: v.optional(
    v.array(
      v.strictObject({
        cap,
        url: v.pipe(v.string(), v.url()),
        sha256,
        glibc: v.pipe(v.string(), v.regex(/^\d+\.\d+$/, "a glibc version like 2.35")),
      }),
    ),
  ),
});
export type EngineConfig = v.InferOutput<typeof EngineSchema>;
export type Prebuilt = NonNullable<EngineConfig["prebuilt"]>[number];
export type CudaRuntime = NonNullable<EngineConfig["cuda"]>["runtime"][number];

export interface Engine extends EngineConfig {
  sha7: string;
  submoduleDir: string;
  supports(cap: string): boolean;
  binDir(cap: string): string;
  /** the pinned prebuilt build for this card, if one is published */
  prebuiltFor(cap: string): Prebuilt | undefined;
}

/** why the card's published build does not apply on this machine (a C library older than the
 *  build's floor, or not glibc at all); undefined when it does */
export function prebuiltSkip(entry: Prebuilt, glibc: string | null): string | undefined {
  const floor = `the published sm_${entry.cap} build needs glibc ${entry.glibc} or newer`;
  if (glibc === null) return `this machine's C library is not glibc (or unreadable); ${floor}`;
  const [major = 0, minor = 0] = glibc.split(".").map(Number);
  const [floorMajor = 0, floorMinor = 0] = entry.glibc.split(".").map(Number);
  const below = major < floorMajor || (major === floorMajor && minor < floorMinor);
  return below ? `glibc ${glibc} on this machine; ${floor}` : undefined;
}

/** why `nvcc` (its version as --version prints it, 13.2.78) must not compile the engine for the
 *  card `cap`, from engine.toml's [[miscompilers]]; undefined when nothing is known against it */
export function miscompiles(
  engine: Pick<EngineConfig, "miscompilers">,
  nvcc: string | null,
  cap: string,
): string | undefined {
  const entry = engine.miscompilers?.find((m) => m.nvcc === nvcc && m.caps.includes(cap));
  return entry && `nvcc ${entry.nvcc} compiles this engine wrong for sm_${cap}: ${entry.why}`;
}

/** the name a packed build of this commit for this card carries, cached or published */
export const engineTarballName = (sha7: string, cap: string) => `engine-sm${cap}-${sha7}.tar.gz`;

export const BUILD_MARKER = "BUILD";
export const TARGETS = ["llama-server", "llama-bench", "llama-kv-mean-center"] as const;

export async function loadEngine(fs: FileSystem, layout: Layout): Promise<Result<Engine>> {
  const file = join(layout.engineDir, "engine.toml");
  if (!(await fs.exists(file))) return fail(ExitCode.Failure, `${file} is missing`);
  let data: unknown;
  try {
    data = Bun.TOML.parse(await fs.readText(file));
  } catch (error) {
    return fail(ExitCode.Failure, `engine.toml does not parse: ${(error as Error).message}`);
  }
  const parsed = v.safeParse(EngineSchema, data);
  if (!parsed.success)
    return fail(
      ExitCode.Failure,
      `engine.toml is invalid: ${parsed.issues.map((issue) => `${v.getDotPath(issue)}: ${issue.message}`).join("; ")}`,
    );
  const config = parsed.output;
  const sha7 = config.fork.sha.slice(0, 7);
  const invalid = prebuiltProblem(config, sha7);
  if (invalid) return fail(ExitCode.Failure, `engine.toml is invalid: ${invalid}`);
  return ok({
    ...config,
    sha7,
    submoduleDir: join(layout.engineDir, "llama.cpp"),
    supports: (cap) => config.archs.some((arch) => arch.cap === cap),
    binDir: (cap) => join(layout.engineBuildsDir, `${sha7}-sm${cap}`),
    prebuiltFor: (cap) => config.prebuilt?.find((entry) => entry.cap === cap),
  });
}

/** a prebuilt is the pin's own build, for a measured card, with the runtime it needs pinned: an
 *  entry left behind by a pin move names the old commit and is refused, never served */
function prebuiltProblem(config: EngineConfig, sha7: string): string | null {
  const caps = new Set<string>();
  for (const entry of config.prebuilt ?? []) {
    const expected = engineTarballName(sha7, entry.cap);
    const named = basename(new URL(entry.url).pathname);
    if (named !== expected)
      return `prebuilt sm_${entry.cap} is ${named}, not the build of the pin (${expected}): publish the pin's build and pin it, or drop the entry`;
    if (!config.archs.some((arch) => arch.cap === entry.cap))
      return `prebuilt sm_${entry.cap} is for a card [[archs]] does not list`;
    if (caps.has(entry.cap)) return `prebuilt sm_${entry.cap} is listed twice`;
    caps.add(entry.cap);
  }
  if (caps.size > 0 && !config.cuda)
    return "a prebuilt needs [cuda] (the runtime it runs on without a toolkit)";
  return null;
}

/** A build directory is complete only with its marker (written last) and its server. */
export async function isBuilt(fs: FileSystem, dir: string): Promise<boolean> {
  return (await fs.exists(join(dir, BUILD_MARKER))) && (await fs.exists(join(dir, "llama-server")));
}

/** The source tree at the pinned commit: the submodule when it is checked out there and clean,
 *  else a one-commit fetch under local/ (a box has the pin but no repo). A dirty tree at the pin,
 *  either one, is not the source: a build stamps the sha on bytes the sha does not describe. The
 *  fetched tree is refused rather than re-fetched, since its changes may be someone's work. */
export async function engineSource(
  fs: FileSystem,
  git: Git,
  layout: Layout,
  engine: Engine,
): Promise<Result<string>> {
  if (
    (await git.revParse(engine.submoduleDir, "HEAD")) === engine.fork.sha &&
    (await git.isClean(engine.submoduleDir))
  )
    return ok(engine.submoduleDir);
  const dir = join(layout.engineSourcesDir, `llama.cpp-${engine.sha7}`);
  if ((await git.revParse(dir, "HEAD")) === engine.fork.sha) {
    if (await git.isClean(dir)) return ok(dir);
    return fail(
      ExitCode.Failure,
      `${dir} is at the pin but carries changes, so it is not the pinned source: remove it and the next build fetches it again`,
    );
  }
  if (await fs.exists(dir)) await fs.remove(dir);
  await fs.mkdirp(dir);
  await git.fetchCommit(dir, engine.fork.repo, engine.fork.sha);
  if ((await git.revParse(dir, "HEAD")) !== engine.fork.sha)
    return fail(
      ExitCode.Failure,
      `fetched ${engine.fork.repo} but ${dir} is not at ${engine.fork.sha}`,
    );
  return ok(dir);
}
