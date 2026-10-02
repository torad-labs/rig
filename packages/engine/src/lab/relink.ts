// The tree's libggml-cuda with chosen sources as HEAD has them and every other file as the working tree has it: in a
// checkout several seats share, a peer's uncommitted edit (a header half rewritten) lands in any build of it, and this
// is the library of one's own change alone. The objects that read a chosen path (ninja's recorded dependencies, or the
// source itself) compile with the build's own commands from a mirror of their sources, the chosen paths exported from
// HEAD into it; every other object is ninja's, from the working tree; the link is the build's own, to `out`. Each
// mirrored object's dependency file is read back: one that read a chosen path's working copy fails by name.
import { dirname, join, relative, resolve, sep } from "node:path";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import type { LabPorts } from "./engine-lab.service.ts";

export interface RelinkRequest {
  /** the cmake tree */
  tree: string;
  /** paths under the engine's source tree (ggml/src/ggml-cuda/fattn-common.cuh) taken as HEAD has them */
  head: string[];
  /** where libggml-cuda.so.0 and the mirrored objects go */
  out: string;
  /** compiles at once */
  jobs: number;
}

export interface RelinkReport {
  lib: string;
  sha256: string;
  /** the objects compiled from the mirror, relative to the tree */
  mirrored: string[];
}

export async function relinkWithHead(
  ports: LabPorts,
  req: RelinkRequest,
): Promise<Result<RelinkReport>> {
  const { shell, fs, git, hasher, log } = ports;
  const ninja = (args: string[]) => shell.run(["ninja", "-C", req.tree, ...args]);

  const cache = await fs.readText(join(req.tree, "CMakeCache.txt")).catch(() => "");
  const source = cache.match(/^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$/m)?.[1]?.trim();
  if (!source)
    return fail(
      ExitCode.Failure,
      `${req.tree} is not a configured cmake tree (no CMAKE_HOME_DIRECTORY)`,
    );
  for (const path of req.head)
    if (path.startsWith("/") || path.split("/").includes(".."))
      return fail(
        ExitCode.Usage,
        `${path}: a path under the engine's source tree, like ggml/src/ggml-cuda/fattn.cu`,
      );

  const targets = await ninja(["-t", "targets", "all"]);
  const target = targets.stdout
    .split("\n")
    .map((line) => line.split(":")[0] ?? "")
    .find((name) => /(^|\/)libggml-cuda\.so\.\d+\.\d+\.\d+$/.test(name));
  if (!target) return fail(ExitCode.Failure, `${req.tree} builds no libggml-cuda.so.<version>`);
  const commands = (await ninja(["-t", "commands", target])).stdout.trim().split("\n");
  const link = commands.at(-1) ?? "";
  if (!link.includes(`-o ${target}`))
    return fail(ExitCode.Failure, `ninja's last command for ${target} is not its link`);
  const objects = link.split(/\s+/).filter((token) => token.endsWith(".o"));
  const compile = new Map<string, string>();
  for (const command of commands)
    for (const object of objects)
      if (command.endsWith(` -o ${object}`)) compile.set(object, command);

  const deps = readNinjaDeps((await ninja(["-t", "deps", ...objects])).stdout);
  const chosen = new Set(req.head.map((path) => join(source, path)));
  const sourceOf = (object: string) => compile.get(object)?.match(/ -c (\S+) -o /)?.[1] ?? "";
  const unrecorded = objects.filter((object) => !deps.has(object));
  if (unrecorded.length > 0)
    return fail(
      ExitCode.Failure,
      `ninja has no dependencies recorded for ${unrecorded.length} objects (${unrecorded[0]}, ...): build the tree once`,
    );
  const mirrored = objects.filter(
    (object) =>
      chosen.has(sourceOf(object)) ||
      (deps.get(object) ?? []).some((dep) => chosen.has(resolve(req.tree, dep))),
  );
  if (mirrored.length === 0)
    return fail(ExitCode.Failure, `no object of ${target} reads ${req.head.join(", ")}`);
  for (const object of mirrored)
    if (!compile.has(object)) return fail(ExitCode.Failure, `no compile command for ${object}`);

  // the rest as the working tree has them
  const rest = objects.filter((object) => !mirrored.includes(object));
  log.info(`${rest.length} objects from the working tree, ${mirrored.length} from the mirror`);
  const built = await ninja(rest);
  if (built.code !== 0)
    return fail(
      ExitCode.Failure,
      `ninja (the working tree's objects): ${tail(built.stdout + built.stderr)}`,
    );

  // the mirror: the smallest directory holding every mirrored source, chosen path and include directory a chosen path
  // is found through, copied whole (its own headers found beside the sources that include them), the chosen paths
  // exported from HEAD over it
  const root = commonDir(
    mirrorDirs(
      source,
      req.tree,
      req.head,
      mirrored.map((object) => compile.get(object) as string),
    ),
  );
  const mirror = join(req.out, "mirror");
  await fs.remove(mirror);
  await fs.mkdirp(dirname(join(mirror, root)));
  await fs.copyTree(join(source, root), join(mirror, root));
  await git.exportTree(source, "HEAD", req.head, mirror);
  const inMirror = (path: string) =>
    path === join(source, root) || path.startsWith(join(source, root) + sep)
      ? join(mirror, relative(source, path))
      : path;

  const objDir = join(req.out, "obj");
  const failures: string[] = [];
  let next = 0;
  const worker = async () => {
    while (next < mirrored.length) {
      const object = mirrored[next++] as string;
      const out = join(objDir, object);
      await fs.mkdirp(dirname(out));
      const src = sourceOf(object);
      const command = (compile.get(object) as string)
        .replace(` -c ${src} -o ${object}`, ` -c ${inMirror(src)} -o ${out}`)
        .replace(`-MF ${object}.d`, `-MF ${out}.d`)
        .replace(/-I(\S+)/g, (_, dir: string) => `-I${inMirror(resolve(req.tree, dir))}`);
      const result = await shell.run(["sh", "-c", command], { cwd: req.tree });
      if (result.code !== 0) {
        failures.push(`${object}: ${tail(result.stdout + result.stderr)}`);
        continue;
      }
      const read = readDepFile(await fs.readText(`${out}.d`).catch(() => ""));
      const leaked = read.map((dep) => resolve(req.tree, dep)).filter((dep) => chosen.has(dep));
      if (leaked.length > 0)
        failures.push(`${object} read the working copy of ${leaked.join(", ")}`);
      else log.info(`compiled ${object}`);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, req.jobs) }, worker));
  if (failures.length > 0) return fail(ExitCode.Failure, failures.join("\n"));

  const lib = join(req.out, "libggml-cuda.so.0");
  let relinked = link
    .replace(`-o ${target}`, `-o ${lib}`)
    .replace(/--dependency-file=\S+/, `--dependency-file=${join(req.out, "link.d")}`);
  for (const object of mirrored)
    relinked = relinked.replace(
      new RegExp(`(?<=\\s)${escapeRegExp(object)}(?=\\s|$)`),
      join(objDir, object),
    );
  const linked = await shell.run(["sh", "-c", relinked], { cwd: req.tree });
  if (linked.code !== 0)
    return fail(ExitCode.Failure, `link: ${tail(linked.stdout + linked.stderr)}`);
  await fs.remove(mirror);
  return ok({ lib, sha256: await hasher.sha256File(lib), mirrored });
}

/** `ninja -t deps`: each object and the files its last compile read, as recorded */
export function readNinjaDeps(text: string): Map<string, string[]> {
  const deps = new Map<string, string[]>();
  let current: string[] | null = null;
  for (const line of text.split("\n")) {
    const head = line.match(/^(\S.*?): #deps \d+/);
    if (head) {
      current = [];
      deps.set(head[1] as string, current);
    } else if (/^\s+\S/.test(line) && current) current.push(line.trim());
    else current = null;
  }
  return deps;
}

/** a make-syntax dependency file's prerequisites (every target's), continuations and escaped spaces read */
export function readDepFile(text: string): string[] {
  return text
    .replace(/\\\n/g, " ")
    .split("\n")
    .flatMap((rule) => {
      const colon = rule.search(/:(\s|$)/);
      return colon < 0 ? [] : (rule.slice(colon + 1).match(/(?:\\ |\S)+/g) ?? []);
    })
    .map((dep) => dep.replace(/\\ /g, " "));
}

/** the source directories (relative to `source`) the mirror holds for `compiles`, the mirrored objects' commands: the
 *  chosen paths', the compiled sources', and each include directory under `source` that a chosen path is in, since a
 *  compile finds the path through it (ggml-cuda.cu's "ggml-cuda/mmq.cuh" through -I ggml/src) and outside the mirror
 *  that is the working copy */
export function mirrorDirs(
  source: string,
  tree: string,
  head: readonly string[],
  compiles: readonly string[],
): string[] {
  const chosen = head.map((path) => join(source, path));
  const sources = compiles.map((command) => command.match(/ -c (\S+) -o /)?.[1] ?? "");
  const includes = new Set(
    compiles.flatMap((command) =>
      [...command.matchAll(/-I(\S+)/g)].map((match) => resolve(tree, match[1] as string)),
    ),
  );
  const reaching = [...includes].filter(
    (dir) => dir.startsWith(source + sep) && chosen.some((path) => path.startsWith(dir + sep)),
  );
  return [
    ...head.map((path) => dirname(path)),
    ...sources.map((src) => dirname(relative(source, src))),
    ...reaching.map((dir) => relative(source, dir)),
  ];
}

/** the deepest directory every one of `dirs` is in (relative paths; "." for none in common) */
export function commonDir(dirs: readonly string[]): string {
  const split = dirs.map((dir) => dir.split("/").filter((part) => part && part !== "."));
  const first = split[0] ?? [];
  let n = 0;
  while (n < first.length && split.every((parts) => parts[n] === first[n])) n++;
  return first.slice(0, n).join("/") || ".";
}

const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const tail = (text: string) => text.trim().split("\n").slice(-4).join(" | ");
