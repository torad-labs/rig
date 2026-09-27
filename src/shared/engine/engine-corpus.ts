// The long text the gates use and the K-cache bias is calibrated on, built from the engine tree
// the head pins instead of shipped as a file: public, licensed, and the same bytes on every
// machine at the same pin. Files come in a fixed order — the docs, then the server, the core,
// common and ggml — sorted by path, each under a `===== <path> =====` header, prose and code in
// the mix a coding head sees, cut at a line boundary once `chars` is reached.
import { join } from "node:path";
import type { FileSystem } from "../ports/index.ts";

export const CORPUS_ROOTS = ["docs", "tools/server", "src", "common", "ggml/src"] as const;
const SKIP = new Set([
  "docs/ops",
  "tools/server/public",
  "tools/server/public_legacy",
  "tools/server/webui",
  "tools/server/tests",
]); // generated tables, bundled JS, fixtures
const TEXT = /\.(md|cpp|c|h|hpp)$/;

async function walk(fs: FileSystem, base: string, rel: string, out: string[]): Promise<void> {
  if (SKIP.has(rel)) return;
  for (const name of [...(await fs.list(join(base, rel)))].sort()) {
    const path = `${rel}/${name}`;
    const entry = await fs.stat(join(base, path));
    if (!entry || entry.isSymlink) continue;
    if (entry.isDirectory) await walk(fs, base, path, out);
    else if (TEXT.test(name)) out.push(path);
  }
}

export async function engineCorpus(
  fs: FileSystem,
  engineDir: string,
  chars: number,
): Promise<string> {
  const files: string[] = [];
  for (const root of CORPUS_ROOTS)
    if (await fs.exists(join(engineDir, root))) await walk(fs, engineDir, root, files);
  if (!files.length)
    throw new Error(`no corpus files under ${engineDir} (${CORPUS_ROOTS.join(", ")})`);
  let text = "";
  for (const rel of files) {
    const piece = `===== ${rel} =====\n${await fs.readText(join(engineDir, rel))}\n`;
    if (text.length + piece.length <= chars) {
      text += piece;
      continue;
    }
    text += piece.slice(0, piece.lastIndexOf("\n", chars - text.length - 1) + 1);
    break;
  }
  return text;
}
