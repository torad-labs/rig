// Each file's sha256 on a worker thread of its own, so the shards of a pack hash on as many cores as there are shards
// (3 × 2 GiB: 3.9 s on one thread, 1.75 s on three workers, in the compiled binary), and a file this process has hashed
// is not read again while it is the same file: the same device, inode, size and modification time, read before and
// after the hash. `rig up --foreground` hashes each shard as fetch publishes it, and serve's verify asked for the same
// 134.3 GB again seconds later; now its answer is the first one. A rename keeps all four, a write moves the time or the
// size, and a process starts knowing nothing, so a box started again reads its pack once.
import { stat } from "node:fs/promises";
import type { Hasher } from "@rig/core";

const WORKER = `self.onmessage = async (event) => {
  try {
    const hasher = new Bun.CryptoHasher("sha256");
    for await (const chunk of Bun.file(event.data).stream()) hasher.update(chunk);
    postMessage({ hash: hasher.digest("hex") });
  } catch (error) {
    postMessage({ code: error?.code, message: String(error?.message ?? error) });
  }
};`;
let workerUrl: string | undefined;

export class BunHasher implements Hasher {
  private readonly known = new Map<string, string>();
  /** the files read through to answer, a remembered answer not counted */
  reads = 0;

  async sha256File(path: string): Promise<string> {
    const before = await identity(path);
    const known = this.known.get(before);
    if (known) return known;
    this.reads++;
    const hash = await hashInWorker(path);
    if ((await identity(path)) === before) this.known.set(before, hash);
    return hash;
  }
}

async function identity(path: string): Promise<string> {
  const s = await stat(path, { bigint: true });
  return `${s.dev}:${s.ino}:${s.size}:${s.mtimeNs}`;
}

/** an errno the worker met reading the file is thrown with its code, as a read on this thread would throw it */
function hashInWorker(path: string): Promise<string> {
  workerUrl ??= URL.createObjectURL(new Blob([WORKER], { type: "application/javascript" }));
  const worker = new Worker(workerUrl);
  return new Promise<string>((resolve, reject) => {
    worker.onmessage = (
      event: MessageEvent<{ hash?: string; code?: string; message?: string }>,
    ) => {
      const { hash, code, message } = event.data;
      if (hash) resolve(hash);
      else reject(Object.assign(new Error(`${message} (${path})`), { code }));
    };
    worker.onerror = (event) =>
      reject(new Error(`the hashing worker failed on ${path}: ${event.message}`));
    worker.postMessage(path);
  }).finally(() => worker.terminate());
}
