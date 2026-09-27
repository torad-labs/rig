import type { Hasher } from "../ports/index.ts";

export class BunHasher implements Hasher {
  async sha256File(path: string): Promise<string> {
    const hasher = new Bun.CryptoHasher("sha256");
    for await (const chunk of Bun.file(path).stream()) hasher.update(chunk);
    return hasher.digest("hex");
  }
}
