// The llama-server command line for a head, in one place, in one order. The order is the golden
// test's contract: golden-argv.json is the 5080 tier's argv with the MTP head, hand-maintained in
// the same commit as head.toml, so it pins order and presence, not values (head.test.ts re-derives
// the values from the constants). llama-server does not care, byte-for-byte comparison does.
// "assets/…" in a head's args resolve against the head's directory, so head.toml never carries an
// absolute path (the lens bundle's --lens-out is the one live-only exception). Free-form lists
// (runtime.extra, runtime.lens.args) come last and could override the geometry before them;
// headInvariants refuses the rendered flags inside them.

import { type CacheFormats, cacheArgv } from "../../shared/head/cache-formats.ts";
import { draftArgv } from "../../shared/head/draft-head.ts";
import type { Head } from "../../shared/head/head.ts";

export interface ServeGeometry {
  slots: number;
  ctx: number;
  cacheRam: number;
  speculative?: boolean | undefined;
  /** the tier's cache formats (tierCache) */
  cache: CacheFormats;
}

export function serverArgv(head: Head, binDir: string, g: ServeGeometry): string[] {
  const resolve = (arg: string) => (arg.startsWith("assets/") ? head.path(arg) : arg);
  return [
    `${binDir}/llama-server`,
    "-m",
    head.servedPath,
    "-ngl",
    "99",
    "--jinja",
    "-fa",
    "on",
    ...cacheArgv(head, g.cache),
    ...head.runtime.args.map(resolve),
    ...draftArgv(head, { speculative: g.speculative }),
    "-c",
    String(g.ctx),
    "-np",
    String(g.slots),
    "--kv-unified",
    "--cache-ram",
    String(g.cacheRam),
    "--no-cache-idle-slots",
    ...head.runtime.sampling,
    "--metrics",
    "--host",
    "127.0.0.1",
    "--port",
    String(head.port),
    ...head.runtime.extra,
    ...(head.runtime.lens?.enabled ? head.runtime.lens.args.map(resolve) : []),
  ];
}

export function serverEnv(binDir: string, gpu: number): Record<string, string> {
  return {
    CUDA_DEVICE_ORDER: "PCI_BUS_ID",
    CUDA_VISIBLE_DEVICES: String(gpu),
    LD_LIBRARY_PATH: binDir,
  };
}
