// The llama-server command line for a head, in one place, in one order. The order is the golden
// test's contract: golden-argv.json is the 5080 profile's argv with the MTP head, hand-maintained in
// the same commit as head.toml, so it pins order and presence, not values (head.test.ts re-derives
// the values from the constants). llama-server does not care, byte-for-byte comparison does.
// "assets/…" in a head's args resolve against the head's directory, so head.toml never carries an
// absolute path (the lens bundle's --lens-out is the one live-only exception). Free-form lists
// (runtime.extra, runtime.lens.args) come last and could override the geometry before them;
// headInvariants refuses the rendered flags inside them.

import type { Head } from "@rig/head";
import {
  type CacheFormats,
  cacheArgv,
  draftArgv,
  type Profile,
  profileEngineEnv,
  type SplitMode,
} from "@rig/head";

export interface ServeGeometry {
  slots: number;
  ctx: number;
  cacheRam: number;
  speculative?: boolean | undefined;
  /** the profile's cache formats (profileCache) */
  cache: CacheFormats;
  /** how the model spans the cards, over several: evenly, the cards of one profile being alike */
  split?: { mode: SplitMode; cards: number } | undefined;
}

export function serverArgv(head: Head, binDir: string, g: ServeGeometry): string[] {
  const resolve = (arg: string) => (arg.startsWith("assets/") ? head.path(arg) : arg);
  return [
    `${binDir}/llama-server`,
    "-m",
    head.servedPath,
    "-ngl",
    "99",
    ...(g.split ? ["-sm", g.split.mode, "-ts", Array(g.split.cards).fill("1").join(",")] : []),
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

/** the cards by nvidia-smi's numbering, which CUDA follows under PCI_BUS_ID, the profile's engine switches (its CUDA
 *  graph cap, profileEngineEnv), and under a tensor split the engine's own all-reduce, the one the whole-evaluation CUDA
 *  graph is captured over and every tensor figure of the GLM head was measured with (the Linux default asks for NCCL
 *  first, which a portable build does not carry) */
export function serverEnv(
  binDir: string,
  gpus: readonly number[],
  profile: Pick<Profile, "cuda_graphs" | "l2_issue" | "split">,
): Record<string, string> {
  return {
    CUDA_DEVICE_ORDER: "PCI_BUS_ID",
    CUDA_VISIBLE_DEVICES: gpus.join(","),
    ...profileEngineEnv(profile),
    ...(profile.split === "tensor" ? { GGML_CUDA_ALLREDUCE: "internal" } : {}),
    LD_LIBRARY_PATH: binDir,
  };
}
