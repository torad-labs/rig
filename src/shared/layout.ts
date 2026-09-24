// Where things live, on this machine and on a rented box alike (layoutAt(<remote_dir>) is the
// box's tree). The repo root holds the program, the engine pin and the heads; everything
// rig fetched or built on this machine — packs, engine builds and their trees, logs, gate runs,
// the rented box's state — lives under local/, gitignored and never in /tmp or a cache dir (it is
// the durable, in-project store).
import { join } from "node:path";

export interface Layout {
  root: string;
  headsDir: string;
  engineDir: string;
  /** the compiled binary (bun run build); what a unit and a box execute */
  binary: string;
  /** the rental market's request, data beside heads/ and engine/ */
  vastConfig: string;
  localDir: string;
  /** a head's packs: source, served, draft */
  packs(head: string): string;
  /** published engine builds, one per commit and card: <sha7>-sm<cap>/ */
  engineBuildsDir: string;
  /** cmake trees the builds came from; a cache */
  engineBuildTreesDir: string;
  /** the engine's source at the pin when the submodule is not (a box) */
  engineSourcesDir: string;
  /** a prebuilt engine and its CUDA runtime archives while they are fetched and unpacked */
  downloadsDir: string;
  logsDir: string;
  /** rig gate's runs: <head>/<run>/ */
  gateRunsDir: string;
  /** the rented box's state, cached build tarballs, pulled runs */
  rentedBoxDir: string;
  /** the calibration recipe's corpus and outputs (scripts/engine-corpus.ts) */
  calibrationDir: string;
  /** the pid of a server started outside systemd (a box) */
  pidFile: string;
}

export function layoutAt(root: string, localDir = join(root, "local")): Layout {
  return {
    root,
    headsDir: join(root, "heads"),
    engineDir: join(root, "engine"),
    binary: join(root, "dist", "rig"),
    vastConfig: join(root, "vast.toml"),
    localDir,
    packs: (head) => join(localDir, "packs", head),
    engineBuildsDir: join(localDir, "engine-builds"),
    engineBuildTreesDir: join(localDir, "engine-build-trees"),
    engineSourcesDir: join(localDir, "engine-sources"),
    downloadsDir: join(localDir, "downloads"),
    logsDir: join(localDir, "logs"),
    gateRunsDir: join(localDir, "gate-runs"),
    rentedBoxDir: join(localDir, "rented-box"),
    calibrationDir: join(localDir, "calibration"),
    pidFile: join(localDir, "server.pid"),
  };
}
