import { censusProbe } from "./census-probe.ts";
import { decodeProbe } from "./decode-probe.ts";
import { depthProbe } from "./depth-probe.ts";
import { fluencyProbe } from "./fluency-probe.ts";
import { humanevalProbe } from "./human-eval-probe.ts";
import { concurrencyProbe, sessionsProbe } from "./live-probes.ts";
import { longctxProbe } from "./longctx-probe.ts";
import { needleProbe } from "./needle-probe.ts";
import type { Probe } from "./probe.ts";
import { refusalProbe } from "./refusal-probe.ts";
import { speculativeProbe } from "./speculative-probe.ts";

/** in run order: cheap and decisive first, the long ones after, the live ones last */
export const allProbes: Probe[] = [
  censusProbe,
  refusalProbe,
  fluencyProbe,
  humanevalProbe,
  decodeProbe,
  speculativeProbe,
  needleProbe,
  longctxProbe,
  depthProbe,
  sessionsProbe,
  concurrencyProbe,
];
