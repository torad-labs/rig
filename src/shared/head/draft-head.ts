// The draft head's share of a server command line, in one place for serve and the gate: nothing
// when the head declares no [speculative] or the tier opts out (a card without the room), else the
// engine's flags for the pinned draft. A sidecar draft names its file; an in-pack head (draft-mtp)
// is already in the served pack, so only the type, the draft length and, when the head sets one, the
// confidence cutoff are passed. The draft's
// context follows the target's -c (this engine has no --ctx-size-draft), which is why the tier
// check charges its compute buffer per pooled token.
import type { Head } from "./head.ts";
import { type Tier, tierSpeculates } from "./head-config.ts";

export function draftArgv(head: Head, tier: Pick<Tier, "speculative">): string[] {
  if (!tierSpeculates(head, tier)) return [];
  const s = head.speculative!;
  const depth = [
    "--spec-draft-n-max",
    String(s.n_max),
    ...(s.p_min === undefined ? [] : ["--spec-draft-p-min", String(s.p_min)]),
    ...(s.chain_p_min === undefined ? [] : ["--spec-draft-chain-p-min", String(s.chain_p_min)]),
  ];
  if ("file" in s) {
    if (!head.draftPath) return [];
    return ["--spec-type", s.type, "-md", head.draftPath, ...depth, "-ngld", "999", ...s.args];
  }
  return ["--spec-type", s.type, ...depth, ...s.args];
}
