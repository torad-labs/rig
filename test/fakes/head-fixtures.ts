// Head fixtures derived from the real head.toml, so a test of a path the real head no longer
// takes (a sidecar draft file beside the pack) still runs against the real schema.
import { type Derive, deriveAsset } from "../../src/shared/head/head-config.ts";

/** the DFlash2 sidecar block the head carried before the in-pack MTP head (pins are real, the
 *  footprint is a fixture's: small enough that every real tier still fits beside it) */
export const SIDECAR_DRAFT = `[speculative]
repo = "ProCreations/Ternary-Bonsai-2-27B-DFlash2"
rev = "4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b"
file = "Bonsai-2-27B-DFlash2-Q8_0.gguf"
sha256 = "9dd11c8adb910058faf9fb77b10d90c1c048a4f3c2887a890f592cbd882deb9a"
bytes = 1
type = "draft-dflash"
n_max = 3
weights_mib = 100
overhead_mib = 50
bytes_per_token = 64
cache = { k = "f16", v = "f16", kv_elements_per_token = 1024 }
`;

/** A Torad machine's view of a head: head.toml plus every [derive] asset it pins — the private ones
 *  (never in git) in the head's directory, the public ones (a url) where `rig fetch` puts them.
 *  Without the private ones (a stranger's clone) the head serves its [public] or source pack. */
export function putHead(
  fs: { put(path: string, text: string): void },
  root: string,
  headToml: string,
  name = "bonsai-2-27b",
): void {
  fs.put(`${root}/heads/${name}/head.toml`, headToml);
  const steps = (Bun.TOML.parse(headToml) as { derive?: Derive[] }).derive ?? [];
  for (const step of steps) {
    const asset = deriveAsset(step);
    const dir = asset.url ? `${root}/local/packs/${name}` : `${root}/heads/${name}`;
    fs.put(`${dir}/${asset.path}`, "asset");
  }
}

/** the real head.toml with its [speculative] block replaced by the sidecar draft */
export function withSidecarDraft(headToml: string): string {
  return headToml.replace(/\[speculative\][\s\S]*?\n\n/, `${SIDECAR_DRAFT}\n`);
}
