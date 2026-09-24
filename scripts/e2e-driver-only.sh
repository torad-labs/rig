#!/usr/bin/env bash
# The gate a prebuilt engine passes before it is published: a machine with only the NVIDIA driver.
# A fresh ubuntu:24.04 container (the oldest base a prebuilt supports: glibc 2.38, libstdc++ 14,
# OpenSSL 3.0) gets one card through CDI (the driver's libcuda and nvidia-smi, nothing else from
# NVIDIA) and only curl. This checkout is packed as release.yml packs it, installed by install.sh,
# and then `rig prepare` must report a prebuilt with no toolkit, `rig build` must install the
# prebuilt and NVIDIA's pinned runtime and pass its ldd -r check, and the installed llama-bench
# must decode <pack.gguf> on the card.
#
#   scripts/e2e-driver-only.sh <pack.gguf> [--prebuilt engine-sm<cap>-<sha7>.tar.gz] [--gpu N]
#
# --prebuilt installs a local tarball instead of the one engine.toml pins (the staged engine.toml
# points its [[prebuilt]] entry at the file, with the file's sha256): the check a build passes
# BEFORE it is uploaded and pinned. Without it, the pinned URL is fetched as a user would.
# RIG_GATE_LOCK=<file> holds that flock around the decode leg only (a shared gate card).
set -euo pipefail

usage() { echo "usage: scripts/e2e-driver-only.sh <pack.gguf> [--prebuilt <tarball>] [--gpu N]" >&2; exit 64; }
[ $# -ge 1 ] || usage
pack=$(realpath "$1")
shift
prebuilt="" gpu=0
while [ $# -gt 0 ]; do
  case $1 in
    --prebuilt) prebuilt=$(realpath "${2:?}"); shift 2 ;;
    --gpu) gpu=${2:?}; shift 2 ;;
    *) usage ;;
  esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
stage=$(mktemp -d "$root/local/e2e-driver-only-XXXXXX")
trap 'rm -rf "$stage"' EXIT

# the release, as release.yml packs it
(cd "$root" && bun run build > /dev/null)
mkdir -p "$stage/pack/rig/engine" "$stage/release"
cp -r "$root/dist" "$root/heads" "$root/LICENSE" "$root/README.md" "$stage/pack/rig/"
cp "$root/engine/engine.toml" "$stage/pack/rig/engine/"
mounts=()
if [ -n "$prebuilt" ]; then
  name=$(basename "$prebuilt")
  sha=$(sha256sum "$prebuilt" | cut -c1-64)
  grep -q "/$name\"$" "$stage/pack/rig/engine/engine.toml" ||
    { echo "engine.toml pins no prebuilt named $name (the pin's own build is engine-sm<cap>-<sha7>.tar.gz)" >&2; exit 1; }
  # the entry that names this file points at the mounted copy, with its sha256
  awk -v name="$name" -v sha="$sha" '
    $0 ~ "^url = .*/" name "\"$" { print "url = \"file:///prebuilt/" name "\""; hit = 1; next }
    hit && /^sha256 = / { print "sha256 = \"" sha "\""; hit = 0; next }
    { print }' "$stage/pack/rig/engine/engine.toml" > "$stage/engine.toml"
  mv "$stage/engine.toml" "$stage/pack/rig/engine/engine.toml"
  mounts+=(-v "$prebuilt:/prebuilt/$name:ro")
fi
tar -C "$stage/pack" --owner=0 --group=0 --numeric-owner -czf "$stage/release/rig-linux-x64.tar.gz" rig
(cd "$stage/release" && sha256sum rig-linux-x64.tar.gz > rig-linux-x64.tar.gz.sha256)
cp "$root/install.sh" "$stage/release/"

lock=/dev/null
if [ -n "${RIG_GATE_LOCK:-}" ]; then lock=$RIG_GATE_LOCK; fi
docker run --rm --device "nvidia.com/gpu=$gpu" -v "$lock:/gate.lock" \
  -v "$stage/release:/rel:ro" -v "$pack:/pack.gguf:ro" "${mounts[@]}" ubuntu:24.04 bash -c '
set -euo pipefail
apt-get update -qq > /dev/null
apt-get install -y -qq curl ca-certificates > /dev/null   # what install.sh needs; rig adds the rest
echo "== the machine: no toolkit, no compiler"
for tool in nvcc cmake ninja git gcc; do
  if command -v $tool > /dev/null; then echo "UNEXPECTED: $tool is here"; exit 1; fi
done
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo "== install.sh"
RIG_RELEASE_URL=file:///rel sh /rel/install.sh
export PATH=$HOME/.local/bin:$PATH
echo "== rig prepare"
rig prepare --json | tee /tmp/prepare.json
grep -q "\"prebuilt\": true" /tmp/prepare.json || { echo "prepare: no prebuilt for this card"; exit 1; }
grep -q "\"toolkitCuda\": null" /tmp/prepare.json || { echo "prepare: a toolkit is on this machine"; exit 1; }
echo "== rig build"
rig build
dir=$(ls -d $HOME/.local/share/rig/local/engine-builds/*-sm*)
cat "$dir/BUILD"
echo "== the runtime resolves from the build directory"
ldd "$dir/libggml-cuda.so" | grep -E "cudart|cublas|gomp|libcuda\.so"
echo "== decode"
flock /gate.lock "$dir/llama-bench" -m /pack.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 2
echo "== PASS"
'
