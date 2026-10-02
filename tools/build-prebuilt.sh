#!/usr/bin/env bash
# Builds the pin's prebuilt engine in tools/prebuilt/Dockerfile's image, so its floor is that
# image's (Ubuntu 22.04, glibc 2.35) and not whatever the building machine runs: `rig build
# --portable` inside the container, as the calling user, with its own local/ (local/prebuilt/, so no
# build tree mixes with the host's), capped at 14 GiB and 6 CPUs (the engine's largest CUDA
# objects need ~13 GiB to compile). The card is passed through only for `rig build` to read its
# compute capability. The tarball it prints is what tools/e2e-driver-only.sh --prebuilt gates.
# --sha builds a fork commit other than the pin (a rental image ahead of it) through a copy of
# engine.toml naming that commit and no published prebuilt; the pin itself is unchanged.
#
#   tools/build-prebuilt.sh [--gpu N] [--jobs J] [--sha FORK_SHA]
set -euo pipefail

gpu=0 jobs=6 sha=
while [ $# -gt 0 ]; do
  case $1 in
    --gpu) gpu=${2:?}; shift 2 ;;
    --jobs) jobs=${2:?}; shift 2 ;;
    --sha) sha=${2:?}; shift 2 ;;
    *) echo "usage: tools/build-prebuilt.sh [--gpu N] [--jobs J] [--sha FORK_SHA]" >&2; exit 64 ;;
  esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
context=$root/tools/prebuilt
image=rig-prebuilt:$(sha256sum "$context/Dockerfile" | cut -c1-12)
docker build -q -t "$image" "$context" > /dev/null
(cd "$root" && bun run build > /dev/null)
out=$root/local/prebuilt
mkdir -p "$out"
pin=()
if [ -n "$sha" ]; then
  [[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "--sha takes a full 40-hex fork commit" >&2; exit 64; }
  # the pin's published [[prebuilt]] entries name the pin's tarballs, so the copy drops them
  sed "s/^sha = \"[0-9a-f]\{40\}\"/sha = \"$sha\"/" "$root/engine/engine.toml" |
    awk '/^\[/ { skip = ($0 == "[[prebuilt]]") } !skip' > "$out/engine.toml"
  grep -q "^sha = \"$sha\"" "$out/engine.toml" || { echo "engine.toml has no [fork] sha line to replace" >&2; exit 1; }
  pin=(-v "$out/engine.toml:/rig/engine/engine.toml:ro")
fi

docker run --rm --device "nvidia.com/gpu=$gpu" --user "$(id -u):$(id -g)" \
  --memory 14g --memory-swap 14g --cpus 6 -e HOME=/tmp -e RIG_ROOT=/rig \
  -v "$root:/rig:ro" -v "$out:/rig/local" "${pin[@]}" "$image" \
  /rig/dist/rig build --gpu 0 --portable --jobs "$jobs"
ls "$out"/engine-builds/engine-sm*-*.tar.gz
