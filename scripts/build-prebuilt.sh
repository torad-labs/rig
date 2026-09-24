#!/usr/bin/env bash
# Builds the pin's prebuilt engine in scripts/prebuilt/Dockerfile's image, so its floor is that
# image's (Ubuntu 22.04, glibc 2.35) and not whatever the building machine runs: `rig build
# --portable` inside the container, as the calling user, with its own local/ (local/prebuilt/, so no
# build tree mixes with the host's), capped at 14 GiB and 6 CPUs (the engine's largest CUDA
# objects need ~13 GiB to compile). The card is passed through only for `rig build` to read its
# compute capability. The tarball it prints is what scripts/e2e-driver-only.sh --prebuilt gates.
#
#   scripts/build-prebuilt.sh [--gpu N] [--jobs J]
set -euo pipefail

gpu=0 jobs=6
while [ $# -gt 0 ]; do
  case $1 in
    --gpu) gpu=${2:?}; shift 2 ;;
    --jobs) jobs=${2:?}; shift 2 ;;
    *) echo "usage: scripts/build-prebuilt.sh [--gpu N] [--jobs J]" >&2; exit 64 ;;
  esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
context=$root/scripts/prebuilt
image=rig-prebuilt:$(sha256sum "$context/Dockerfile" | cut -c1-12)
docker build -q -t "$image" "$context" > /dev/null
(cd "$root" && bun run build > /dev/null)
out=$root/local/prebuilt
mkdir -p "$out"

docker run --rm --device "nvidia.com/gpu=$gpu" --user "$(id -u):$(id -g)" \
  --memory 14g --memory-swap 14g --cpus 6 -e HOME=/tmp -e RIG_ROOT=/rig \
  -v "$root:/rig:ro" -v "$out:/rig/local" "$image" \
  /rig/dist/rig build --gpu 0 --portable --jobs "$jobs"
ls "$out"/engine-builds/engine-sm*-*.tar.gz
