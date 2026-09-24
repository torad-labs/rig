#!/bin/sh
# Installs rig from a GitHub release: the binary, the engine pin and the heads into
# ~/.local/share/rig, linked as ~/.local/bin/rig. What rig builds and fetches afterwards (the
# engine, the packs, logs) lives in ~/.local/share/rig/local/, which an upgrade never touches.
#
#   curl -fsSL https://github.com/torad-labs/rig/releases/latest/download/install.sh | sh
#
# RIG_VERSION=v0.1.0 pins a release; RIG_HOME and RIG_BIN_DIR move the two directories;
# RIG_RELEASE_URL fetches the release files from somewhere else (a mirror).
set -eu

asset=rig-linux-x64.tar.gz
home=${RIG_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/rig}
bin_dir=${RIG_BIN_DIR:-$HOME/.local/bin}
if [ -n "${RIG_RELEASE_URL:-}" ]; then
  base=$RIG_RELEASE_URL
elif [ -n "${RIG_VERSION:-}" ]; then
  base=https://github.com/torad-labs/rig/releases/download/$RIG_VERSION
else
  base=https://github.com/torad-labs/rig/releases/latest/download
fi

fail() {
  echo "rig install: $*" >&2
  exit 1
}

[ "$(uname -s)" = Linux ] || fail "rig runs on Linux with an NVIDIA card (this is $(uname -s))"
[ "$(uname -m)" = x86_64 ] || fail "rig is built for x86_64 (this is $(uname -m))"
for tool in curl tar sha256sum; do
  command -v "$tool" > /dev/null 2>&1 || fail "$tool is missing on this machine"
done

mkdir -p "$home" "$bin_dir"
# staged beside the install, so every replacement below is a rename on one filesystem
staging=$(mktemp -d "$home/.install-XXXXXX")
trap 'rm -rf "$staging"' EXIT
trap 'exit 130' INT TERM

echo "rig install: fetching $base/$asset"
curl -fsSL --retry 3 -o "$staging/$asset" "$base/$asset"
curl -fsSL --retry 3 -o "$staging/$asset.sha256" "$base/$asset.sha256"
(cd "$staging" && sha256sum -c --quiet "$asset.sha256") ||
  fail "$asset does not match the sha256 published with it"

mkdir "$staging/rig"
tar -C "$staging/rig" -xzf "$staging/$asset" --strip-components=1
[ -x "$staging/rig/dist/rig" ] || fail "the release holds no dist/rig"
[ ! -e "$staging/rig/local" ] || fail "the release holds a local/, which is this machine's own"

# each entry the release holds replaces the installed one; anything else under $home stays
for path in "$staging/rig"/*; do
  entry=$(basename "$path")
  if [ -e "$home/$entry" ]; then mv "$home/$entry" "$staging/replaced-$entry"; fi
  mv "$path" "$home/$entry"
done
ln -sfn "$home/dist/rig" "$bin_dir/rig"

echo "rig install: $("$bin_dir/rig" --version) in $home, linked as $bin_dir/rig"
case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *) echo "rig install: $bin_dir is not on PATH; add it in your shell's rc file" ;;
esac
echo "next: rig up bonsai-2-27b   (checks the card and driver, installs the engine, fetches the model)"
