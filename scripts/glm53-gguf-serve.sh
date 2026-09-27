#!/usr/bin/env bash
# Run on the separate two-GPU Vast rental with a pinned rig llama-server build.
set -euo pipefail
umask 077

model=BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF
revision=1bc0afadd92f4ff9fc2b1281a95a7921b058d35d
model_dir=/workspace/glm53-gguf-model
engine_dir=/workspace/glm53-engine

on_error() {
  local status=$?
  printf '%s\n' "$status" > /workspace/glm53.failed
  exit "$status"
}
trap on_error ERR
rm -f /workspace/glm53.failed

[[ -x "$engine_dir/llama-server" && -s "$engine_dir/BUILD" ]] || {
  printf 'Pinned rig engine is missing\n' >&2
  printf '1\n' > /workspace/glm53.failed
  exit 1
}
if [[ ! -f "$model_dir/.download-complete" ]]; then
  mkdir -p "$model_dir"
  hf download "$model" --revision "$revision" --include 'IQ3_XXS/*.gguf' --local-dir "$model_dir"
  sha256sum -c <<CHECKSUMS
9916f01db53f8a2bfd5db1d6eb8cd48f93ae2b732fe1dfce993450745334cae1  $model_dir/IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf
139f2ea5e2d1bb847cd21c5d5122e3a838fde7b5ae2895c6e2623ff318519a00  $model_dir/IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00002-of-00003.gguf
b94bcd70929d7f4445d3acb6490a801511452214c8cf633fa2d912bef89f2188  $model_dir/IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00003-of-00003.gguf
CHECKSUMS
  touch "$model_dir/.download-complete"
fi

export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH="$engine_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$engine_dir/llama-server" \
  -m "$model_dir/IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf" \
  --host 127.0.0.1 --port 8000 --metrics --jinja \
  --chat-template-file /workspace/glm53-chat-template.jinja \
  --alias GLM-5.3-Flash-Uncensored-IQ3_XXS \
  -sm layer -ts 1/1 -fa on -ctk f16 -ctv f16 \
  -c 524288 -np 1 -b 4096 -ub 512 \
  --spec-type draft-mtp --spec-draft-n-max 2
