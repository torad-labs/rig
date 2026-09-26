#!/usr/bin/env bash
# Run inside Vast's pinned vLLM image. SSH stays available for the local-only tunnel.
set -euo pipefail
umask 077

model=orcarouter/GLM-5.3-Flash-Uncensored-NVFP4
revision=ec0adf4f49c9570807cc11a5f650538c1893ae54
model_dir=/workspace/glm53-model
export HF_HOME=/root/.cache/huggingface
mkdir -p "$model_dir"
on_error() {
  local status=$?
  rm -f "$HF_HOME/token"
  printf '%s\n' "$status" > /workspace/glm53.failed
  exit "$status"
}
trap on_error ERR
rm -f /workspace/glm53.failed

# The short-lived access token arrives over SSH and is needed only for a fresh download.
if [[ ! -f "$model_dir/.download-complete" ]]; then
  HF_TOKEN=$(< "$HF_HOME/token")
  export HF_TOKEN
  hf download "$model" --revision "$revision" --local-dir "$model_dir"
  unset HF_TOKEN
  touch "$model_dir/.download-complete"
fi
rm -f "$HF_HOME/token"

export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_KV_CACHE_LAYOUT=HND
vllm serve "$model_dir" \
  --served-model-name GLM-5.3-Flash-Uncensored-NVFP4 \
  --tensor-parallel-size 2 --max-model-len 1048576 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --max-cudagraph-capture-size 64 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --enable-prefix-caching --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
