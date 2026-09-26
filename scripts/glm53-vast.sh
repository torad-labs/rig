#!/usr/bin/env bash
# One GLM rental beside rig's existing single-box rental; no public model port.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
state="$repo/local/glm53-vast"
units="$HOME/.config/systemd/user"
label=rig-glm53
image=vllm/vllm-openai@sha256:f8649f17426cb41a522d9de76ff40b173b9b92c0057fa4c68e2fcb194e444c63
mkdir -p "$state"
if [[ -n "${VASTAI:-}" ]]; then
  vastai=$VASTAI
elif [[ -s "$state/vastai-client" ]]; then
  vastai=$(< "$state/vastai-client")
else
  vastai=$HOME/.local/bin/vastai
fi

inventory() {
  timeout 15s "$vastai" --raw show instances | jq -e 'if type == "array" then . else error("Invalid Vast inventory") end'
}

matching_ids() {
  jq -r --arg label "$label" '.[] | select(.label == $label and .cur_state != "destroyed" and (.id | type == "number")) | .id' | sort -n | uniq
}

instance() {
  local listed saved
  listed=$(inventory) || return 1
  listed=$(matching_ids <<< "$listed") || return 1
  if [[ -s "$state/id" ]]; then
    saved=$(< "$state/id")
    if [[ -z "$listed" ]]; then
      printf 'Local GLM rental ID %s is absent from provider inventory\n' "$saved" >&2
      return 1
    fi
    if grep -Fxq -- "$saved" <<< "$listed"; then
      printf '%s\n' "$saved"
      return
    fi
    if [[ "$listed" == *$'\n'* ]]; then
      printf 'Multiple GLM rentals and local ID does not match; keeping watchdog armed\n' >&2
      return 1
    fi
    printf 'GLM rental changed from %s to %s; reconciling watchdog state\n' "$saved" "$listed" >&2
  fi
  printf '%s\n' "$listed"
}

ssh_args() {
  local address hostport
  address=$(timeout 15s "$vastai" ssh-url "$1") || return 1
  [[ "$address" == ssh://root@*:* ]] || { printf 'Invalid Vast SSH address\n' >&2; return 1; }
  hostport=${address#ssh://root@}
  ssh_port=${hostport##*:}
  ssh_host=${hostport%:*}
  [[ "$ssh_port" =~ ^[0-9]+$ && -n "$ssh_host" ]] || return 1
}

remote() {
  local id=$1
  shift
  ssh_args "$id" || return 1
  timeout 30s ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$state/known_hosts-$id" -p "$ssh_port" "root@$ssh_host" "$@"
}

admission() {
  curl -fsS --max-time 150 --unix-socket "$state/admission.sock" -X POST "http://localhost/$1" >/dev/null
}

restore_admission() {
  if (( ${fencing:-0} )) && [[ -s "$state/id" && $(< "$state/id") == "$id" ]]; then
    if ! admission resume; then
      if (( ${drained:-0} )); then
        printf 'Could not restore GLM admission; restarting its drained proxy\n' >&2
        systemctl --user restart glm53-vast-admission.service
      else
        printf 'GLM proxy drain was not confirmed; its bounded lease will reopen admission\n' >&2
      fi
    fi
  fi
}

repair_local_path() {
  if curl -fsS --max-time 3 http://127.0.0.1:18102/health >/dev/null 2>&1; then
    systemctl --user restart glm53-vast-admission.service
    printf 'Remote GLM and local tunnel healthy; restarted the unavailable admission proxy\n'
  else
    systemctl --user restart glm53-vast-tunnel.service
    printf 'Remote GLM healthy; restarted the unavailable local tunnel\n'
  fi
}

bootstrap_running() {
  remote "$1" 'pgrep -f "^bash /workspace/glm53-serve.sh$|^/usr/bin/python3 /usr/local/bin/vllm serve /workspace/glm53-model" >/dev/null' 2>/dev/null
}

install_timer() {
  local bun_bin
  bun_bin=$(command -v bun)
  mkdir -p "$units"
  printf '[Unit]\nDescription=Check GLM Vast rental for idle teardown\n[Service]\nType=oneshot\nTimeoutStartSec=240s\nExecStart=%s idle-check\n' \
    "$repo/scripts/glm53-vast.sh" > "$units/glm53-vast-idle.service"
  printf '[Unit]\nDescription=Check GLM Vast rental each minute\n[Timer]\nOnBootSec=1min\nOnUnitActiveSec=1min\nAccuracySec=10s\n[Install]\nWantedBy=timers.target\n' \
    > "$units/glm53-vast-idle.timer"
  printf '[Unit]\nDescription=Local-only tunnel to GLM Vast rental\nAfter=network-online.target\n[Service]\nType=simple\nExecStart=%s tunnel\nRestart=always\nRestartSec=10\n[Install]\nWantedBy=default.target\n' \
    "$repo/scripts/glm53-vast.sh" > "$units/glm53-vast-tunnel.service"
  printf '[Unit]\nDescription=Local GLM request admission and drain\nAfter=glm53-vast-tunnel.service\n[Service]\nType=simple\nEnvironment=GLM53_ADMISSION_SOCKET=%s/admission.sock\nExecStart=%s %s/glm53-admission.ts\nRestart=always\nRestartSec=5\n[Install]\nWantedBy=default.target\n' \
    "$state" "$bun_bin" "$repo/scripts" > "$units/glm53-vast-admission.service"
  systemctl --user daemon-reload
  systemctl --user enable --now glm53-vast-idle.timer
}

mark_activity() {
  date +%s > "$state/last-activity"
  printf '%s\n' "$1" > "$state/counter"
}

activity() {
  awk '
    function finite(value) {
      return value ~ /^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/ &&
        tolower(sprintf("%g", value + 0)) !~ /(nan|inf)/
    }
    $1 ~ /^vllm:num_requests_running(\{|$)/ {
      if (!finite($NF) || $NF + 0 < 0) invalid = 1
      else {running += $NF; running_seen++}
    }
    $1 ~ /^vllm:num_requests_waiting(\{|$)/ {
      if (!finite($NF) || $NF + 0 < 0) invalid = 1
      else {waiting += $NF; waiting_seen++}
    }
    $1 ~ /^vllm:(prompt_tokens(_total)?|generation_tokens(_total)?|request_success_total)(\{|$)/ {
      if (!finite($NF) || $NF + 0 < 0) invalid = 1
      else {counter += $NF; samples++}
    }
    END {
      if (invalid || !running_seen || !waiting_seen || !samples ||
          !finite(running) || !finite(waiting) || !finite(counter)) exit 1
      printf "%d %d %.0f\n", running, waiting, counter
    }
  ' <<< "$1"
}

metrics_outage() {
  local id=$1 now since
  now=$(date +%s)
  if [[ -s "$state/metrics-outage-since" ]]; then
    since=$(< "$state/metrics-outage-since")
  else
    since=0
  fi
  if [[ ! "$since" =~ ^[0-9]+$ ]] || (( since == 0 || since > now )); then
    since=$now
    printf '%s\n' "$since" > "$state/metrics-outage-since"
  fi
  if (( now - since >= 7200 )); then
    printf 'GLM metrics unavailable for two hours; destroying billed rental\n' >&2
    teardown "$id"
    exit 1
  fi
}

teardown() {
  local id=$1 listed remaining next
  # The caller has proved the server idle, or that remote bootstrap terminated.
  listed=$(inventory) || return 1
  if ! matching_ids <<< "$listed" | grep -Fxq -- "$id"; then
    printf 'Refusing to destroy unverified GLM instance %s\n' "$id" >&2
    return 1
  fi
  timeout 30s "$vastai" destroy instance "$id" --yes
  listed=$(inventory) || return 1
  remaining=$(jq -r --argjson id "$id" '[.[] | select(.id == $id and .cur_state != "destroyed")] | length' <<< "$listed")
  [[ "$remaining" == 0 ]] || { printf 'Vast still lists instance %s; keeping watchdog armed\n' "$id" >&2; return 1; }
  next=$(matching_ids <<< "$listed") || return 1
  if [[ -n "$next" ]]; then
    next=${next%%$'\n'*}
    printf '%s\n' "$next" > "$state/id"
    rm -f "$state/ready" "$state/maintenance-until" "$state/metrics-outage-since"
    date +%s > "$state/started"
    mark_activity 0
    systemctl --user restart glm53-vast-tunnel.service glm53-vast-admission.service
    printf 'Destroyed GLM instance %s; monitoring remaining instance %s\n' "$id" "$next"
    return 0
  fi
  systemctl --user disable --now glm53-vast-idle.timer glm53-vast-admission.service glm53-vast-tunnel.service
  rm -f "$state/id" "$state/counter" "$state/last-activity" "$state/started" "$state/ready" "$state/maintenance-until" "$state/metrics-outage-since" "$state/vastai-client"
  printf 'Destroyed idle/failed GLM instance %s\n' "$id"
}

command=${1:-status}
if [[ ${0##*/} == glm53-off && $# == 0 ]]; then command=down; fi
case "$command" in
  up)
    offer=${2:?usage: glm53-vast.sh up OFFER_ID}
    [[ "$offer" =~ ^[0-9]+$ ]] || { printf 'Offer ID must be numeric\n' >&2; exit 2; }
    exec 9> "$state/up.lock"
    flock -n 9 || { printf 'GLM rental setup already in progress\n' >&2; exit 1; }
    existing=$(instance) || { printf 'Could not inspect GLM rentals; refusing a second rental\n' >&2; exit 1; }
    [[ -z "$existing" ]] || { printf 'GLM rental already exists\n' >&2; exit 1; }
    "$vastai" --raw search offers \
      'gpu_name=H200 num_gpus=2 disk_space>=400 bw_nvlink>0 cuda_vers>=12.9' \
      --storage 400 --limit 200 -o dph | jq -e --argjson id "$offer" \
      '[.[] | select(.id == $id and (.dph_total | type == "number" and isfinite and . >= 0 and . <= 11))] | length == 1' \
      >/dev/null || { printf 'Offer is not an eligible two-H200 rental under 11 USD/h\n' >&2; exit 1; }
    hf download orcarouter/GLM-5.3-Flash-Uncensored-NVFP4 config.json --dry-run \
      --revision ec0adf4f49c9570807cc11a5f650538c1893ae54 >/dev/null
    rm -f "$state/ready" "$state/metrics-outage-since"
    vastai=$(realpath "$(command -v "$vastai")")
    printf '%s\n' "$vastai" > "$state/vastai-client"
    printf '%s\n' "$(( $(date +%s) + 900 ))" > "$state/maintenance-until"
    install_timer  # Arm cost control before the billed instance exists.
    response=$("$vastai" --raw create instance "$offer" --image "$image" --disk 400 \
      --ssh --direct --cancel-unavail --label "$label")
    id=$(jq -er '.new_contract' <<< "$response")
    printf '%s\n' "$id" > "$state/id"
    date +%s > "$state/started"
    mark_activity 0
    printf 'Rented GLM box %s; waiting for SSH and starting remote download\n' "$id"
    for attempt in {1..60}; do
      if remote "$id" true 2>/dev/null; then break; fi
      if (( attempt == 60 )); then
        printf 'SSH not ready; rental and idle timer remain armed: %s\n' "$id" >&2
        exit 1
      fi
      sleep 10
    done
    remote "$id" 'mkdir -p /workspace && cat > /workspace/glm53-serve.sh && chmod 700 /workspace/glm53-serve.sh' \
      < "$repo/scripts/glm53-serve.sh"
    hf auth token | remote "$id" 'umask 077; mkdir -p /root/.cache/huggingface; cat > /root/.cache/huggingface/token'
    if ! remote "$id" 'nohup bash /workspace/glm53-serve.sh > /workspace/glm53.log 2>&1 < /dev/null &'; then
      remote "$id" 'rm -f /root/.cache/huggingface/token' || true
      printf 'Remote launch failed; credential cleanup attempted and idle timer remains armed\n' >&2
      exit 1
    fi
    systemctl --user enable --now glm53-vast-tunnel.service glm53-vast-admission.service
    for attempt in {1..5}; do
      if bootstrap_running "$id"; then
        rm -f "$state/maintenance-until"
        printf 'GLM downloading and starting on Vast box %s; inspect with glm53-vast.sh status\n' "$id"
        exit 0
      fi
      sleep 2
    done
    printf 'GLM launch not yet confirmed; bounded bootstrap lease and idle watchdog remain armed\n' >&2
    exit 1
    ;;
  refresh-watchdog)
    install_timer
    ;;
  tunnel)
    id=$(instance)
    [[ "$id" =~ ^[0-9]+$ ]] || exit 1
    ssh_args "$id"
    exec ssh -MN \
      -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$state/known_hosts-$id" \
      -p "$ssh_port" -L 127.0.0.1:18102:127.0.0.1:8000 "root@$ssh_host"
    ;;
  down)
    id=$(instance)
    [[ "$id" =~ ^[0-9]+$ ]] || { printf 'No unambiguous GLM rental to stop\n' >&2; exit 1; }
    if ! metrics=$(curl -fsS --max-time 10 http://127.0.0.1:8102/metrics 2>/dev/null) &&
      ! metrics=$(remote "$id" 'curl -fsS --max-time 10 http://127.0.0.1:8000/metrics' 2>/dev/null); then
      printf 'Cannot verify GLM request activity; refusing manual shutdown\n' >&2
      exit 1
    fi
    sample=$(activity "$metrics") || { printf 'Cannot parse GLM request activity; refusing manual shutdown\n' >&2; exit 1; }
    read -r running waiting _ <<< "$sample"
    if (( running > 0 || waiting > 0 )); then
      printf 'GLM has active or queued requests; refusing manual shutdown\n' >&2
      exit 1
    fi
    fencing=1
    trap restore_admission EXIT
    admission drain || { printf 'Could not drain GLM admission; refusing manual shutdown\n' >&2; exit 1; }
    drained=1
    final=$(remote "$id" 'curl -fsS --max-time 10 http://127.0.0.1:8000/metrics' 2>/dev/null) || {
      printf 'Cannot verify drained GLM activity; refusing manual shutdown\n' >&2
      exit 1
    }
    final_sample=$(activity "$final") || { printf 'Cannot parse drained GLM activity; refusing manual shutdown\n' >&2; exit 1; }
    read -r running waiting _ <<< "$final_sample"
    if (( running > 0 || waiting > 0 )); then
      printf 'GLM activity remains after drain; refusing manual shutdown\n' >&2
      exit 1
    fi
    teardown "$id"
    fencing=0
    trap - EXIT
    ;;
  idle-check)
    id=$(instance)
    [[ -n "$id" ]] || exit 0
    [[ "$id" =~ ^[0-9]+$ ]] || { printf 'Ambiguous GLM rental state: %s\n' "$id" >&2; exit 1; }
    if [[ -s "$state/id" && $(< "$state/id") != "$id" ]]; then
      printf '%s\n' "$id" > "$state/id"
      rm -f "$state/ready" "$state/maintenance-until" "$state/metrics-outage-since"
      date +%s > "$state/started"
      mark_activity 0
      systemctl --user restart glm53-vast-tunnel.service
    fi
    if rental_json=$(timeout 15s "$vastai" --raw show instance "$id"); then
      rental_state=$(jq -r 'if type == "array" then .[0] else . end | .cur_state' <<< "$rental_json") || rental_state=unknown
    else
      rental_state=unknown
      printf 'Vast status temporarily unavailable; checking local request activity\n' >&2
    fi
    if [[ "$rental_state" == stopped ]]; then
      teardown "$id"  # A stopped instance still bills for its allocated disk.
      exit 0
    fi
    [[ -s "$state/id" ]] || printf '%s\n' "$id" > "$state/id"
    [[ -s "$state/started" ]] || date +%s > "$state/started"
    [[ -s "$state/last-activity" ]] || mark_activity 0
    if [[ -s "$state/maintenance-until" ]]; then
      maintenance_until=$(< "$state/maintenance-until")
      if [[ "$maintenance_until" =~ ^[0-9]+$ ]] && (( $(date +%s) < maintenance_until )); then
        printf 'GLM restart in progress; bounded maintenance lease active\n'
        exit 0
      fi
      rm -f "$state/maintenance-until"
    fi
    remote_fallback=0
    if ! metrics=$(curl -fsS --max-time 10 http://127.0.0.1:8102/metrics 2>/dev/null); then
      if metrics=$(remote "$id" 'curl -fsS --max-time 10 http://127.0.0.1:8000/metrics' 2>/dev/null); then
        remote_fallback=1
      else
        metrics_outage "$id"
      if remote "$id" 'test -f /workspace/glm53.failed' 2>/dev/null; then
        remote "$id" 'tail -n 1000 /workspace/glm53.log' > "$state/failure.log" 2>&1 || true
        printf 'GLM startup failed; log saved under local/glm53-vast/failure.log\n' >&2
        teardown "$id"
      elif remote "$id" true 2>/dev/null; then
        now=$(date +%s)
        started=$(< "$state/started")
        if bootstrap_running "$id" &&
          { [[ -f "$state/ready" ]] || (( now - started < 5400 )); }; then
          printf 'GLM process still active; preserving instance until metrics recover\n' >&2
        else
          remote "$id" 'rm -f /root/.cache/huggingface/token; tail -n 1000 /workspace/glm53.log' > "$state/failure.log" 2>&1 || true
          printf 'GLM bootstrap failed or exceeded 90 minutes; destroying billed rental\n' >&2
          teardown "$id"
        fi
      else
        printf 'GLM rental unreachable; preserving instance during bounded metrics outage\n' >&2
      fi
      exit 1
      fi
    fi
    sample=$(activity "$metrics") || {
      metrics_outage "$id"
      printf 'Incomplete vLLM activity metrics; preserving instance during outage grace\n' >&2
      exit 1
    }
    remote "$id" 'rm -f /workspace/glm53.failed' 2>/dev/null || true
    read -r running waiting counter <<< "$sample"
    if [[ ! -f "$state/ready" ]]; then
      rm -f "$state/metrics-outage-since"
      touch "$state/ready"
      mark_activity "$counter"  # The idle budget starts only after the model serves.
      if (( remote_fallback )); then repair_local_path; fi
      exit 0
    fi
    previous=$(< "$state/counter")
    if (( running > 0 || waiting > 0 )) || [[ "$counter" != "$previous" ]]; then
      rm -f "$state/metrics-outage-since"
      mark_activity "$counter"
      exit 0
    fi
    now=$(date +%s)
    last=$(< "$state/last-activity")
    if (( now - last >= 600 )); then
      fencing=1
      trap restore_admission EXIT
      if ! admission drain; then
        printf 'Could not drain GLM admission; preserving rental instead of interrupting active work\n' >&2
        exit 1
      fi
      drained=1
      second=$(remote "$id" 'curl -fsS --max-time 10 http://127.0.0.1:8000/metrics' 2>/dev/null) || {
        metrics_outage "$id"
        exit 1
      }
      second_sample=$(activity "$second") || {
        metrics_outage "$id"
        exit 1
      }
      rm -f "$state/metrics-outage-since"
      if [[ "$second_sample" != "$sample" ]]; then
        read -r _ _ counter <<< "$second_sample"
        mark_activity "$counter"
        printf 'GLM activity changed before teardown; preserving instance\n'
        exit 0
      fi
      teardown "$id"
      fencing=0
      trap - EXIT
    else
      rm -f "$state/metrics-outage-since"
      if (( remote_fallback )); then repair_local_path; fi
      printf 'GLM box %s idle for %s seconds; timer active\n' "$id" "$((now - last))"
    fi
    ;;
  status)
    id=$(instance)
    [[ -n "$id" ]] || { printf 'No GLM rental\n'; exit 0; }
    printf 'GLM rental %s\n' "$id"
    "$vastai" --raw show instance "$id" | jq -r 'if type == "array" then .[0] else . end | "Vast: \(.cur_state), \(.num_gpus)x \(.gpu_name), $\(.dph_total)/h"'
    systemctl --user is-active glm53-vast-idle.timer glm53-vast-tunnel.service glm53-vast-admission.service || true
    curl -fsS --max-time 3 http://127.0.0.1:8102/health >/dev/null && printf 'vLLM healthy\n' || printf 'vLLM not ready\n'
    ;;
  *) printf 'usage: glm53-vast.sh {up OFFER_ID|down|status|idle-check|tunnel|refresh-watchdog}\n' >&2; exit 2 ;;
esac
