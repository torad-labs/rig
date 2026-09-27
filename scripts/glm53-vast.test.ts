import { expect, test } from "bun:test";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const runner = join(import.meta.dir, "glm53-vast.sh");

function checkIdle(
  metrics: string | null,
  ready = true,
  failed = false,
  rejectDestroy = false,
  crashed = false,
  secondMetrics: string | null = null,
  options: {
    remoteHealthy?: boolean;
    remoteMetrics?: string;
    bootstrapAlive?: boolean;
    startedRecently?: boolean;
    maintenance?: "active" | "expired";
    mode?: "up" | "up-gguf" | "down" | "off" | "pause" | "resume";
    providerState?: "running" | "stopped" | "scheduling";
    paused?: boolean;
    stopIgnored?: boolean;
    startFails?: boolean;
    scheduling?: boolean;
    sshDown?: boolean;
    remoteEngineStale?: boolean;
    modelOnDisk?: boolean;
    portableEngine?: boolean;
    nativeEngine?: boolean;
    offerGpuName?: string;
    offerGpuCount?: number;
    offerCuda?: number | string;
    createRental?: boolean;
    smokeFails?: boolean;
    uploadFails?: boolean;
    launcherCopyFails?: boolean;
    templateCopyFails?: boolean;
    launcherFails?: boolean;
    actualPrice?: string;
    checkoutAhead?: boolean;
    inventoryFails?: boolean;
    outageSeconds?: number;
    providerHangs?: boolean;
    sshLookupHangs?: boolean;
    drainFails?: boolean;
    offerPrice?: string;
    sibling?: boolean;
    wrongLabel?: boolean;
    persistedClient?: boolean;
    recentlyActive?: boolean;
    backendHealthy?: boolean;
    idleAgeSeconds?: number;
    runtime?: "gguf";
  } = {},
) {
  const scratch = mkdtempSync(join(tmpdir(), "rig-glm53-idle-"));
  const scripts = join(scratch, "scripts");
  const state = join(scratch, "local", "glm53-vast");
  const bin = join(scratch, "bin");
  mkdirSync(scripts);
  mkdirSync(state, { recursive: true });
  mkdirSync(bin);
  const script = join(scripts, "glm53-vast.sh");
  copyFileSync(runner, script);
  chmodSync(script, 0o755);
  if (options.mode === "up-gguf") {
    copyFileSync(
      join(import.meta.dir, "glm53-gguf-serve.sh"),
      join(scripts, "glm53-gguf-serve.sh"),
    );
    copyFileSync(
      join(import.meta.dir, "glm53-chat-template.jinja"),
      join(scripts, "glm53-chat-template.jinja"),
    );
  }
  if (options.portableEngine) {
    const pin = "1234567abcdef1234567abcdef1234567abcdef12";
    const engine = join(scratch, "local", "engine-builds", "1234567-sm120");
    mkdirSync(engine, { recursive: true });
    mkdirSync(join(scratch, "engine"));
    writeFileSync(
      join(scratch, "engine", "engine.toml"),
      `[fork]\nsha = "${pin}"\n[[prebuilt]]\ncap = "120"\n`,
    );
    writeFileSync(
      join(engine, "BUILD"),
      `fork=${pin} cap=sm_120 native=${options.nativeEngine ? "on" : "off"} source=tarball:test\n`,
    );
    writeFileSync(join(engine, "llama-server"), "#!/bin/bash\nprintf 'version 1234567\\n'\n");
    chmodSync(join(engine, "llama-server"), 0o755);
    writeFileSync(join(engine, "llama-bench"), "#!/bin/bash\nexit 0\n");
    chmodSync(join(engine, "llama-bench"), 0o755);
    for (const lib of ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"])
      writeFileSync(join(engine, lib), "pinned runtime");
    writeFileSync(
      join(bin, "git"),
      `#!/bin/bash\nprintf '${options.checkoutAhead ? "bc59cc7a8bc59cc7a8bc59cc7a8bc59cc7a8bc59" : pin}\\n'\n`,
    );
    chmodSync(join(bin, "git"), 0o755);
  }
  if (options.mode !== "up" && options.mode !== "up-gguf")
    writeFileSync(join(state, "id"), "123\n");
  if (options.paused) writeFileSync(join(state, "paused"), "1\n");
  if (options.runtime) writeFileSync(join(state, "runtime"), `${options.runtime}\n`);
  if (options.persistedClient)
    writeFileSync(join(state, "vastai-client"), `${join(bin, "vastai")}\n`);
  writeFileSync(
    join(state, "started"),
    `${options.startedRecently ? Math.floor(Date.now() / 1000) : 1}\n`,
  );
  writeFileSync(
    join(state, "last-activity"),
    `${Math.floor(Date.now() / 1000) - (options.idleAgeSeconds ?? (options.recentlyActive ? 30 : 4000))}\n`,
  );
  writeFileSync(join(state, "counter"), "12\n");
  if (ready) writeFileSync(join(state, "ready"), "");
  if (options.outageSeconds)
    writeFileSync(
      join(state, "metrics-outage-since"),
      `${Math.floor(Date.now() / 1000) - options.outageSeconds}\n`,
    );
  if (options.maintenance)
    writeFileSync(
      join(state, "maintenance-until"),
      `${Math.floor(Date.now() / 1000) + (options.maintenance === "active" ? 900 : -1)}\n`,
    );
  const log = join(scratch, "calls");
  // the provider's rental 123: running, stopped (paused, disk kept), or scheduling (a start waiting for the machine's
  // GPUs), in the fields Vast reports (cur_state: the hardware allocation; intended_status: the requested state)
  writeFileSync(join(scratch, "provider"), `${options.providerState ?? "running"}\n`);
  writeFileSync(
    join(bin, "vastai"),
    `#!/bin/bash
rental() {
  case $(< "$MOCK_PROVIDER") in
    stopped) printf '{"id":123,"label":"rig-glm53","cur_state":"stopped","intended_status":"stopped","actual_status":"stopped"}' ;;
    scheduling) printf '{"id":123,"label":"rig-glm53","cur_state":"stopped","intended_status":"running","actual_status":"stopped"}' ;;
    *) printf '{"id":123,"label":"rig-glm53","cur_state":"running","intended_status":"running","actual_status":"running"}' ;;
  esac
}
if [[ $1 == --raw && $2 == show && $3 == instances ]]; then
  if [[ $MOCK_INVENTORY_FAIL == 1 ]]; then exit 7; fi
  if [[ $MOCK_MODE_UP == 1 ]]; then printf '[]\\n'
  elif [[ $MOCK_REJECT != 1 ]] && grep -q '^destroy instance ' "$MOCK_LOG" 2>/dev/null; then
    if [[ $MOCK_SIBLING == 1 ]]; then printf '[{"id":124,"label":"rig-glm53","cur_state":"running"}]\\n'; else printf '[]\\n'; fi
  elif [[ $MOCK_WRONG_LABEL == 1 ]]; then printf '[{"id":123,"label":"other","cur_state":"running"}]\\n'
  elif [[ $MOCK_SIBLING == 1 ]]; then printf '[%s,{"id":124,"label":"rig-glm53","cur_state":"running"}]\\n' "$(rental)"
  else printf '[%s]\\n' "$(rental)"; fi
elif [[ $1 == --raw && $2 == search ]]; then printf '[{"id":123,"dph_total":%s}]\\n' "$MOCK_OFFER_PRICE"
elif [[ $1 == --raw && $2 == create ]]; then
  if [[ -s "$MOCK_STATE/maintenance-until" ]]; then printf 'leased\\n'; else printf 'unleased\\n'; fi >> "$MOCK_LOG"; exit 77
elif [[ $1 == --raw && $2 == show && $3 == instance ]]; then
  if [[ $MOCK_PROVIDER_HANG == 1 ]]; then sleep 5; fi
  rental; printf '\\n'
elif [[ $1 == --raw ]]; then printf '{"cur_state":"running"}\\n'
elif [[ $1 == ssh-url ]]; then if [[ $MOCK_SSH_HANG == 1 ]]; then sleep 5; fi; printf 'ssh://root@127.0.0.1:22\\n'
elif [[ $1 == destroy ]]; then
  if [[ $* == *--yes* && $MOCK_REJECT != 1 ]]; then printf '%s\\n' "$*" >> "$MOCK_LOG"; else printf 'Aborted.\\n'; fi
elif [[ $1 == stop && $2 == instance ]]; then
  printf '%s\\n' "$*" >> "$MOCK_LOG"
  if [[ $MOCK_STOP_IGNORED != 1 ]]; then printf 'stopped\\n' > "$MOCK_PROVIDER"; fi
elif [[ $1 == start && $2 == instance ]]; then
  printf '%s\\n' "$*" >> "$MOCK_LOG"
  if [[ $MOCK_START_FAIL == 1 ]]; then exit 1; fi
  if [[ $MOCK_SCHEDULING == 1 ]]; then printf 'scheduling\\n'; else printf 'running\\n'; fi > "$MOCK_PROVIDER"
fi
`,
  );
  writeFileSync(
    join(bin, "ssh"),
    '#!/bin/bash\nif [[ $* == *"curl -fsS --max-time 10"* && -f "$MOCK_STATE/quiesced" ]]; then printf "%s\\n" "$MOCK_SECOND_METRICS"; exit 0; fi\nif [[ $MOCK_FAILED == 1 && $* == *"test -f /workspace/glm53.failed"* ]]; then exit 0; fi\nif [[ $* == *"tail -n 1000"* ]]; then printf "load failed\\n"; exit 0; fi\nif [[ $MOCK_REMOTE_HEALTHY == 1 && $* == *"curl -fsS --max-time 10"* ]]; then printf "%s\\n" "$MOCK_REMOTE_METRICS"; exit 0; fi\nif [[ $MOCK_BOOT_ALIVE == 1 && $* == *"pgrep -f"* ]]; then exit 0; fi\nif [[ $MOCK_CRASHED == 1 && $* == *"root@127.0.0.1 true"* ]]; then exit 0; fi\nexit 1\n',
  );
  writeFileSync(
    join(bin, "curl"),
    '#!/bin/bash\nif [[ $* == *"http://localhost/drain"* ]]; then if [[ $MOCK_DRAIN_FAIL == 1 ]]; then exit 7; fi; touch "$MOCK_STATE/quiesced"; printf "drain\\n" >> "$MOCK_LOG"; exit 0; fi\nif [[ $* == *"http://localhost/resume"* ]]; then printf "resume\\n" >> "$MOCK_LOG"; exit 0; fi\nif [[ $* == *"127.0.0.1:18102/health"* ]]; then [[ $MOCK_BACKEND_HEALTHY == 1 ]]; exit; fi\n[[ $MOCK_DOWN == 1 ]] && exit 7\nif [[ -f $MOCK_CURL_CALLS ]]; then printf "%s\\n" "$MOCK_SECOND_METRICS"; else touch "$MOCK_CURL_CALLS"; printf "%s\\n" "$MOCK_METRICS"; fi\n',
  );
  writeFileSync(
    join(bin, "timeout"),
    '#!/bin/bash\nif [[ $MOCK_PROVIDER_HANG == 1 && $* == *"show instance "* ]]; then exit 124; fi\nif [[ $MOCK_SSH_HANG == 1 && $* == *"ssh-url"* ]]; then exit 124; fi\nexec /usr/bin/timeout "$@"\n',
  );
  writeFileSync(
    join(bin, "hf"),
    '#!/bin/bash\nif [[ $* == "auth token" ]]; then printf "token\\n" >> "$MOCK_LOG"; printf "hf_test\\n"; fi\nexit 0\n',
  );
  writeFileSync(
    join(bin, "systemctl"),
    '#!/bin/bash\nif [[ $* == *"restart glm53-vast-"* || $* == "--user "*"able --now glm53-vast-tunnel.service glm53-vast-admission.service" ]]; then printf "%s\\n" "$*" >> "$MOCK_LOG"; fi\nexit 0\n',
  );
  // the waits between provider and SSH polls cost nothing here
  writeFileSync(join(bin, "sleep"), "#!/bin/bash\nexit 0\n");
  if (options.mode === "resume") {
    copyFileSync(
      join(import.meta.dir, "glm53-gguf-serve.sh"),
      join(scripts, "glm53-gguf-serve.sh"),
    );
    copyFileSync(
      join(import.meta.dir, "glm53-chat-template.jinja"),
      join(scripts, "glm53-chat-template.jinja"),
    );
    copyFileSync(join(import.meta.dir, "glm53-serve.sh"), join(scripts, "glm53-serve.sh"));
    writeFileSync(
      join(bin, "ssh"),
      `#!/bin/bash
if [[ $* == *"sha256sum /workspace/glm53-engine/llama-server"* ]]; then
  if [[ $MOCK_REMOTE_ENGINE_STALE == 1 && ! -f "$MOCK_UPLOADED" ]]; then printf 'stale  /workspace/glm53-engine/llama-server\\n'; else sha256sum "$MOCK_ENGINE_DIR/llama-server"; fi
elif [[ $* == *"ldd -r /workspace/glm53-engine/llama-server"* ]]; then
  for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do printf '%s => /workspace/glm53-engine/%s (0x1)\\n' "$lib" "$lib"; done
elif [[ $* == *"llama-server --version"* ]]; then printf 'version 1234567\\n'
elif [[ $* == *"tar -C /workspace/glm53-engine -xf -"* ]]; then printf 'upload\\n' >> "$MOCK_LOG"; dd of=/dev/null status=none; touch "$MOCK_UPLOADED"
elif [[ $* == *"test -f /workspace/glm53-model/.download-complete"* ]]; then [[ $MOCK_MODEL_ON_DISK == 1 ]]
elif [[ $* == *"cat > /root/.cache/huggingface/token"* ]]; then cat > /dev/null
elif [[ $* == *"nohup bash /workspace/"* ]]; then if [[ $MOCK_LAUNCH_FAIL == 1 ]]; then exit 7; fi; printf 'launch\\n' >> "$MOCK_LOG"
elif [[ $* == *"root@127.0.0.1 true"* ]]; then [[ $MOCK_SSH_DOWN != 1 ]]
elif [[ $* == *"pgrep -f"* ]]; then exit 0
elif [[ $* == *"curl -fsS"* ]]; then exit 1
fi
`,
    );
  }
  if (options.mode === "up-gguf") {
    const offer = JSON.stringify([
      {
        id: 123,
        num_gpus: options.offerGpuCount ?? 2,
        gpu_name: options.offerGpuName ?? "RTX PRO 6000 WS",
        gpu_ram: 97887,
        cuda_max_good: options.offerCuda ?? 13.3,
        dph_total: Number(options.offerPrice ?? 2.3),
      },
    ]);
    writeFileSync(
      join(bin, "vastai"),
      `#!/bin/bash\nif [[ $* == *"show instances"* ]]; then if [[ -s "$MOCK_STATE/id" ]] && ! grep -q '^destroy instance' "$MOCK_LOG" 2>/dev/null; then printf '[{"id":123,"label":"rig-glm53","cur_state":"running"}]\\n'; else printf '[]\\n'; fi; elif [[ $* == *"search offers"* ]]; then printf '%s\\n' '${offer}'; elif [[ $* == *"create instance"* ]]; then if [[ -s "$MOCK_STATE/maintenance-until" && -s "$MOCK_STATE/runtime" ]]; then printf 'leased\\n' >> "$MOCK_LOG"; fi; if [[ $MOCK_CREATE_RENTAL == 1 ]]; then printf '{"new_contract":123}\\n'; else exit 77; fi; elif [[ $* == *"show instance 123"* ]]; then printf '{"id":123,"label":"rig-glm53","cur_state":"running","num_gpus":2,"gpu_name":"RTX PRO 6000 WS","dph_total":%s}\\n' "$MOCK_ACTUAL_PRICE"; elif [[ $1 == ssh-url ]]; then printf 'ssh://root@127.0.0.1:22\\n'; elif [[ $1 == destroy ]]; then printf '%s\\n' "$*" >> "$MOCK_LOG"; fi\n`,
    );
    if (options.createRental) {
      writeFileSync(
        join(bin, "ssh"),
        '#!/bin/bash\nif [[ $* == *"sha256sum /workspace/glm53-engine/llama-server"* ]]; then sha256sum "$MOCK_ENGINE_DIR/llama-server"; elif [[ $* == *"ldd -r /workspace/glm53-engine/llama-server"* ]]; then for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do printf "%s => /workspace/glm53-engine/%s (0x1)\\n" "$lib" "$lib"; done; elif [[ $* == *"llama-server --version"* ]]; then printf "version 1234567\\n"; elif [[ $* == *"llama-bench"* ]]; then printf "smoke\\n" >> "$MOCK_LOG"; if [[ $MOCK_SMOKE_FAIL == 1 ]]; then exit 7; fi; printf "ggml_cuda_init: found 2 CUDA devices\\n"; elif [[ $* == *"tar -C /workspace/glm53-engine -xf -"* ]]; then if [[ $MOCK_UPLOAD_FAIL == 1 ]]; then exit 7; fi; dd of=/dev/null status=none; elif [[ $* == *"cat > /workspace/glm53-gguf-serve.sh"* && $MOCK_LAUNCHER_COPY_FAIL == 1 ]]; then exit 7; elif [[ $* == *"nohup bash /workspace/glm53-gguf-serve.sh"* && $MOCK_LAUNCH_FAIL == 1 ]]; then exit 7; elif [[ $* == *"cat > /workspace/glm53-chat-template.jinja"* && $MOCK_TEMPLATE_COPY_FAIL == 1 ]]; then exit 7; elif [[ $* == *"pgrep -f"* ]]; then exit 0; fi\n',
      );
    }
  }
  for (const command of ["vastai", "ssh", "curl", "timeout", "hf", "systemctl", "sleep"])
    chmodSync(join(bin, command), 0o755);

  try {
    const result = Bun.spawnSync(
      options.mode === "up" || options.mode === "up-gguf"
        ? [script, options.mode, "123"]
        : [script, options.mode ?? "idle-check"],
      {
        env: {
          ...process.env,
          HOME: scratch,
          PATH: `${bin}:${process.env.PATH}`,
          ...(options.persistedClient ? {} : { VASTAI: join(bin, "vastai") }),
          MOCK_LOG: log,
          MOCK_STATE: state,
          MOCK_DOWN: metrics === null ? "1" : "0",
          MOCK_FAILED: failed ? "1" : "0",
          MOCK_REJECT: rejectDestroy ? "1" : "0",
          MOCK_CRASHED: crashed ? "1" : "0",
          MOCK_REMOTE_HEALTHY: options.remoteHealthy ? "1" : "0",
          MOCK_REMOTE_METRICS: options.remoteMetrics ?? metrics ?? "",
          MOCK_BOOT_ALIVE: options.bootstrapAlive ? "1" : "0",
          MOCK_METRICS: metrics ?? "",
          MOCK_SECOND_METRICS: secondMetrics ?? options.remoteMetrics ?? metrics ?? "",
          MOCK_CURL_CALLS: join(scratch, "curl-calls"),
          MOCK_INVENTORY_FAIL: options.inventoryFails ? "1" : "0",
          MOCK_PROVIDER_HANG: options.providerHangs ? "1" : "0",
          MOCK_SSH_HANG: options.sshLookupHangs ? "1" : "0",
          MOCK_DRAIN_FAIL: options.drainFails ? "1" : "0",
          MOCK_OFFER_PRICE: options.offerPrice ?? "1",
          MOCK_MODE_UP: options.mode === "up" || options.mode === "up-gguf" ? "1" : "0",
          MOCK_SIBLING: options.sibling ? "1" : "0",
          MOCK_WRONG_LABEL: options.wrongLabel ? "1" : "0",
          MOCK_BACKEND_HEALTHY: options.backendHealthy ? "1" : "0",
          MOCK_CREATE_RENTAL: options.createRental ? "1" : "0",
          MOCK_SMOKE_FAIL: options.smokeFails ? "1" : "0",
          MOCK_UPLOAD_FAIL: options.uploadFails ? "1" : "0",
          MOCK_LAUNCHER_COPY_FAIL: options.launcherCopyFails ? "1" : "0",
          MOCK_TEMPLATE_COPY_FAIL: options.templateCopyFails ? "1" : "0",
          MOCK_LAUNCH_FAIL: options.launcherFails ? "1" : "0",
          MOCK_ENGINE_DIR: join(scratch, "local", "engine-builds", "1234567-sm120"),
          MOCK_ACTUAL_PRICE: options.actualPrice ?? "2.3",
          MOCK_PROVIDER: join(scratch, "provider"),
          MOCK_STOP_IGNORED: options.stopIgnored ? "1" : "0",
          MOCK_START_FAIL: options.startFails ? "1" : "0",
          MOCK_SCHEDULING: options.scheduling ? "1" : "0",
          MOCK_SSH_DOWN: options.sshDown ? "1" : "0",
          MOCK_REMOTE_ENGINE_STALE: options.remoteEngineStale ? "1" : "0",
          MOCK_UPLOADED: join(scratch, "uploaded"),
          MOCK_MODEL_ON_DISK: options.modelOnDisk ? "1" : "0",
        },
        ...(options.providerHangs || options.sshLookupHangs ? { timeout: 2500 } : {}),
      },
    );
    return {
      exitCode: result.exitCode,
      calls: existsSync(log) ? readFileSync(log, "utf8") : "",
      instanceRemains: existsSync(join(state, "id")),
      ...(options.sibling
        ? {
            nextInstance: existsSync(join(state, "id"))
              ? readFileSync(join(state, "id"), "utf8").trim()
              : null,
          }
        : {}),
      ...(options.mode === "up" || options.mode === "up-gguf"
        ? {
            stderr: result.stderr.toString(),
            clientPinned: existsSync(join(state, "vastai-client")),
          }
        : {}),
      ...(options.providerHangs || options.sshLookupHangs
        ? { outageRecorded: existsSync(join(state, "metrics-outage-since")) }
        : {}),
      ...(options.providerState ||
      options.paused ||
      options.mode === "pause" ||
      options.mode === "resume"
        ? {
            paused: existsSync(join(state, "paused")),
            leased: existsSync(join(state, "maintenance-until")),
            stderr: result.stderr.toString(),
          }
        : {}),
    };
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

const idleMetrics =
  "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:prompt_tokens_total 12\n";

const llamaMetrics =
  "# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total 12\n" +
  "# TYPE llamacpp:prompt_tokens_cached_total counter\nllamacpp:prompt_tokens_cached_total 0\n" +
  "# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total 0\n" +
  "# TYPE llamacpp:requests_processing gauge\nllamacpp:requests_processing 0\n" +
  "# TYPE llamacpp:requests_deferred gauge\nllamacpp:requests_deferred 0\n";

test("GGUF rental rejects an absent portable build before billing", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, { mode: "up-gguf" });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("");
  expect(result.stderr).toContain("Portable GLM engine unavailable");
});

test("GGUF rental arms cost control before billing a qualifying two-card offer", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
  });
  expect(result.exitCode).toBe(77);
  expect(result.calls).toBe("leased\n");
  expect(result.clientPinned).toBe(true);
});

test("GGUF rental verifies CUDA before starting model download", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    createRental: true,
  });
  expect(result.exitCode).toBe(0);
  expect(result.calls).toBe(
    "leased\nsmoke\n--user enable --now glm53-vast-tunnel.service glm53-vast-admission.service\n",
  );
  expect(result.instanceRemains).toBe(true);
});

test("GGUF rental uses its pinned artifact when the shared source branch advances", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    checkoutAhead: true,
  });
  expect(result.exitCode).toBe(77);
  expect(result.calls).toBe("leased\n");
});

test("GGUF rental refuses an invalid provider price after creation", () => {
  for (const actualPrice of ["4.01", "null", '"unknown"', "1e999", "-1"]) {
    const result = checkIdle(llamaMetrics, true, false, false, false, null, {
      mode: "up-gguf",
      portableEngine: true,
      createRental: true,
      actualPrice,
    });
    expect(result.exitCode).toBe(1);
    expect(result.calls).toBe("leased\ndestroy instance 123 --yes\n");
    expect(result.instanceRemains).toBe(false);
  }
});

test("GGUF engine upload failure destroys a new rental", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    createRental: true,
    uploadFails: true,
  });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("leased\ndestroy instance 123 --yes\n");
  expect(result.instanceRemains).toBe(false);
});

test("GGUF launcher copy failure destroys a new rental", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    createRental: true,
    launcherCopyFails: true,
  });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("leased\nsmoke\ndestroy instance 123 --yes\n");
  expect(result.instanceRemains).toBe(false);
});

test("GGUF template copy failure destroys a new rental", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    createRental: true,
    templateCopyFails: true,
  });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("leased\nsmoke\ndestroy instance 123 --yes\n");
  expect(result.instanceRemains).toBe(false);
});

test("GGUF launch failure destroys a new rental", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    createRental: true,
    launcherFails: true,
  });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("leased\nsmoke\ndestroy instance 123 --yes\n");
  expect(result.instanceRemains).toBe(false);
});

test("GGUF CUDA smoke failure destroys a new rental", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    createRental: true,
    smokeFails: true,
  });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("leased\nsmoke\ndestroy instance 123 --yes\n");
  expect(result.instanceRemains).toBe(false);
});

test("GGUF rental refuses a native-only engine before billing", () => {
  const result = checkIdle(llamaMetrics, true, false, false, false, null, {
    mode: "up-gguf",
    portableEngine: true,
    nativeEngine: true,
  });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("");
});

test("GGUF rental rejects wrong cards, stale drivers, and unaffordable offers", () => {
  for (const offer of [
    { offerGpuCount: 4 },
    { offerGpuName: "H200" },
    { offerCuda: 12.9 },
    { offerCuda: "13.3" },
    { offerPrice: "4.01" },
  ]) {
    const result = checkIdle(llamaMetrics, true, false, false, false, null, {
      mode: "up-gguf",
      portableEngine: true,
      ...offer,
    });
    expect(result.exitCode).toBe(1);
    expect(result.calls).toBe("");
  }
});

test("GGUF mode recognizes llama-server gauges and stops after ten idle minutes", () => {
  expect(checkIdle(llamaMetrics, true, false, false, false, null, { runtime: "gguf" })).toEqual({
    exitCode: 0,
    calls:
      "drain\nstop instance 123\n--user disable --now glm53-vast-tunnel.service glm53-vast-admission.service\n",
    instanceRemains: true,
  });
  expect(checkIdle(llamaMetrics).exitCode).toBe(1);
  expect(
    checkIdle(idleMetrics, true, false, false, false, null, { runtime: "gguf" }).exitCode,
  ).toBe(1);
});

test("GGUF mode preserves active, queued, newly generated, and newly cached work", () => {
  for (const metrics of [
    llamaMetrics.replace("requests_processing 0", "requests_processing 1"),
    llamaMetrics.replace("requests_deferred 0", "requests_deferred 1"),
    llamaMetrics.replace("tokens_predicted_total 0", "tokens_predicted_total 1"),
    llamaMetrics.replace("prompt_tokens_cached_total 0", "prompt_tokens_cached_total 1"),
  ]) {
    expect(checkIdle(metrics, true, false, false, false, null, { runtime: "gguf" }).calls).toBe("");
  }
  expect(
    checkIdle(llamaMetrics.replace("requests_deferred 0\n", ""), true, false, false, false, null, {
      runtime: "gguf",
    }).exitCode,
  ).toBe(1);
});

test("GGUF watchdog accepts labelled llama-server counters and gauges", () => {
  const labelled = llamaMetrics.replace(/^(llamacpp:[a-z_]+)(?= )/gm, '$1{server="glm53"}');
  expect(checkIdle(labelled, true, false, false, false, null, { runtime: "gguf" })).toEqual({
    exitCode: 0,
    calls:
      "drain\nstop instance 123\n--user disable --now glm53-vast-tunnel.service glm53-vast-admission.service\n",
    instanceRemains: true,
  });
});

test("GGUF mode refuses malformed gauges and late activity before teardown", () => {
  for (const value of ["NaN", "+Inf", "0.5", "-1"]) {
    expect(
      checkIdle(
        llamaMetrics.replace("requests_processing 0", `requests_processing ${value}`),
        true,
        false,
        false,
        false,
        null,
        {
          runtime: "gguf",
        },
      ).exitCode,
    ).toBe(1);
  }
  expect(
    checkIdle(
      llamaMetrics,
      true,
      false,
      false,
      false,
      llamaMetrics.replace("tokens_predicted_total 0", "tokens_predicted_total 1"),
      { runtime: "gguf" },
    ),
  ).toEqual({ exitCode: 0, calls: "drain\nresume\n", instanceRemains: true });
  expect(
    checkIdle(
      llamaMetrics.replace("requests_deferred 0", "requests_deferred 1"),
      true,
      false,
      false,
      false,
      null,
      {
        mode: "down",
        runtime: "gguf",
      },
    ).exitCode,
  ).toBe(1);
});

test("inventory failure cannot create a second billed rental", () => {
  const result = checkIdle(idleMetrics, true, false, false, false, null, {
    mode: "up",
    inventoryFails: true,
  });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("");
  expect(result.stderr).toContain("Could not inspect GLM rentals");
});

test("offers without a numeric hourly rate cannot be rented", () => {
  for (const offerPrice of ["null", '"unknown"', "1e999"]) {
    const result = checkIdle(idleMetrics, true, false, false, false, null, {
      mode: "up",
      offerPrice,
    });
    expect(result.exitCode).toBe(1);
    expect(result.calls).toBe("");
  }
});

test("rental is protected during the initial provisioning gap", () => {
  const result = checkIdle(idleMetrics, true, false, false, false, null, { mode: "up" });
  expect(result.exitCode).toBe(77);
  expect(result.calls).toBe("leased\n");
});

test("the chosen Vast client survives into scheduled watchdog checks", () => {
  const setup = checkIdle(idleMetrics, true, false, false, false, null, { mode: "up" });
  expect(setup.clientPinned).toBe(true);
  expect(
    checkIdle(idleMetrics, true, false, false, false, null, { persistedClient: true }),
  ).toEqual({
    exitCode: 0,
    calls:
      "drain\nstop instance 123\n--user disable --now glm53-vast-tunnel.service glm53-vast-admission.service\n",
    instanceRemains: true,
  });
});

test("manual down drains admission and destroys an idle rental", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { mode: "down" })).toEqual({
    exitCode: 0,
    calls: "drain\ndestroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

test("manual down refuses to interrupt an active request", () => {
  expect(
    checkIdle(idleMetrics.replace("running 0", "running 1"), true, false, false, false, null, {
      mode: "down",
    }),
  ).toEqual({ exitCode: 1, calls: "", instanceRemains: true });
});

test("GLM rental stays up until ten idle minutes have elapsed", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { idleAgeSeconds: 590 })).toEqual({
    exitCode: 0,
    calls: "",
    instanceRemains: true,
  });
});

test("GLM rental is stopped, not destroyed, after ten idle minutes", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { idleAgeSeconds: 601 })).toEqual({
    exitCode: 0,
    calls:
      "drain\nstop instance 123\n--user disable --now glm53-vast-tunnel.service glm53-vast-admission.service\n",
    instanceRemains: true,
  });
});

test("idle GLM rental is paused with its disk kept after the timeout", () => {
  const result = checkIdle(idleMetrics, true, false, false, false, null, {
    providerState: "running",
  });
  expect(result.exitCode).toBe(0);
  expect(result.calls).toBe(
    "drain\nstop instance 123\n--user disable --now glm53-vast-tunnel.service glm53-vast-admission.service\n",
  );
  expect(result.instanceRemains).toBe(true);
  expect(result.paused).toBe(true);
  expect(result.leased).toBe(false);
});

test("a stale local ID cannot destroy another account's rental", () => {
  const result = checkIdle(idleMetrics, true, false, false, false, null, { wrongLabel: true });
  expect(result.exitCode).toBe(1);
  expect(result.calls).toBe("");
  expect(result.instanceRemains).toBe(true);
});

test("another billed GLM rental remains monitored after teardown", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { sibling: true })).toEqual({
    exitCode: 0,
    calls:
      "drain\ndestroy instance 123 --yes\n--user restart glm53-vast-tunnel.service glm53-vast-admission.service\n",
    instanceRemains: true,
    nextInstance: "124",
  });
});

test("unavailable admission control fails closed instead of destroying", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { drainFails: true })).toEqual({
    exitCode: 1,
    calls: "resume\n",
    instanceRemains: true,
  });
});

test("late request arrival cancels idle teardown", () => {
  expect(
    checkIdle(
      idleMetrics,
      true,
      false,
      false,
      false,
      idleMetrics.replace("running 0", "running 1"),
    ),
  ).toEqual({
    exitCode: 0,
    calls: "drain\nresume\n",
    instanceRemains: true,
  });
});

test("download time is excluded from the idle budget", () => {
  expect(checkIdle(idleMetrics, false)).toEqual({
    exitCode: 0,
    calls: "",
    instanceRemains: true,
  });
});

test("active requests and changed counters prevent teardown", () => {
  expect(checkIdle(idleMetrics.replace("running 0", "running 1")).calls).toBe("");
  expect(checkIdle(idleMetrics.replace("total 12", "total 13")).calls).toBe("");
  expect(checkIdle(idleMetrics.replace("prompt_tokens_total 12", "prompt_tokens 13")).calls).toBe(
    "",
  );
});

test("missing counters cannot be mistaken for idle", () => {
  expect(checkIdle("vllm:num_requests_running 0\n")).toEqual({
    exitCode: 1,
    calls: "",
    instanceRemains: true,
  });
});

test("invalid activity values cannot be mistaken for idle", () => {
  for (const value of ["NaN", "+Inf", "not-a-number"]) {
    const result = checkIdle(idleMetrics.replace("running 0", `running ${value}`));
    expect(result.exitCode).toBe(1);
    expect(result.calls).toBe("");
    expect(result.instanceRemains).toBe(true);
  }
});

test("bounded maintenance preserves a restarting rental", () => {
  expect(checkIdle(null, false, true, false, false, null, { maintenance: "active" })).toEqual({
    exitCode: 0,
    calls: "",
    instanceRemains: true,
  });
});

test("expired maintenance cannot disable failed-startup teardown", () => {
  expect(checkIdle(null, false, true, false, false, null, { maintenance: "expired" })).toEqual({
    exitCode: 1,
    calls: "destroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

test("stale failure marker cannot destroy a healthy restarted server", () => {
  expect(checkIdle(idleMetrics, false, true)).toEqual({
    exitCode: 0,
    calls: "",
    instanceRemains: true,
  });
});

test("failed startup destroys the rental after preserving failure logs", () => {
  expect(checkIdle(null, false, true)).toEqual({
    exitCode: 1,
    calls: "destroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

test("aborted destroy restores forwarding and keeps the watchdog armed", () => {
  expect(checkIdle(idleMetrics, true, false, true, false, null, { mode: "off" })).toEqual({
    exitCode: 1,
    calls: "drain\nresume\n",
    instanceRemains: true,
  });
});

test("missing waiting gauge cannot erase queued requests", () => {
  expect(checkIdle(idleMetrics.replace("vllm:num_requests_waiting 0\n", ""))).toEqual({
    exitCode: 1,
    calls: "",
    instanceRemains: true,
  });
});

test("crashed bootstrap destroys the billed rental", () => {
  expect(checkIdle(null, false, false, false, true)).toEqual({
    exitCode: 1,
    calls: "destroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

test("remote metrics still trigger idle teardown when the tunnel is broken", () => {
  expect(
    checkIdle(null, true, false, false, false, null, {
      remoteHealthy: true,
      remoteMetrics: idleMetrics,
    }),
  ).toEqual({
    exitCode: 0,
    calls:
      "drain\nstop instance 123\n--user disable --now glm53-vast-tunnel.service glm53-vast-admission.service\n",
    instanceRemains: true,
  });
});

test("healthy tunnel backend repairs only the admission proxy", () => {
  expect(
    checkIdle(null, true, false, false, false, null, {
      remoteHealthy: true,
      remoteMetrics: idleMetrics,
      recentlyActive: true,
      backendHealthy: true,
    }),
  ).toEqual({
    exitCode: 0,
    calls: "--user restart glm53-vast-admission.service\n",
    instanceRemains: true,
  });
});

test("active remote requests survive a broken tunnel", () => {
  expect(
    checkIdle(null, true, false, false, false, null, {
      remoteHealthy: true,
      remoteMetrics: idleMetrics.replace("running 0", "running 1"),
    }),
  ).toEqual({ exitCode: 0, calls: "", instanceRemains: true });
});

test("active bootstrap stays rented within its startup budget", () => {
  expect(
    checkIdle(null, false, false, false, true, null, {
      bootstrapAlive: true,
      startedRecently: true,
    }),
  ).toEqual({ exitCode: 1, calls: "", instanceRemains: true });
});

test("running model survives a transient metrics failure after startup budget", () => {
  expect(checkIdle(null, true, false, false, true, null, { bootstrapAlive: true })).toEqual({
    exitCode: 1,
    calls: "",
    instanceRemains: true,
  });
});

test("stalled bootstrap is torn down after its startup budget", () => {
  expect(checkIdle(null, false, false, false, true, null, { bootstrapAlive: true })).toEqual({
    exitCode: 1,
    calls: "destroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

test("an unhealthy running server cannot bill forever without metrics", () => {
  expect(
    checkIdle(null, true, false, false, true, null, {
      bootstrapAlive: true,
      outageSeconds: 7300,
    }),
  ).toEqual({ exitCode: 1, calls: "destroy instance 123 --yes\n", instanceRemains: false });
});

test("an unreachable rental is retired after a bounded metrics outage", () => {
  expect(checkIdle(null, true, false, false, false, null, { outageSeconds: 7300 })).toEqual({
    exitCode: 1,
    calls: "destroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

test("hung Vast status lookup does not stall outage accounting", () => {
  expect(checkIdle(null, true, false, false, false, null, { providerHangs: true })).toEqual({
    exitCode: 1,
    calls: "",
    instanceRemains: true,
    outageRecorded: true,
  });
});

test("hung SSH lookup does not stall outage accounting", () => {
  expect(checkIdle(null, true, false, false, false, null, { sshLookupHangs: true })).toEqual({
    exitCode: 1,
    calls: "",
    instanceRemains: true,
    outageRecorded: true,
  });
});

test("an unreachable server cannot be mistaken for idle", () => {
  expect(checkIdle(null)).toEqual({ exitCode: 1, calls: "", instanceRemains: true });
});

const disableServices =
  "--user disable --now glm53-vast-tunnel.service glm53-vast-admission.service\n";
const enableServices =
  "--user enable --now glm53-vast-tunnel.service glm53-vast-admission.service\n";

test("a rental Vast will not stop keeps its watchdog and reopens admission", () => {
  const result = checkIdle(idleMetrics, true, false, false, false, null, {
    providerState: "running",
    stopIgnored: true,
  });
  expect(result).toMatchObject({
    exitCode: 1,
    calls: "drain\nstop instance 123\nresume\n",
    instanceRemains: true,
    paused: false,
  });
});

test("a stopped rental keeps its disk instead of being destroyed", () => {
  expect(
    checkIdle(null, true, false, false, false, null, { providerState: "stopped" }),
  ).toMatchObject({
    exitCode: 0,
    calls: disableServices,
    instanceRemains: true,
    paused: true,
  });
});

test("a paused rental is left alone", () => {
  expect(
    checkIdle(null, false, false, false, false, null, { providerState: "stopped", paused: true }),
  ).toMatchObject({ exitCode: 0, calls: "", instanceRemains: true, paused: true });
});

test("a start left waiting for GPUs outside resume is cancelled with the disk kept", () => {
  expect(
    checkIdle(null, false, false, false, false, null, {
      providerState: "scheduling",
      paused: true,
    }),
  ).toMatchObject({
    exitCode: 0,
    calls: `stop instance 123\n${disableServices}`,
    instanceRemains: true,
    paused: true,
  });
});

test("resume's own wait for GPUs is not cancelled by the watchdog", () => {
  expect(
    checkIdle(null, false, false, false, false, null, {
      providerState: "scheduling",
      paused: true,
      maintenance: "active",
    }),
  ).toMatchObject({ exitCode: 0, calls: "", paused: true });
});

test("a paused rental started outside resume is stopped again when it serves nothing", () => {
  expect(
    checkIdle(null, false, false, false, false, null, { providerState: "running", paused: true }),
  ).toMatchObject({
    exitCode: 0,
    calls: `stop instance 123\n${disableServices}`,
    instanceRemains: true,
    paused: true,
  });
});

test("a paused rental serving work outside resume is adopted, not stopped", () => {
  expect(
    checkIdle(null, false, false, false, false, null, {
      providerState: "running",
      paused: true,
      remoteHealthy: true,
      remoteMetrics: idleMetrics.replace("running 0", "running 1"),
    }),
  ).toMatchObject({ exitCode: 0, calls: enableServices, instanceRemains: true, paused: false });
});

test("manual pause drains admission and stops an idle rental with its disk kept", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { mode: "pause" })).toMatchObject({
    exitCode: 0,
    calls: `drain\nstop instance 123\n${disableServices}`,
    instanceRemains: true,
    paused: true,
  });
});

test("manual pause refuses to interrupt an active request", () => {
  expect(
    checkIdle(idleMetrics.replace("running 0", "running 1"), true, false, false, false, null, {
      mode: "pause",
    }),
  ).toMatchObject({ exitCode: 1, calls: "", instanceRemains: true, paused: false });
});

test("off destroys a paused rental, which has no server to drain", () => {
  expect(
    checkIdle(null, false, false, false, false, null, {
      mode: "off",
      providerState: "stopped",
      paused: true,
    }),
  ).toMatchObject({
    exitCode: 0,
    calls: "destroy instance 123 --yes\n",
    instanceRemains: false,
    paused: false,
  });
});

test("off drains and destroys a running idle rental", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { mode: "off" })).toEqual({
    exitCode: 0,
    calls: "drain\ndestroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

const resumeGguf = {
  mode: "resume",
  providerState: "stopped",
  paused: true,
  runtime: "gguf",
  portableEngine: true,
} as const;

test("resume starts the paused rental and relaunches GLM on its kept disk", () => {
  expect(checkIdle(null, false, false, false, false, null, resumeGguf)).toMatchObject({
    exitCode: 0,
    calls: `start instance 123\nlaunch\n${enableServices}`,
    instanceRemains: true,
    paused: false,
    leased: false,
  });
});

test("resume uploads the pinned engine when the rental's copy differs", () => {
  expect(
    checkIdle(null, false, false, false, false, null, { ...resumeGguf, remoteEngineStale: true }),
  ).toMatchObject({ exitCode: 0, calls: `start instance 123\nupload\nlaunch\n${enableServices}` });
});

test("resume bounds the wait for GPUs, then stops again and names the guarded replacement", () => {
  const result = checkIdle(null, false, false, false, false, null, {
    ...resumeGguf,
    scheduling: true,
  });
  expect(result).toMatchObject({
    exitCode: 1,
    calls: `start instance 123\nstop instance 123\n${disableServices}`,
    instanceRemains: true,
    paused: true,
    leased: false,
  });
  expect(result.stderr).toContain("glm53-vast.sh up-gguf OFFER_ID");
});

test("resume that Vast refuses leaves the rental paused", () => {
  expect(
    checkIdle(null, false, false, false, false, null, { ...resumeGguf, startFails: true }),
  ).toMatchObject({
    exitCode: 1,
    calls: `start instance 123\nstop instance 123\n${disableServices}`,
    paused: true,
    leased: false,
  });
});

test("resume without SSH stops the rental again instead of billing its GPUs", () => {
  expect(
    checkIdle(null, false, false, false, false, null, { ...resumeGguf, sshDown: true }),
  ).toMatchObject({
    exitCode: 1,
    calls: `start instance 123\nstop instance 123\n${disableServices}`,
    paused: true,
  });
});

test("resume launch failure stops the rental again with its disk kept", () => {
  expect(
    checkIdle(null, false, false, false, false, null, { ...resumeGguf, launcherFails: true }),
  ).toMatchObject({
    exitCode: 1,
    calls: `start instance 123\nstop instance 123\n${disableServices}`,
    instanceRemains: true,
    paused: true,
  });
});

test("resume leaves a running rental alone and never creates one", () => {
  expect(
    checkIdle(idleMetrics, true, false, false, false, null, {
      mode: "resume",
      providerState: "running",
    }),
  ).toMatchObject({ exitCode: 0, calls: "", instanceRemains: true, paused: false });
});

test("resume sends no credential when the model is already on disk", () => {
  expect(
    checkIdle(null, false, false, false, false, null, {
      mode: "resume",
      providerState: "stopped",
      paused: true,
      modelOnDisk: true,
    }),
  ).toMatchObject({ exitCode: 0, calls: `start instance 123\nlaunch\n${enableServices}` });
});

test("resume sends the download credential only for an incomplete model", () => {
  expect(
    checkIdle(null, false, false, false, false, null, {
      mode: "resume",
      providerState: "stopped",
      paused: true,
    }),
  ).toMatchObject({ exitCode: 0, calls: `start instance 123\ntoken\nlaunch\n${enableServices}` });
});
