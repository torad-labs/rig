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
    mode?: "up" | "down";
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
  if (options.mode !== "up") writeFileSync(join(state, "id"), "123\n");
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
  writeFileSync(
    join(bin, "vastai"),
    '#!/bin/bash\nif [[ $1 == --raw && $2 == show && $3 == instances ]]; then if [[ $MOCK_INVENTORY_FAIL == 1 ]]; then exit 7; fi; if [[ $MOCK_MODE_UP == 1 ]]; then printf "[]\\n"; elif [[ $MOCK_REJECT != 1 ]] && grep -q "^destroy instance " "$MOCK_LOG" 2>/dev/null; then if [[ $MOCK_SIBLING == 1 ]]; then printf "[{\\"id\\":124,\\"label\\":\\"rig-glm53\\",\\"cur_state\\":\\"running\\"}]\\n"; else printf "[]\\n"; fi; elif [[ $MOCK_WRONG_LABEL == 1 ]]; then printf "[{\\"id\\":123,\\"label\\":\\"other\\",\\"cur_state\\":\\"running\\"}]\\n"; elif [[ $MOCK_SIBLING == 1 ]]; then printf "[{\\"id\\":123,\\"label\\":\\"rig-glm53\\",\\"cur_state\\":\\"running\\"},{\\"id\\":124,\\"label\\":\\"rig-glm53\\",\\"cur_state\\":\\"running\\"}]\\n"; else printf "[{\\"id\\":123,\\"label\\":\\"rig-glm53\\",\\"cur_state\\":\\"running\\"}]\\n"; fi\nelif [[ $1 == --raw && $2 == search ]]; then printf "[{\\"id\\":123,\\"dph_total\\":%s}]\\n" "$MOCK_OFFER_PRICE"\nelif [[ $1 == --raw && $2 == create ]]; then if [[ -s "$MOCK_STATE/maintenance-until" ]]; then printf "leased\\n"; else printf "unleased\\n"; fi >> "$MOCK_LOG"; exit 77\nelif [[ $1 == --raw && $2 == show && $3 == instance ]]; then if [[ $MOCK_PROVIDER_HANG == 1 ]]; then sleep 5; fi; printf "{\\"cur_state\\":\\"running\\"}\\n"\nelif [[ $1 == --raw ]]; then printf "{\\"cur_state\\":\\"running\\"}\\n"\nelif [[ $1 == ssh-url ]]; then if [[ $MOCK_SSH_HANG == 1 ]]; then sleep 5; fi; printf "ssh://root@127.0.0.1:22\\n"\nelif [[ $1 == destroy ]]; then if [[ $* == *--yes* && $MOCK_REJECT != 1 ]]; then printf "%s\\n" "$*" >> "$MOCK_LOG"; else printf "Aborted.\\n"; fi; fi\n',
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
  writeFileSync(join(bin, "hf"), "#!/bin/bash\nexit 0\n");
  writeFileSync(
    join(bin, "systemctl"),
    '#!/bin/bash\nif [[ $* == *"restart glm53-vast-"* ]]; then printf "%s\\n" "$*" >> "$MOCK_LOG"; fi\nexit 0\n',
  );
  for (const command of ["vastai", "ssh", "curl", "timeout", "hf", "systemctl"])
    chmodSync(join(bin, command), 0o755);

  try {
    const result = Bun.spawnSync(
      options.mode === "up"
        ? [script, "up", "123"]
        : [script, options.mode === "down" ? "down" : "idle-check"],
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
          MOCK_MODE_UP: options.mode === "up" ? "1" : "0",
          MOCK_SIBLING: options.sibling ? "1" : "0",
          MOCK_WRONG_LABEL: options.wrongLabel ? "1" : "0",
          MOCK_BACKEND_HEALTHY: options.backendHealthy ? "1" : "0",
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
      ...(options.mode === "up"
        ? {
            stderr: result.stderr.toString(),
            clientPinned: existsSync(join(state, "vastai-client")),
          }
        : {}),
      ...(options.providerHangs || options.sshLookupHangs
        ? { outageRecorded: existsSync(join(state, "metrics-outage-since")) }
        : {}),
    };
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

const idleMetrics =
  "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:prompt_tokens_total 12\n";

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
    calls: "drain\ndestroy instance 123 --yes\n",
    instanceRemains: false,
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

test("GLM rental is destroyed after ten idle minutes", () => {
  expect(checkIdle(idleMetrics, true, false, false, false, null, { idleAgeSeconds: 601 })).toEqual({
    exitCode: 0,
    calls: "drain\ndestroy instance 123 --yes\n",
    instanceRemains: false,
  });
});

test("idle GLM rental is destroyed after the timeout", () => {
  expect(checkIdle(idleMetrics)).toEqual({
    exitCode: 0,
    calls: "drain\ndestroy instance 123 --yes\n",
    instanceRemains: false,
  });
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
  expect(checkIdle(idleMetrics, true, false, true)).toEqual({
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
  ).toEqual({ exitCode: 0, calls: "drain\ndestroy instance 123 --yes\n", instanceRemains: false });
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
