// The user units a rented box needs on THIS machine, each box its own, named by its instance: the ssh tunnel (the only
// path to the box's server), the idle timer (the cost control) and the hard stop (the box destroyed at a fixed time
// whatever it reads). Rendered here, installed by the use case.
export const SWEEP_SERVICE = "rig-vast-sweep.service";

/** one box's units */
export interface BoxUnits {
  idleService: string;
  idleTimer: string;
  tunnelUnit: string;
  /** null for a legacy box, which was created without one */
  stopService: string | null;
  stopTimer: string | null;
}

export function boxUnits(instanceId: number): BoxUnits {
  return {
    idleService: `rig-vast-idle-${instanceId}.service`,
    idleTimer: `rig-vast-idle-${instanceId}.timer`,
    tunnelUnit: `rig-vast-tunnel-${instanceId}.service`,
    stopService: `rig-vast-stop-${instanceId}.service`,
    stopTimer: `rig-vast-stop-${instanceId}.timer`,
  };
}

/** the units of a rig that held one box, named for none (box-state.ts: a legacy box) */
export const LEGACY_UNITS: BoxUnits = {
  idleService: "rig-vast-idle.service",
  idleTimer: "rig-vast-idle.timer",
  tunnelUnit: "rig-vast-tunnel.service",
  stopService: null,
  stopTimer: null,
};
export const SWEEP_TIMER = "rig-vast-sweep.timer";

export function renderTunnelUnit(o: {
  tunnelEnv: string;
  knownHosts: string;
  localPort: number;
  remotePort: number;
}): string {
  return `# SSH local forward: 127.0.0.1:${o.localPort} here -> 127.0.0.1:${o.remotePort} on the rented box. This is the
# ONLY path to the box's llama-server (it binds loopback there; no vast port is mapped), which is
# why a proxy can point a head at a loopback URL. HOST/PORT come from ${o.tunnelEnv}, written by
# \`rig vast up\`; \`rig vast down\` stops this unit. Not enabled at boot on purpose: the box it
# points at may be gone.
[Unit]
Description=SSH tunnel to the rented llama-server (127.0.0.1:${o.localPort} -> box :${o.remotePort})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=${o.tunnelEnv}
ExecStart=/usr/bin/ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=${o.knownHosts} -o LogLevel=ERROR -L 127.0.0.1:${o.localPort}:127.0.0.1:${o.remotePort} -p \${PORT} root@\${HOST}
Restart=always
RestartSec=3
`;
}

/** How often the idle check reads the box. The card is a point sample, so the gap between two reads is a gap in what the
 *  reaper can see: a run shorter than it can fall wholly inside. Three minutes is the gap a run of a few minutes cannot
 *  hide in, at a few seconds of ssh per read. */
export const IDLE_CHECK_MINUTES = 3;

export function renderIdleService(o: {
  self: readonly string[];
  idleMinutes: number;
  instanceId: number;
}): string {
  return `# One idle check of box ${o.instanceId}; ${boxUnits(o.instanceId).idleTimer} runs it every ${IDLE_CHECK_MINUTES} minutes while the box exists.
[Unit]
Description=Destroy rented box ${o.instanceId} after ${o.idleMinutes} idle minutes

[Service]
Type=oneshot
ExecStart=${[...o.self, "vast", "idle-check", "--box", String(o.instanceId)].join(" ")}
`;
}

export function renderIdleTimer(instanceId: number): string {
  return `# The cost control: a box that nobody is talking to is destroyed, not left billing overnight.
# Enabled and started when the box is created (so a user-manager restart or a reboot arms it again while
# the box bills), re-armed by \`rig vast status\` if found dead, disabled and removed by \`rig vast down\`.
[Unit]
Description=Idle check for rented box ${instanceId}, every ${IDLE_CHECK_MINUTES} minutes

[Timer]
OnActiveSec=${IDLE_CHECK_MINUTES}min
OnUnitActiveSec=${IDLE_CHECK_MINUTES}min
AccuracySec=1min

[Install]
WantedBy=timers.target
`;
}

/** The box destroyed at a fixed time whatever its readings: an idle check reads a busy card as work, and a card a hung
 *  process holds at 100 % reads busy for ever. Rental 2c's stop was written by hand as these two units (FREEZE-RUNBOOK
 *  section 6); every box now gets them when it is created.
 *  The hard stop is the backstop for the code under change, so only its first line runs rig (rig-orchestrator,
 *  2026-10-03): "-" lets the next lines run whatever rig's exit, a tree that does not load included. vastai's destroy
 *  exits 0 when vast refuses it (vastai/cli/main.py prints an HTTP error and returns), so the last line is what fails a
 *  stop that did not take: it exits 0 only once vast no longer lists the box, and Restart= runs the stop again. */
export function renderStopService(o: {
  self: readonly string[];
  /** the vast CLI by its absolute path: systemd's search path has no ~/.local/bin */
  vastai: string;
  instanceId: number;
}): string {
  const id = o.instanceId;
  return `# The hard stop of box ${id}: ${boxUnits(id).stopTimer} runs it once, at the box's last hour.
[Unit]
Description=Destroy rented box ${id} at its hard stop

[Service]
Type=oneshot
# rig destroys the box and retires its units; "-" runs the next lines whatever rig's exit
ExecStart=-${[...o.self, "vast", "down", "--box", String(id)].join(" ")}
# no rig code from here: vast destroys the box (on a box already gone it prints vast's refusal and exits 0) ...
ExecStart=${o.vastai} destroy instance ${id} -y
# ... and the stop has taken only once vast reads every page and no longer lists the box; listed, or vast unread, it is run again
ExecStart=python3 -c "${goneCheck(o.vastai, id)}"
Restart=on-failure
RestartSec=300
`;
}

/** the pages a listing is read to, 25 boxes a page: a next page still named past them is the listing looping */
const LISTING_PAGES = 100;

/** Python that exits 0 only when `vastai show instances-v1` reads every page and does not list the box. Each page is
 *  asked for by the token the one before named, until one names none: under --raw the CLI returns one page whatever -a
 *  says (vastai 1.0.12, cli/commands/instances.py:1270), as the adapter reads it (vast-ai-rental.ts). The JSON is read
 *  from its first bracket. A page that is not instances-v1's object (an error, the old command's bare array, nothing)
 *  is a listing not read. Written with no double quote, backslash, "%" or "$", which systemd would read in an
 *  ExecStart= line, and with no compound statement, so the loop is a comprehension over the pages read so far. */
export function goneCheck(vastai: string, id: number): string {
  return [
    "import json,subprocess,sys",
    `run=lambda token: subprocess.run(['${vastai}','show','instances-v1']+(['--next-token',token] if token else [])+['--raw'],capture_output=True,text=True,timeout=120).stdout`,
    "parse=lambda out: json.loads(out[min(i for i in (out.find('['),out.find('{')) if i>=0):]) if ('[' in out or '{' in out) else None",
    "pages=[]",
    `[pages.append(parse(run(pages[-1].get('next_token') if pages else None))) for page in range(${LISTING_PAGES}) if not pages or (isinstance(pages[-1],dict) and pages[-1].get('next_token'))]`,
    "read=bool(pages) and all(isinstance(p,dict) and isinstance(p.get('instances'),list) for p in pages) and not pages[-1].get('next_token')",
    `held=read and any(isinstance(r,dict) and r.get('id')==${id} for p in pages for r in p['instances'])`,
    `print('vast lists box ${id}: stop again' if held else 'vast no longer lists box ${id}' if read else 'vast not read: stop again')`,
    "sys.exit(0 if read and not held else 1)",
  ].join(";");
}

export function renderStopTimer(o: { instanceId: number; at: number }): string {
  const when = `${new Date(Math.ceil(o.at / 1000) * 1000).toISOString().slice(0, 19).replace("T", " ")} UTC`;
  return `# The hard stop of box ${o.instanceId}, armed when it was created. Persistent: a machine that was off or asleep at
# the time runs it as soon as it is back. Disabled and removed by \`rig vast down\`.
[Unit]
Description=Hard stop for rented box ${o.instanceId}, at ${when}

[Timer]
OnCalendar=${when}
Persistent=true
AccuracySec=1min

[Install]
WantedBy=timers.target
`;
}

export function renderSweepService(o: { self: readonly string[] }): string {
  return `# One sweep; ${SWEEP_TIMER} runs it every hour.
[Unit]
Description=Destroy rig's vast boxes that have been stopped for vast.toml's stopped_hours

[Service]
Type=oneshot
ExecStart=${[...o.self, "vast", "sweep"].join(" ")}
`;
}

export function renderSweepTimer(): string {
  return `# The cost control for rig's boxes that no idle timer here watches (one rented from a template in vast's console,
# stopped by its own guard): a stopped box bills its disk until it is destroyed. Enabled by \`rig vast template\`.
[Unit]
Description=Sweep rig's stopped vast boxes, every hour

[Timer]
OnCalendar=hourly
Persistent=true
RandomizedDelaySec=5min

[Install]
WantedBy=timers.target
`;
}
