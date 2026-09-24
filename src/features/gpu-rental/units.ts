// The two user units a rented box needs on THIS machine: the ssh tunnel (the only path to the
// box's server) and the idle timer (the cost control). Rendered here, installed by the use case.
export const TUNNEL_UNIT = "rig-vast-tunnel.service";
export const IDLE_SERVICE = "rig-vast-idle.service";
export const IDLE_TIMER = "rig-vast-idle.timer";

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

export function renderIdleService(o: { self: readonly string[]; idleMinutes: number }): string {
  return `# One idle check; ${IDLE_TIMER} runs it every 10 minutes while a box exists.
[Unit]
Description=Destroy the rented box after ${o.idleMinutes} idle minutes

[Service]
Type=oneshot
ExecStart=${[...o.self, "vast", "idle-check"].join(" ")}
`;
}

export function renderIdleTimer(): string {
  return `# The cost control: a box that nobody is talking to is destroyed, not left billing overnight.
# Started by \`rig vast up\`, stopped by \`rig vast down\`.
[Unit]
Description=Idle check for the rented box, every 10 minutes

[Timer]
OnActiveSec=10min
OnUnitActiveSec=10min
AccuracySec=1min

[Install]
WantedBy=timers.target
`;
}
