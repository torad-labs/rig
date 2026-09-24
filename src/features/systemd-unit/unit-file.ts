// The systemd user unit for one head, rendered from the plan: the full llama-server command line
// as ExecStart (no supervisor between systemd and the server — signals, restarts and the main PID
// are the server's), the card and library path as Environment, and `rig verify` as ExecStartPre
// so a unit never starts on a pack or build that is not the pinned one. Rendered, never edited:
// head.toml and the machine are the inputs, and re-running install re-renders.
import type { Head } from "../../shared/head/head.ts";

export interface UnitInputs {
  head: Head;
  root: string;
  logPath: string;
  argv: readonly string[];
  env: Readonly<Record<string, string>>;
  gpu: number;
  /** how to invoke rig itself, as argv (the compiled binary, or bun + main.ts in a checkout) */
  self: readonly string[];
}

export const unitName = (head: string) => `rig-${head}.service`;

/** Shield the server from the box's memory killer, on the RUNNING pid.
 *
 *  A head is the biggest anonymous allocation on a workstation — this one's prompt cache alone
 *  is --cache-ram MiB of it — so a reaper that picks the largest unprotected process picks the
 *  head, and every seat loses its runtime (measured 2026-09-20 03:28:38: SIGTERM at 13,379 MiB
 *  while a gate loaded a second pack on the other card). `OOMScoreAdjust=` here would be worse
 *  than nothing: a --user manager may raise but never lower, so systemd clamps it to the
 *  inherited +200 and exits clean, and the unit READS as protected while the process is not.
 *  The pin therefore goes through the privileged choom grant on $MAINPID, and `rig unit status`
 *  reports /proc/<pid>/oom_score_adj, never the unit property. A box without the grant starts
 *  exactly as before: "-" makes the failure non-fatal. */
const OOM_SHIELD = "ExecStartPost=-/usr/bin/sudo -n /usr/bin/choom -p $MAINPID -n -800";

export function renderUnit(inputs: UnitInputs): string {
  const env = Object.entries(inputs.env)
    .map(([name, value]) => `Environment=${quote(`${name}=${value}`)}`)
    .join("\n");
  // the exact -m path this render put in ExecStart, so ExecStartPre checks THAT file (whichever
  // pin it names) rather than re-resolving the head fresh: a re-resolution can disagree with what
  // -m already fixed, in either direction (unit-file.ts's own header comment above)
  const pack = packPathOf(inputs.argv);
  const verify = [...inputs.self, "verify", inputs.head.name, "--gpu", String(inputs.gpu)];
  if (pack !== undefined) verify.push("--pack", pack);
  return `# ${inputs.head.title} on llama-server :${inputs.head.port} — rendered by \`rig unit install ${inputs.head.name}\` from
# heads/${inputs.head.name}/head.toml and this machine (card, RAM). Edit the head or re-run install; never this copy.
[Unit]
Description=${inputs.head.title} llama-server, GPU ${inputs.gpu} (rig head ${inputs.head.name} :${inputs.head.port})
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=${inputs.root}
${env}
ExecStartPre=${verify.map(quote).join(" ")}
ExecStart=${inputs.argv.map(quote).join(" ")}
${OOM_SHIELD}
StandardOutput=append:${inputs.logPath}
StandardError=append:${inputs.logPath}
Restart=on-failure
RestartSec=5
RestartSteps=6
RestartMaxDelaySec=120

[Install]
WantedBy=default.target
`;
}

/** the `-m` value out of a rendered argv, when it carries one */
function packPathOf(argv: readonly string[]): string | undefined {
  const i = argv.indexOf("-m");
  return i >= 0 ? argv[i + 1] : undefined;
}

/** systemd's command-line quoting: double quotes around anything with whitespace or quotes. */
function quote(text: string): string {
  const needsQuotes = /[\s"'\\]/.test(text);
  return needsQuotes ? `"${text.replace(/[\\"]/g, (char) => `\\${char}`)}"` : text;
}

/** the --cache-ram a rendered unit carries, so a re-render on a shared host keeps the host's bound */
export function cacheRamOf(unitText: string): number | undefined {
  const match = /^ExecStart=.*\s--cache-ram\s+(\d+)(?:\s|$)/m.exec(unitText);
  return match ? Number(match[1]) : undefined;
}
