// What runs on a rented box for the idle check, as shell text: the sampler that keeps one line every few seconds, and the
// read of its window. The text is the contract with the box, so it is built here from named pieces and a test runs the
// pieces under sh against fixtures: a fake ssh cannot tell an awk program that parses from one that does not (the
// first draft of the download column summed packets for every interface whose name is over five characters long, and
// printed a newline inside an awk string).
export interface SamplerPaths {
  script: string;
  pid: string;
  /** the first generation's pid file, which wrote no download column: read once to retire it */
  firstPid: string;
  samples: string;
}

/** /var/tmp, which a container and a VM both have */
export const SAMPLER_PATHS: SamplerPaths = {
  script: "/var/tmp/rig-card-sampler.sh",
  pid: "/var/tmp/rig-card-sampler.v2.pid",
  firstPid: "/var/tmp/rig-card-sampler.pid",
  samples: "/var/tmp/rig-card.samples",
};

/** total bytes received on every interface but loopback. /proc/net/dev pads a short interface name ("  eth0:") and not
 *  a long one ("enp5s0:", "br-0a1b2c3d4e5f:"), so the bytes are located after the colon, never by field position. */
export const rxBytes = (source = "/proc/net/dev"): string =>
  `awk '/:/ { name = $0; sub(/:.*/, "", name); gsub(/ /, "", name); rest = $0; sub(/^[^:]*:/, "", rest); split(rest, f, " "); if (name != "lo") s += f[1] } END { printf "%.0f", s }' ${source} 2>/dev/null`;

/** the script a box runs for ever: "epoch percent KB/s" a line, one per tick, the last hour kept. The percent is the
 *  busiest card's; the KB/s is what the box received over the tick, the pack being pulled, an image pushed to it, a
 *  clone, all of which leave the card and the server idle. */
export function samplerScript(paths: SamplerPaths = SAMPLER_PATHS, tickSeconds = 5): string[] {
  return [
    `echo $$ > ${paths.pid}`,
    "n=0; p=; pt=",
    "while :; do",
    "  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)",
    "  t=$(date +%s)",
    `  r=$(${rxBytes()})`,
    "  k=0",
    '  if [ -n "$p" ] && [ -n "$r" ] && [ "$t" -gt "$pt" ] && [ "$r" -ge "$p" ]; then k=$(( (r - p) / 1024 / (t - pt) )); fi',
    "  p=$r; pt=$t",
    `  [ -n "$u" ] && echo "$t $u $k" >> ${paths.samples}`,
    "  n=$((n + 1))",
    `  if [ $((n % 720)) -eq 0 ]; then tail -n 720 ${paths.samples} > ${paths.samples}.t && mv ${paths.samples}.t ${paths.samples}; fi`,
    `  sleep ${tickSeconds}`,
    "done",
  ];
}

/** the samples of the last `windowSeconds`: `window=` the busiest card's peak, `download=` the average KB/s received
 *  (printed only when a sample in the window carries the column: the first generation's lines do not) */
export function windowRead(samples: string, windowSeconds: number): string {
  return `awk -v since=$(( $(date +%s) - ${windowSeconds} )) '$1 >= since { n++; if ($2 + 0 > m) m = $2 + 0; if (NF >= 3) { d += $3; dn++ } } END { if (n) print "window=" m; if (dn) printf "download=%d\\n", d / dn }' ${samples} 2>/dev/null`;
}
