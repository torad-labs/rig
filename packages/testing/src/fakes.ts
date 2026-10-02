// In-memory ports for tests: typed against the same interfaces the adapters implement, so a use
// case cannot tell them apart. No mocking library — a fake is a small class you can read.
import type {
  Bucket,
  CardLease,
  Clock,
  ContainerRun,
  Containers,
  CreateOptions,
  FileStat,
  FileSystem,
  Git,
  Gpu,
  GpuInfo,
  Hasher,
  Host,
  Http,
  HttpResponse,
  Instance,
  LeaseTerms,
  Log,
  ObjectStore,
  ObjectStores,
  Offer,
  Ports,
  Process,
  Rental,
  RunOptions,
  RunResult,
  Secrets,
  Shell,
  SpawnOptions,
  Ssh,
  SshTarget,
  Systemd,
  TemplateSpec,
} from "@rig/core";

export class InMemoryFileSystem implements FileSystem {
  /** a file's modification time where a test sets one; 0 otherwise */
  mtimes = new Map<string, number>();
  /** a file's bytes on disk where they are not its length: a sparse file */
  allocated = new Map<string, number>();
  files = new Map<string, Uint8Array>();
  dirs = new Set<string>();
  renames: Array<[string, string]> = [];
  /** path → errno code `presence` throws as, instead of answering (EACCES, EIO, a stale mount) */
  denied = new Map<string, string>();
  private enc = new TextEncoder();
  private dec = new TextDecoder();
  put(path: string, text: string) {
    this.files.set(path, this.enc.encode(text));
    this.dirs.add(dirOf(path));
    return this;
  }
  /** make `presence(path)` throw as if lstat failed with `code`, rather than answering "absent"
   *  the way a plain missing file would */
  deny(path: string, code = "EACCES") {
    this.denied.set(path, code);
    return this;
  }
  text(path: string) {
    const b = this.files.get(path);
    return b === undefined ? undefined : this.dec.decode(b);
  }
  async readText(path: string) {
    const b = this.files.get(path);
    if (b === undefined) throw new Error(`ENOENT ${path}`);
    return this.dec.decode(b);
  }
  async readBytes(path: string) {
    const b = this.files.get(path);
    if (b === undefined) throw new Error(`ENOENT ${path}`);
    return b;
  }
  async readRange(path: string, offset: number, length: number) {
    const b = await this.readBytes(path);
    return b.slice(offset, offset + length);
  }
  async writeText(path: string, text: string) {
    this.put(path, text);
  }
  /** every replaced path, in order: the renames a partial-file reader can never see into */
  replaced: string[] = [];
  async replaceText(path: string, text: string) {
    this.replaced.push(path);
    this.put(path, text);
  }
  async writeBytes(path: string, bytes: Uint8Array) {
    this.files.set(path, bytes);
    this.dirs.add(dirOf(path));
  }
  async writeAt(path: string, offset: number, bytes: Uint8Array) {
    const cur = this.files.get(path);
    if (!cur) throw new Error(`ENOENT ${path}`);
    const out = new Uint8Array(Math.max(cur.length, offset + bytes.length));
    out.set(cur);
    out.set(bytes, offset);
    this.files.set(path, out);
  }
  async exists(path: string) {
    return (
      this.files.has(path) ||
      this.dirs.has(path) ||
      [...this.files.keys(), ...this.dirs].some((p) => p.startsWith(`${path}/`))
    );
  }
  async presence(path: string): Promise<"present" | "absent"> {
    const code = this.denied.get(path);
    if (code && code !== "ENOENT" && code !== "ENOTDIR") {
      const err = new Error(`fake ${code}: ${path}`) as NodeJS.ErrnoException;
      err.code = code;
      throw err;
    }
    return (await this.exists(path)) ? "present" : "absent";
  }
  async stat(path: string): Promise<FileStat | null> {
    if (this.files.has(path))
      return {
        size: this.files.get(path)!.length,
        allocated: this.allocated.get(path) ?? this.files.get(path)!.length,
        mtimeMs: this.mtimes.get(path) ?? 0,
        isDirectory: false,
        isSymlink: false,
      };
    return (await this.exists(path))
      ? { size: 0, allocated: 0, mtimeMs: 0, isDirectory: true, isSymlink: false }
      : null;
  }
  async mkdirp(path: string) {
    this.dirs.add(path);
  }
  async rename(from: string, to: string) {
    this.renames.push([from, to]);
    const moved: Array<[string, Uint8Array]> = [];
    for (const [p, b] of this.files)
      if (p === from || p.startsWith(`${from}/`)) moved.push([to + p.slice(from.length), b]);
    if (!moved.length && !this.dirs.has(from)) throw new Error(`ENOENT ${from}`);
    for (const [p] of this.files) if (p === from || p.startsWith(`${from}/`)) this.files.delete(p);
    for (const [p, b] of moved) this.files.set(p, b);
    if (this.dirs.delete(from)) this.dirs.add(to);
  }
  async remove(path: string) {
    for (const p of [...this.files.keys()])
      if (p === path || p.startsWith(`${path}/`)) this.files.delete(p);
    for (const d of [...this.dirs]) if (d === path || d.startsWith(`${path}/`)) this.dirs.delete(d);
  }
  async list(dir: string) {
    const names = new Set<string>();
    for (const p of [...this.files.keys(), ...this.dirs])
      if (p.startsWith(`${dir}/`)) names.add(p.slice(dir.length + 1).split("/")[0]!);
    return [...names].sort();
  }
  async copy(from: string, to: string) {
    const b = this.files.get(from);
    if (!b) throw new Error(`ENOENT ${from}`);
    this.files.set(to, new Uint8Array(b));
  }
  async copyTree(from: string, to: string) {
    for (const [p, b] of [...this.files])
      if (p.startsWith(`${from}/`)) this.files.set(to + p.slice(from.length), new Uint8Array(b));
    this.dirs.add(to);
  }
  async linkOrCopy(from: string, to: string) {
    await this.copy(from, to);
  }
  async realpath(path: string) {
    if (!(await this.exists(path))) throw new Error(`ENOENT ${path}`);
    return path;
  }
  /** the disk a test sets; unset, a disk no fetch fills */
  free = Number.MAX_SAFE_INTEGER;
  async freeBytes(_path: string) {
    return this.free;
  }
}
const dirOf = (p: string) => p.slice(0, p.lastIndexOf("/"));

export type Script = (cmd: readonly string[], opts?: RunOptions) => RunResult | Promise<RunResult>;
export class FakeShell implements Shell {
  calls: string[][] = [];
  spawned: Array<{ cmd: string[]; opts?: SpawnOptions }> = [];
  tools = new Set<string>();
  private scripts: Array<[RegExp, Script]> = [];
  /** answer commands whose joined argv matches `re`; the latest registration wins */
  on(re: RegExp, script: Script | RunResult) {
    this.scripts.unshift([re, typeof script === "function" ? script : () => script]);
    return this;
  }
  async run(cmd: readonly string[], opts?: RunOptions): Promise<RunResult> {
    this.calls.push([...cmd]);
    const line = cmd.join(" ");
    for (const [re, s] of this.scripts)
      if (re.test(line)) {
        const result = await s(cmd, opts);
        // the scripted output, line by line, as a real run would hand it over while it ran
        for (const [text, stream] of [
          [result.stdout, "stdout"],
          [result.stderr, "stderr"],
        ] as const)
          for (const each of text.split("\n").filter(Boolean)) opts?.onLine?.(each, stream);
        return result;
      }
    return { code: 127, stdout: "", stderr: `fake shell: no script for ${line}` };
  }
  /** how a spawned process ends: at once with this code (a command run to completion, a server
   *  that dies at load), or only when it is killed (a server that runs until it is stopped) */
  spawnExit: number | "on-kill" = 0;
  spawn(cmd: readonly string[], opts?: SpawnOptions): Process {
    this.spawned.push({ cmd: [...cmd], ...(opts ? { opts } : {}) });
    const pid = 4242 + this.spawned.length;
    if (this.spawnExit !== "on-kill")
      return { pid, kill: () => {}, exited: Promise.resolve(this.spawnExit) };
    let exit: (code: number) => void = () => {};
    const exited = new Promise<number>((resolve) => {
      exit = resolve;
    });
    return { pid, kill: () => exit(0), exited };
  }
  async which(name: string) {
    return this.tools.has(name) ? `/usr/bin/${name}` : null;
  }
}

/** what the Http port rejects with when nothing listens on the port (FetchHttp's contract) */
export function connectionRefused(): Error {
  const error = new Error("fake http: nothing is listening");
  error.name = "ConnectionRefused";
  return error;
}

type Route = (
  url: string,
  body: unknown,
  request: { method: string; headers: Record<string, string> },
) => HttpResponse | Promise<HttpResponse>;
export class FakeHttp implements Http {
  requests: Array<{
    method: string;
    url: string;
    body?: unknown;
    headers?: Record<string, string>;
  }> = [];
  private routes: Array<[RegExp, Route]> = [];
  /** answer urls matching `re`; the latest registration wins */
  on(re: RegExp, h: Route) {
    this.routes.unshift([re, h]);
    return this;
  }
  json(re: RegExp, value: unknown, status = 200) {
    return this.on(re, () => ({ status, text: JSON.stringify(value) }));
  }
  async request(
    method: "GET" | "POST" | "HEAD" | "PUT",
    url: string,
    opts: { body?: unknown; headers?: Record<string, string> } = {},
  ) {
    this.requests.push({
      method,
      url,
      ...(opts.body !== undefined ? { body: opts.body } : {}),
      ...(opts.headers ? { headers: opts.headers } : {}),
    });
    for (const [re, h] of this.routes)
      if (re.test(url)) return h(url, opts.body, { method, headers: opts.headers ?? {} });
    throw new Error(`fake http: no route for ${method} ${url}`);
  }
}

/** the cards a run held, and its command run on the shell as it would be under the lease */
export class FakeCardLease implements CardLease {
  leases: Array<{ cards: number[]; terms: LeaseTerms; cmd: string[] }> = [];
  constructor(private readonly shell: Shell) {}
  async run(
    cards: readonly number[],
    terms: LeaseTerms,
    cmd: readonly string[],
    opts?: RunOptions,
  ) {
    this.leases.push({ cards: [...cards], terms: { ...terms }, cmd: [...cmd] });
    return this.shell.run(cmd, opts);
  }
}

export class FakeGpu implements Gpu {
  cards = new Map<number, GpuInfo>();
  driver: string | null = "13.0";
  toolkit: string | null = "13.0";
  /** MiB each pid holds, on whichever card a test asks about */
  held = new Map<number, number>();
  card(index: number, info: Partial<GpuInfo> = {}) {
    this.cards.set(index, {
      index,
      name: "NVIDIA GeForce RTX 5080",
      memoryMiB: 16303,
      usedMiB: 0,
      computeCap: "120",
      driver: "610.43.02",
      ...info,
    });
    return this;
  }
  async query(index: number) {
    return this.cards.get(index) ?? null;
  }
  async list() {
    return [...this.cards.values()].sort((a, b) => a.index - b.index);
  }
  async processMiB(_index: number, pid: number) {
    return this.held.get(pid) ?? 0;
  }
  /** each card's utilization, as nvidia-smi would read it now */
  util: number[] = [0];
  async utilization() {
    return this.util;
  }
  async driverCuda() {
    return this.driver;
  }
  /** the compilers asked for their version, undefined for nvcc on PATH */
  toolkitAsked: (string | undefined)[] = [];
  async toolkitCuda(compiler?: string) {
    this.toolkitAsked.push(compiler);
    return this.toolkit;
  }
}

export class FakeSystemd implements Systemd {
  units = new Map<string, string>();
  ops: string[] = [];
  active = new Set<string>();
  pids = new Map<string, number>();
  constructor(private readonly dir = "/home/u/.config/systemd/user") {}
  unitDir() {
    return this.dir;
  }
  async daemonReload() {
    this.ops.push("daemon-reload");
  }
  async enable(u: string) {
    this.ops.push(`enable ${u}`);
  }
  async disable(u: string) {
    this.ops.push(`disable ${u}`);
  }
  async restart(u: string) {
    this.ops.push(`restart ${u}`);
    this.active.add(u);
  }
  async stop(u: string) {
    this.ops.push(`stop ${u}`);
    this.active.delete(u);
  }
  async isActive(u: string) {
    return this.active.has(u);
  }
  /** a unit's last Result; absent reads as success, what systemd reports for a unit that never failed */
  results = new Map<string, string | null>();
  async lastResult(u: string) {
    return this.results.has(u) ? this.results.get(u)! : "success";
  }
  async mainPid(u: string) {
    return this.pids.get(u) ?? null;
  }
  /** logind's Linger for this user (null: unreadable), and whether enable-linger is allowed */
  lingering: boolean | null = true;
  lingerAllowed = true;
  async linger() {
    return this.lingering;
  }
  async enableLinger() {
    this.ops.push("enable-linger");
    if (this.lingerAllowed) this.lingering = true;
    return this.lingerAllowed;
  }
}

export class FakeGit implements Git {
  heads = new Map<string, string>();
  fetched: Array<{ dir: string; repo: string; sha: string }> = [];
  dirty = new Set<string>();
  /** what HEAD holds, repo-relative path → content: exportTree writes what its paths name of it */
  committed = new Map<string, string | Uint8Array>();
  exported: Array<{ ref: string; paths: string[]; into: string }> = [];
  constructor(private readonly fs?: InMemoryFileSystem) {}
  async revParse(dir: string, _ref: string) {
    return this.heads.get(dir) ?? null;
  }
  async isClean(dir: string) {
    return !this.dirty.has(dir);
  }
  async fetchCommit(dir: string, repo: string, sha: string) {
    this.fetched.push({ dir, repo, sha });
    this.heads.set(dir, sha);
  }
  async exportTree(_dir: string, ref: string, paths: readonly string[], into: string) {
    this.exported.push({ ref, paths: [...paths], into });
    for (const [path, content] of this.committed) {
      if (!paths.some((named) => path === named || path.startsWith(`${named}/`))) continue;
      const bytes = typeof content === "string" ? new TextEncoder().encode(content) : content;
      this.fs?.files.set(`${into}/${path}`, bytes);
    }
  }
}

/** sha256 of the fake file's bytes, so a test controls a hash by controlling content — or pins one. */
export class FakeHasher implements Hasher {
  pinned = new Map<string, string>();
  /** path → errno code reading it throws as: a file lstat sees but open refuses (EACCES, EIO) */
  unreadable = new Map<string, string>();
  constructor(private readonly fs: InMemoryFileSystem) {}
  async sha256File(path: string) {
    const code = this.unreadable.get(path);
    if (code) {
      const err = new Error(`fake ${code}: ${path}`) as NodeJS.ErrnoException;
      err.code = code;
      throw err;
    }
    const p = this.pinned.get(path);
    if (p) return p;
    const b = this.fs.files.get(path);
    if (!b) throw new Error(`ENOENT ${path}`);
    return new Bun.CryptoHasher("sha256").update(b).digest("hex");
  }
}

export class FakeHost implements Host {
  name = "box";
  cpus = 16;
  ram = 62818;
  /** port → pid of the process listening on it */
  listeners = new Map<number, number>();
  libc: string | null = "2.39";
  async glibc() {
    return this.libc;
  }
  async listeningPid(port: number) {
    return this.listeners.get(port) ?? null;
  }
  hostname() {
    return this.name;
  }
  cpuCount() {
    return this.cpus;
  }
  async ramMiB() {
    return this.ram;
  }
  /** the container's CPU time in microseconds; a test moves it to make the box busy */
  cpu: number | null = 0;
  async cpuMicros() {
    return this.cpu;
  }
}

export class FakeRental implements Rental {
  balance = 95;
  keys = new Set<string>();
  offers: Offer[] = [];
  instances = new Map<number, Instance>();
  ops: string[] = [];
  nextId = 1000;
  /** what `show` answers for an instance over successive calls (statuses), the last repeated */
  statuses: string[] = ["loading", "running"];
  async funds() {
    return this.balance;
  }
  async hasSshKey(k: string) {
    return this.keys.has(k);
  }
  async registerSshKey(k: string) {
    this.keys.add(k);
    this.ops.push("register-key");
  }
  async searchOffers(q: string, _diskGb: number) {
    this.ops.push(`search ${q}`);
    return this.offers;
  }
  async create(offerId: number, o: CreateOptions) {
    const id = this.nextId++;
    const offer = this.offers.find((x) => x.id === offerId);
    this.instances.set(id, {
      id,
      status: "loading",
      label: o.label,
      dph: offer?.dph ?? 0,
      sshHost: "ssh5.vast.ai",
      sshPort: 12345,
      gpuUtil: 0, // vast samples a running box's card; a test drops it to model a box with none
    });
    this.ops.push(
      "templateHash" in o
        ? `create ${offerId} template ${o.templateHash} ${o.diskGb} label ${o.label}`
        : `create ${offerId} ${o.image} ${o.diskGb}`,
    );
    return id;
  }
  private shows = 0;
  async show(id: number) {
    const i = this.instances.get(id);
    if (!i) return null;
    const st = this.statuses[Math.min(this.shows++, this.statuses.length - 1)]!;
    return { ...i, status: st };
  }
  async list() {
    return [...this.instances.values()];
  }
  async destroy(id: number) {
    this.ops.push(`destroy ${id}`);
    this.instances.delete(id);
  }
  templates: Array<{ spec: TemplateSpec; hashId?: string }> = [];
  async saveTemplate(spec: TemplateSpec, hashId?: string) {
    this.templates.push({ spec, ...(hashId ? { hashId } : {}) });
    return { id: 77, hashId: `hash-${this.templates.length}` };
  }
}

export class FakeSsh implements Ssh {
  calls: string[] = [];
  pushed: Array<[string, string]> = [];
  pulled: Array<[string, string]> = [];
  private scripts: Array<[RegExp, (cmd: string) => RunResult | Promise<RunResult>]> = [];
  /** answer remote commands matching `re`; the latest registration wins; unmatched succeed silently */
  on(re: RegExp, s: ((cmd: string) => RunResult | Promise<RunResult>) | RunResult) {
    this.scripts.unshift([re, typeof s === "function" ? s : () => s]);
    return this;
  }
  async run(_t: SshTarget, cmd: string) {
    this.calls.push(cmd);
    for (const [re, s] of this.scripts) if (re.test(cmd)) return s(cmd);
    return { code: 0, stdout: "", stderr: "" };
  }
  async push(_t: SshTarget, local: string, remote: string) {
    this.pushed.push([local, remote]);
  }
  async pull(_t: SshTarget, remote: string, local: string) {
    this.pulled.push([remote, local]);
  }
}

/** docker as a record: each build, run and save, runs answered by `on` scripts like FakeShell's */
export class FakeContainers implements Containers {
  builds: Array<{ context: string; tag: string; labels: Record<string, string> }> = [];
  runs: Array<{ image: string; cmd: string[]; options?: ContainerRun }> = [];
  saved: Array<{ image: string; tarball: string }> = [];
  private scripts: Array<[RegExp, (cmd: readonly string[]) => RunResult]> = [];
  buildResult: RunResult = { code: 0, stdout: "", stderr: "" };
  /** answer runs whose joined command matches `re`; the latest registration wins */
  on(re: RegExp, answer: RunResult | ((cmd: readonly string[]) => RunResult)) {
    this.scripts.unshift([re, typeof answer === "function" ? answer : () => answer]);
    return this;
  }
  async build(context: string, tag: string, labels: Record<string, string>, _logPath: string) {
    this.builds.push({ context, tag, labels });
    return this.buildResult;
  }
  async run(image: string, cmd: readonly string[], options?: ContainerRun) {
    this.runs.push({ image, cmd: [...cmd], ...(options ? { options } : {}) });
    const line = cmd.join(" ");
    for (const [re, answer] of this.scripts) if (re.test(line)) return answer(cmd);
    return { code: 127, stdout: "", stderr: `fake containers: no script for ${line}` };
  }
  async save(image: string, tarball: string) {
    this.saved.push({ image, tarball });
    return { code: 0, stdout: "", stderr: "" };
  }
}

/** buckets as maps: bucket name → key → the bytes and their content type */
export class FakeObjectStores implements ObjectStores {
  buckets = new Map<string, Map<string, { bytes: Uint8Array; type?: string }>>();
  opened: Bucket[] = [];
  constructor(private readonly fs: InMemoryFileSystem) {}
  open(bucket: Bucket): ObjectStore {
    this.opened.push(bucket);
    const objects = this.buckets.get(bucket.bucket) ?? new Map();
    this.buckets.set(bucket.bucket, objects);
    return {
      exists: async (key) => objects.has(key),
      put: async (key, from, contentType) => {
        const bytes =
          "file" in from
            ? this.fs.files.get(from.file)
            : typeof from.bytes === "string"
              ? new TextEncoder().encode(from.bytes)
              : from.bytes;
        if (!bytes) throw new Error(`ENOENT ${"file" in from ? from.file : key}`);
        objects.set(key, { bytes, ...(contentType ? { type: contentType } : {}) });
      },
    };
  }
}

export class FakeClock implements Clock {
  t = 1_700_000_000_000;
  slept: number[] = [];
  now() {
    return this.t;
  }
  async sleep(ms: number) {
    this.slept.push(ms);
    this.t += ms;
  }
}

/** a keyring as a map of "service/key" to its value */
export class FakeSecrets implements Secrets {
  held = new Map<string, string>();
  lookups: string[] = [];
  async lookup(service: string, key: string) {
    this.lookups.push(`${service}/${key}`);
    return this.held.get(`${service}/${key}`) ?? null;
  }
}

export class FakeLog implements Log {
  lines: string[] = [];
  info(m: string) {
    this.lines.push(`info ${m}`);
  }
  warn(m: string) {
    this.lines.push(`warn ${m}`);
  }
  error(m: string) {
    this.lines.push(`error ${m}`);
  }
}

export interface FakePorts extends Ports {
  shell: FakeShell;
  fs: InMemoryFileSystem;
  http: FakeHttp;
  gpu: FakeGpu;
  cardLease: FakeCardLease;
  systemd: FakeSystemd;
  git: FakeGit;
  hasher: FakeHasher;
  host: FakeHost;
  rental: FakeRental;
  ssh: FakeSsh;
  clock: FakeClock;
  containers: FakeContainers;
  objectStores: FakeObjectStores;
  secrets: FakeSecrets;
  log: FakeLog;
}
export function fakePorts(): FakePorts {
  const fs = new InMemoryFileSystem();
  const shell = new FakeShell();
  return {
    shell,
    fs,
    http: new FakeHttp(),
    gpu: new FakeGpu().card(0),
    cardLease: new FakeCardLease(shell),
    systemd: new FakeSystemd(),
    git: new FakeGit(fs),
    hasher: new FakeHasher(fs),
    host: new FakeHost(),
    rental: new FakeRental(),
    ssh: new FakeSsh(),
    clock: new FakeClock(),
    containers: new FakeContainers(),
    objectStores: new FakeObjectStores(fs),
    secrets: new FakeSecrets(),
    log: new FakeLog(),
  };
}

export const sha256Of = (text: string) => new Bun.CryptoHasher("sha256").update(text).digest("hex");
