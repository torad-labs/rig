// In-memory ports for tests: typed against the same interfaces the adapters implement, so a use
// case cannot tell them apart. No mocking library — a fake is a small class you can read.
import type {
  Clock,
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
  Log,
  Offer,
  Ports,
  Process,
  Rental,
  RunOptions,
  RunResult,
  Shell,
  SpawnOptions,
  Ssh,
  SshTarget,
  Systemd,
} from "../../src/shared/ports/index.ts";

export class InMemoryFileSystem implements FileSystem {
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
        mtimeMs: 0,
        isDirectory: false,
        isSymlink: false,
      };
    return (await this.exists(path))
      ? { size: 0, mtimeMs: 0, isDirectory: true, isSymlink: false }
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
    for (const [re, s] of this.scripts) if (re.test(line)) return s(cmd, opts);
    return { code: 127, stdout: "", stderr: `fake shell: no script for ${line}` };
  }
  spawn(cmd: readonly string[], opts?: SpawnOptions): Process {
    this.spawned.push({ cmd: [...cmd], ...(opts ? { opts } : {}) });
    return { pid: 4242 + this.spawned.length, kill: () => {}, exited: Promise.resolve(0) };
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

export class FakeHttp implements Http {
  requests: Array<{ method: string; url: string; body?: unknown }> = [];
  private routes: Array<
    [RegExp, (url: string, body: unknown) => HttpResponse | Promise<HttpResponse>]
  > = [];
  /** answer urls matching `re`; the latest registration wins */
  on(re: RegExp, h: (url: string, body: unknown) => HttpResponse | Promise<HttpResponse>) {
    this.routes.unshift([re, h]);
    return this;
  }
  json(re: RegExp, value: unknown, status = 200) {
    return this.on(re, () => ({ status, text: JSON.stringify(value) }));
  }
  async request(method: "GET" | "POST", url: string, opts: { body?: unknown } = {}) {
    this.requests.push({ method, url, ...(opts.body !== undefined ? { body: opts.body } : {}) });
    for (const [re, h] of this.routes) if (re.test(url)) return h(url, opts.body);
    throw new Error(`fake http: no route for ${method} ${url}`);
  }
}

export class FakeGpu implements Gpu {
  cards = new Map<number, GpuInfo>();
  driver: string | null = "13.0";
  toolkit: string | null = "13.0";
  card(index: number, info: Partial<GpuInfo> = {}) {
    this.cards.set(index, {
      index,
      name: "NVIDIA GeForce RTX 5080",
      memoryMiB: 16303,
      computeCap: "120",
      driver: "610.43.02",
      ...info,
    });
    return this;
  }
  async query(index: number) {
    return this.cards.get(index) ?? null;
  }
  async driverCuda() {
    return this.driver;
  }
  async toolkitCuda() {
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
  async searchOffers(q: string) {
    this.ops.push(`search ${q}`);
    return this.offers;
  }
  async create(offerId: number, o: { image: string; diskGb: number; label: string }) {
    const id = this.nextId++;
    const offer = this.offers.find((x) => x.id === offerId);
    this.instances.set(id, {
      id,
      status: "loading",
      label: o.label,
      dph: offer?.dph ?? 0,
      sshHost: "ssh5.vast.ai",
      sshPort: 12345,
    });
    this.ops.push(`create ${offerId} ${o.image} ${o.diskGb}`);
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
  systemd: FakeSystemd;
  git: FakeGit;
  hasher: FakeHasher;
  host: FakeHost;
  rental: FakeRental;
  ssh: FakeSsh;
  clock: FakeClock;
  log: FakeLog;
}
export function fakePorts(): FakePorts {
  const fs = new InMemoryFileSystem();
  return {
    shell: new FakeShell(),
    fs,
    http: new FakeHttp(),
    gpu: new FakeGpu().card(0),
    systemd: new FakeSystemd(),
    git: new FakeGit(),
    hasher: new FakeHasher(fs),
    host: new FakeHost(),
    rental: new FakeRental(),
    ssh: new FakeSsh(),
    clock: new FakeClock(),
    log: new FakeLog(),
  };
}

export const sha256Of = (text: string) => new Bun.CryptoHasher("sha256").update(text).digest("hex");
