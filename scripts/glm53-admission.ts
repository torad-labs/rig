// Request-level fence for the GLM SSH tunnel. Only this process binds the public local port.
import { chmodSync, rmSync } from "node:fs";

const port = Number(process.env.GLM53_ADMISSION_PORT ?? 8102);
const backendPort = Number(process.env.GLM53_BACKEND_PORT ?? 18102);
const socketPath = process.env.GLM53_ADMISSION_SOCKET;
if (!socketPath) throw new Error("GLM53_ADMISSION_SOCKET is required");
process.umask(0o077);
rmSync(socketPath, { force: true });

let listenPort = port;
let publicServer: ReturnType<typeof Bun.serve> | undefined;
let drainPromise: Promise<void> | undefined;
let lease: ReturnType<typeof setTimeout> | undefined;
let epoch = 0;

function reopen() {
  epoch++;
  if (lease) clearTimeout(lease);
  lease = undefined;
  drainPromise = undefined;
  if (publicServer) return;
  publicServer = Bun.serve({
    hostname: "127.0.0.1",
    port: listenPort,
    idleTimeout: 0,
    async fetch(request) {
      const upstream = new URL(request.url);
      upstream.hostname = "127.0.0.1";
      upstream.port = String(backendPort);
      const headers = new Headers(request.headers);
      headers.delete("host");
      headers.delete("connection");
      try {
        const response = await fetch(upstream, {
          method: request.method,
          headers,
          body: request.body,
          duplex: "half",
          signal: request.signal,
        });
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      } catch {
        return new Response("GLM tunnel unavailable", { status: 502 });
      }
    },
  });
  if (listenPort === 0) {
    if (!publicServer.port) throw new Error("GLM admission proxy has no TCP port");
    listenPort = publicServer.port;
  }
}

reopen();
const control = Bun.serve({
  unix: socketPath,
  async fetch(request) {
    this.timeout(request, 0);
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
    const path = new URL(request.url).pathname;
    if (path === "/resume") {
      reopen();
      return new Response("admission open\n");
    }
    if (path !== "/drain") return new Response("Not found", { status: 404 });
    if (!drainPromise) {
      const server = publicServer;
      if (!server) return new Response("admission already closed", { status: 409 });
      publicServer = undefined;
      const current = ++epoch;
      lease = setTimeout(() => {
        if (current === epoch) reopen();
      }, 300_000);
      drainPromise = server.stop(false);
    }
    const current = epoch;
    await drainPromise;
    if (current !== epoch || publicServer)
      return new Response("drain lease expired", { status: 409 });
    return new Response("admission drained\n");
  },
});
chmodSync(socketPath, 0o600);
process.stdout.write(`READY ${listenPort}\n`);
process.on("SIGTERM", () => {
  void control.stop(true);
  void publicServer?.stop(true);
  rmSync(socketPath, { force: true });
  process.exit(0);
});
