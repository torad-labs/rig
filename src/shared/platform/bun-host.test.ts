import { describe, expect, test } from "bun:test";
import { connect } from "node:net";
import { BunHost } from "./bun-host.ts";
import { BunShell } from "./bun-shell.ts";

// Over the real /proc: a minimal container has no ss (iproute2), and rig once reported the head it
// had just started as "held by no process ss can name" there (splice setup e2e, 2026-09-24).
describe("BunHost.listeningPid", () => {
  test("the process listening on a port is named, over IPv4 and IPv6; a closed port is null", async () => {
    const host = new BunHost(new BunShell());
    for (const hostname of ["127.0.0.1", "::1"]) {
      const server = Bun.serve({ port: 0, hostname, fetch: () => new Response("") });
      const pid = await host.listeningPid(server.port as number);
      server.stop(true);
      expect(pid, hostname).toBe(process.pid);
    }
    expect(await host.listeningPid(1)).toBeNull();
  });
  test("a connection's own local port is not a listener: only a LISTEN row names an owner", async () => {
    const host = new BunHost(new BunShell());
    const server = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: () => new Response("") });
    const client = connect(server.port as number, "127.0.0.1");
    await new Promise((connected) => client.once("connect", connected));
    const ephemeral = client.localPort as number; // ESTABLISHED under this pid, listening nowhere
    const pid = await host.listeningPid(ephemeral);
    client.destroy();
    server.stop(true);
    expect(pid).toBeNull();
  });
});
