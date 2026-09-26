import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { connect } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

const runner = join(import.meta.dir, "glm53-admission.ts");

test("drain finishes active responses but closes idle keep-alive admission", async () => {
  const scratch = mkdtempSync(join(tmpdir(), "rig-glm53-admission-"));
  const socketPath = join(scratch, "control.sock");
  let release: (() => void) | undefined;
  let requests = 0;
  const backend = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch() {
      requests++;
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode("started"));
            release = () => {
              controller.enqueue(new TextEncoder().encode("done"));
              controller.close();
            };
          },
        }),
      );
    },
  });
  const server = Bun.spawn(["bun", runner], {
    env: {
      ...process.env,
      GLM53_ADMISSION_PORT: "0",
      GLM53_BACKEND_PORT: String(backend.port),
      GLM53_ADMISSION_SOCKET: socketPath,
    },
    stdout: "pipe",
    stderr: "pipe",
  });
  let client: ReturnType<typeof connect> | undefined;
  try {
    const ready = await server.stdout.getReader().read();
    const match = new TextDecoder().decode(ready.value).match(/READY (\d+)/);
    expect(match).not.toBeNull();
    const port = Number(match?.[1]);
    client = connect({ host: "127.0.0.1", port });
    await new Promise<void>((resolve, reject) => {
      client?.once("error", reject);
      client?.once("connect", resolve);
    });
    const firstResponse = new Promise<void>((resolve) => client?.once("data", () => resolve()));
    client.write("GET /held HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n");
    await firstResponse;
    expect(requests).toBe(1);

    const drain = Bun.spawn(
      [
        "curl",
        "-fsS",
        "--max-time",
        "10",
        "--unix-socket",
        socketPath,
        "-X",
        "POST",
        "http://localhost/drain",
      ],
      { stdout: "pipe", stderr: "pipe" },
    );
    let finished = false;
    void drain.exited.then(() => {
      finished = true;
    });
    await Bun.sleep(100);
    expect(finished).toBe(false);
    release?.();
    expect(await drain.exited).toBe(0);
    await Bun.sleep(50);
    if (!client.destroyed)
      client.write("GET /too-late HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n");
    await Bun.sleep(50);
    expect(requests).toBe(1);

    const resume = Bun.spawnSync([
      "curl",
      "-fsS",
      "--max-time",
      "5",
      "--unix-socket",
      socketPath,
      "-X",
      "POST",
      "http://localhost/resume",
    ]);
    expect(resume.exitCode).toBe(0);
    const again = await fetch(`http://127.0.0.1:${port}/again`);
    expect(again.ok).toBe(true);
    expect(requests).toBe(2);
    release?.();
    await again.text();
    const secondDrain = Bun.spawnSync([
      "curl",
      "-fsS",
      "--max-time",
      "5",
      "--unix-socket",
      socketPath,
      "-X",
      "POST",
      "http://localhost/drain",
    ]);
    expect(secondDrain.exitCode).toBe(0);
  } finally {
    client?.destroy();
    server.kill();
    await server.exited;
    await backend.stop(true);
    rmSync(scratch, { recursive: true, force: true });
  }
}, 30000);

test("long prefill can finish without proxy idle timeout", async () => {
  const scratch = mkdtempSync(join(tmpdir(), "rig-glm53-prefill-"));
  const socketPath = join(scratch, "control.sock");
  const backend = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    idleTimeout: 0,
    async fetch() {
      await Bun.sleep(11_000);
      return new Response("READY");
    },
  });
  const server = Bun.spawn(["bun", runner], {
    env: {
      ...process.env,
      GLM53_ADMISSION_PORT: "0",
      GLM53_BACKEND_PORT: String(backend.port),
      GLM53_ADMISSION_SOCKET: socketPath,
    },
    stdout: "pipe",
    stderr: "pipe",
  });
  try {
    const ready = await server.stdout.getReader().read();
    const match = new TextDecoder().decode(ready.value).match(/READY (\d+)/);
    expect(match).not.toBeNull();
    const response = await fetch(`http://127.0.0.1:${match?.[1]}/slow`);
    expect(await response.text()).toBe("READY");
  } finally {
    server.kill();
    await server.exited;
    await backend.stop(true);
    rmSync(scratch, { recursive: true, force: true });
  }
}, 20000);
