import { describe, expect, test } from "bun:test";
import { createServer } from "node:net";
import { FetchHttp } from "./fetch-http.ts";

/** a TCP server that reads the request and closes (or resets) the socket without a response */
async function closingServer(reset: boolean) {
  const server = createServer((socket) =>
    socket.once("data", () => (reset ? socket.resetAndDestroy() : socket.destroy())),
  );
  await new Promise<void>((listening) => server.listen(0, "127.0.0.1", listening));
  const { port } = server.address() as { port: number };
  return { url: `http://127.0.0.1:${port}/completion`, close: () => server.close() };
}

describe("FetchHttp", () => {
  test("a connection closed or reset before any response rejects as ConnectionClosed", async () => {
    for (const reset of [false, true]) {
      const server = await closingServer(reset);
      const error = await new FetchHttp()
        .request("POST", server.url, { body: {} })
        .catch((e: unknown) => e as Error);
      server.close();
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).name).toBe("ConnectionClosed");
      expect((error as Error).message).toContain("/completion");
    }
  });
  test("a refused connection rejects as ConnectionRefused, the one answer that proves nothing listens", async () => {
    const error = await new FetchHttp()
      .request("GET", "http://127.0.0.1:1/health")
      .catch((e: unknown) => e as Error);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).name).toBe("ConnectionRefused");
    expect((error as Error).message).toContain("127.0.0.1:1/health");
  });
});
