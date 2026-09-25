import { describe, expect, test } from "bun:test";
import { FetchHttp } from "../platform/fetch-http.ts";
import { HeadEndpoint } from "./head-endpoint.ts";

// Over the real adapter, not a fake: the fakes once threw a refusal no real fetch produces, and
// every fresh machine's `rig up` reported a head serving on a port nothing listened on.
describe("HeadEndpoint.presence", () => {
  test("HeadEndpoint over the real adapter: a port nothing listens on is none, a listening one is a server", async () => {
    const http = new FetchHttp();
    const clock = { now: () => Date.now(), sleep: async () => {} };
    expect(await new HeadEndpoint(http, clock, "http://127.0.0.1:1").presence()).toBe("none");
    const server = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      fetch: () => new Response("", { status: 503 }),
    });
    const presence = await new HeadEndpoint(
      http,
      clock,
      `http://127.0.0.1:${server.port}`,
    ).presence();
    server.stop(true);
    expect(presence).toBe("server");
  });
});
