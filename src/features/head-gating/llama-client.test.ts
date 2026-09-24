import { describe, expect, test } from "bun:test";
import { FakeHttp } from "../../../test/fakes/index.ts";
import { LlamaClient } from "./llama-client.ts";

const BASE = "http://127.0.0.1:8098";

function connectionClosed(): Error {
  const error = new Error("POST /completion: the connection closed before a response");
  error.name = "ConnectionClosed";
  return error;
}

/** /completion and /v1/chat/completions fail with `failures` in turn, then answer; /health as `healthy` */
function server(failures: Error[], healthy = true) {
  const http = new FakeHttp();
  const answer = (text: string) => () => {
    const failure = failures.shift();
    if (failure) throw failure;
    return { status: 200, text };
  };
  http.on(/\/completion$/, answer(JSON.stringify({ content: "  return a + b\n" })));
  http.on(
    /\/v1\/chat\/completions$/,
    answer(JSON.stringify({ choices: [{ message: { content: "four" } }] })),
  );
  http.on(/\/health$/, () => ({ status: healthy ? 200 : 503, text: "" }));
  const posts = () => http.requests.filter((r) => r.method === "POST").length;
  return { http, posts, client: new LlamaClient(http, BASE) };
}

describe("LlamaClient: a connection closed before a response", () => {
  test("a greedy completion is sent once more while the server answers /health, and yields its reply", async () => {
    const { client, posts } = server([connectionClosed()]);
    expect((await client.completion("def add(a, b):", { nPredict: 64 })).text).toBe(
      "  return a + b\n",
    );
    expect(posts()).toBe(2);
  });
  test("once only: a second close on the repeat fails the request", async () => {
    const { client, posts } = server([connectionClosed(), connectionClosed()]);
    await expect(client.completion("def add(a, b):", { nPredict: 64 })).rejects.toThrow(
      "connection closed",
    );
    expect(posts()).toBe(2);
  });
  test("a server that does not answer /health fails the request: a dead server is never retried", async () => {
    const { client, posts } = server([connectionClosed()], false);
    await expect(client.completion("def add(a, b):", { nPredict: 64 })).rejects.toThrow(
      "connection closed",
    );
    expect(posts()).toBe(1);
  });
  test("a sampled (not greedy) chat is not repeated: a second draw is a different sample", async () => {
    const { client, posts, http } = server([connectionClosed()]);
    await expect(client.chat("2 + 2?", { maxTokens: 8, greedy: false })).rejects.toThrow(
      "connection closed",
    );
    expect(posts()).toBe(1);
    expect(http.requests.some((r) => r.url.endsWith("/health"))).toBe(false);
  });
  test("a greedy chat is repeated like a completion", async () => {
    const { client, posts } = server([connectionClosed()]);
    expect((await client.chat("2 + 2?", { maxTokens: 8 })).text).toBe("four");
    expect(posts()).toBe(2);
  });
  test("any other failure (a timeout) is not repeated", async () => {
    const timeout = new Error("The operation timed out.");
    timeout.name = "TimeoutError";
    const { client, posts } = server([timeout]);
    await expect(client.completion("def add(a, b):", { nPredict: 64 })).rejects.toThrow(
      "timed out",
    );
    expect(posts()).toBe(1);
  });
});
