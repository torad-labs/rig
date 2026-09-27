import type { Http, HttpResponse } from "../ports/index.ts";

export class FetchHttp implements Http {
  async request(
    method: "GET" | "POST",
    url: string,
    opts: { body?: unknown; timeoutMs?: number | false } = {},
  ): Promise<HttpResponse> {
    const init: RequestInit & { timeout?: false } = {
      method,
      headers: { "content-type": "application/json" },
    };
    if (opts.body !== undefined) init.body = JSON.stringify(opts.body);
    if (opts.timeoutMs === false)
      init.timeout = false; // Bun: no idle timeout (a 250K prefill runs ~330 s before the first byte)
    else if (opts.timeoutMs) init.signal = AbortSignal.timeout(opts.timeoutMs);
    let response: Response;
    try {
      response = await fetch(url, init);
    } catch (error) {
      // Bun 1.4 codes a refused connection ConnectionRefused (Node's is ECONNREFUSED) and one
      // closed or reset before the response ECONNRESET; the port names both, anything else
      // (a timeout, a DNS failure) stays what it is
      const code = (error as { code?: unknown }).code;
      if (code === "ConnectionRefused" || code === "ECONNREFUSED")
        throw named("ConnectionRefused", `${method} ${url}: nothing is listening`, error);
      if (code !== "ECONNRESET") throw error;
      throw named(
        "ConnectionClosed",
        `${method} ${url}: the connection closed before a response`,
        error,
      );
    }
    return { status: response.status, text: await response.text() };
  }
}

function named(name: string, message: string, cause: unknown): Error {
  const error = new Error(message, { cause });
  error.name = name;
  return error;
}
