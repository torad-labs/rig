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
      // Bun names a connection closed or reset before the response ECONNRESET; a refused one is
      // ConnectionRefused, and stays what it is
      if ((error as { code?: unknown }).code !== "ECONNRESET") throw error;
      const closed = new Error(`${method} ${url}: the connection closed before a response`, {
        cause: error,
      });
      closed.name = "ConnectionClosed";
      throw closed;
    }
    return { status: response.status, text: await response.text() };
  }
}
