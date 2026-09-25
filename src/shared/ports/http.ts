// Http: one seam between rig and the machine. A port names a capability, never a tool.
export interface HttpResponse {
  status: number;
  text: string;
}
/** A request whose connection closed before any response rejects with an Error named
 *  "ConnectionClosed": the server may still be up (a keep-alive socket reset under load), so a
 *  caller whose request is safe to repeat may repeat it. A request to a port nothing listens on
 *  rejects with an Error named "ConnectionRefused": the one answer that proves no server is there. */
export interface Http {
  request(
    method: "GET" | "POST",
    url: string,
    opts?: { body?: unknown; timeoutMs?: number | false },
  ): Promise<HttpResponse>;
}
