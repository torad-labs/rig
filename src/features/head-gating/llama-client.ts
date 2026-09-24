// The llama-server API a probe speaks, over the Http port: greedy by default (temperature 0,
// top_k 1, seed 1, prompt cache off) so the same weights give the same tokens.
import type { Http, HttpResponse } from "../../shared/ports/index.ts";
import type {
  ChatOptions,
  ChatReply,
  CompletionOptions,
  CompletionReply,
  HeadClient,
  Timings,
  TokenLogprob,
} from "./head-client.ts";

export type { ChatOptions, ChatReply, CompletionReply, Timings };

interface ChatBody {
  error?: unknown;
  choices?: Array<{
    message?: { content?: string; reasoning_content?: string; reasoning?: string };
    logprobs?: {
      content?: Array<{ token: string; logprob: number; top_logprobs?: TokenChoiceBody[] }>;
    };
  }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
  timings?: Timings;
}

interface TokenChoiceBody {
  token: string;
  logprob: number;
}

interface CompletionBody {
  error?: unknown;
  content?: string;
  timings?: Timings;
}

const GREEDY = { temperature: 0, top_k: 1, seed: 1, cache_prompt: false };
const GENERATION_TIMEOUT_MS = 600_000;

export class LlamaClient implements HeadClient {
  constructor(
    private readonly http: Http,
    readonly base: string,
  ) {}

  async chat(prompt: string, options: ChatOptions): Promise<ChatReply> {
    const body = {
      messages: [{ role: "user", content: prompt }],
      max_tokens: options.maxTokens,
      chat_template_kwargs: { enable_thinking: options.thinking ?? false },
      ...((options.greedy ?? true) ? GREEDY : {}),
      ...(options.topLogprobs ? { logprobs: true, top_logprobs: options.topLogprobs } : {}),
    };
    const reply = await this.post(
      "/v1/chat/completions",
      body,
      options.timeoutMs ?? GENERATION_TIMEOUT_MS,
      options.greedy ?? true,
    );
    const parsed = JSON.parse(reply.text) as ChatBody;
    const choice = parsed.choices?.[0];
    if (parsed.error || !choice) {
      throw new Error(
        `chat: HTTP ${reply.status} ${JSON.stringify(parsed.error ?? parsed).slice(0, 200)}`,
      );
    }
    const message = choice.message ?? {};
    return {
      text: message.content ?? "",
      reasoning: message.reasoning_content ?? message.reasoning ?? "",
      promptTokens: parsed.usage?.prompt_tokens ?? 0,
      completionTokens: parsed.usage?.completion_tokens ?? 0,
      timings: parsed.timings ?? {},
      tokens: (choice.logprobs?.content ?? []).map(
        (entry): TokenLogprob => ({
          token: entry.token,
          logprob: entry.logprob,
          top: (entry.top_logprobs ?? []).map(({ token, logprob }) => ({ token, logprob })),
        }),
      ),
    };
  }

  async completion(prompt: string, options: CompletionOptions): Promise<CompletionReply> {
    const body = {
      prompt,
      n_predict: options.nPredict,
      ...(options.stop ? { stop: options.stop } : {}),
      ...GREEDY,
    };
    const reply = await this.post(
      "/completion",
      body,
      options.timeoutMs ?? GENERATION_TIMEOUT_MS,
      true,
    );
    const parsed = JSON.parse(reply.text) as CompletionBody;
    if (parsed.error || parsed.content === undefined) {
      throw new Error(
        `completion: HTTP ${reply.status} ${JSON.stringify(parsed.error ?? parsed).slice(0, 200)}`,
      );
    }
    return { text: parsed.content, timings: parsed.timings ?? {} };
  }

  /** the server's own count, never chars/4, which is off by 30% on code */
  async countTokens(content: string): Promise<number> {
    const reply = await this.post("/tokenize", { content }, GENERATION_TIMEOUT_MS, true);
    return (JSON.parse(reply.text) as { tokens: number[] }).tokens.length;
  }

  /** a POST, sent once more when its connection closed before any response while the server still
   *  answers /health and the request is `repeatable` (greedy with the prompt cache off, so the
   *  repeat yields the same tokens). A gate lost its humaneval leg to one reset keep-alive socket
   *  with the server healthy and mid-batch (2026-09-24); a server that is down still fails. */
  private async post(
    path: string,
    body: unknown,
    timeoutMs: number | false,
    repeatable: boolean,
  ): Promise<HttpResponse> {
    const send = () => this.http.request("POST", `${this.base}${path}`, { body, timeoutMs });
    try {
      return await send();
    } catch (error) {
      const closed = error instanceof Error && error.name === "ConnectionClosed";
      if (!closed || !repeatable || !(await this.healthy())) throw error;
      return send();
    }
  }

  async healthy(): Promise<boolean> {
    try {
      const reply = await this.http.request("GET", `${this.base}/health`, { timeoutMs: 3000 });
      return reply.status === 200;
    } catch {
      return false;
    }
  }

  async props(): Promise<{ model: string; slots: number }> {
    const reply = await this.http.request("GET", `${this.base}/props`, { timeoutMs: 3000 });
    const parsed = JSON.parse(reply.text) as { model_path?: string; total_slots?: number };
    const model = (parsed.model_path ?? "?").split("/").at(-1) ?? "?";
    return { model, slots: parsed.total_slots ?? 0 };
  }
}
