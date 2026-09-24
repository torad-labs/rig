// What a probe may ask a llama-server: the domain's view of the API, implemented by
// infrastructure/LlamaClient.ts. Greedy by default so the same weights give the same tokens.
export interface Timings {
  predicted_per_second?: number;
  prompt_per_second?: number;
  draft_n?: number;
  draft_n_accepted?: number;
}
/** one ranked alternative for a generated position */
export interface TokenChoice {
  token: string;
  logprob: number;
}
/** a generated token with the alternatives the server ranked at that position, best first */
export interface TokenLogprob extends TokenChoice {
  top: TokenChoice[];
}
export interface ChatReply {
  text: string;
  reasoning: string;
  promptTokens: number;
  completionTokens: number;
  timings: Timings;
  /** one entry per generated token when `topLogprobs` was asked for, else empty */
  tokens: TokenLogprob[];
}
export interface CompletionReply {
  text: string;
  timings: Timings;
}
export interface ChatOptions {
  maxTokens: number;
  thinking?: boolean;
  greedy?: boolean;
  timeoutMs?: number | false;
  /** ask for this many ranked alternatives per generated token */
  topLogprobs?: number;
}
export interface CompletionOptions {
  nPredict: number;
  stop?: string[];
  timeoutMs?: number | false;
}

export interface HeadClient {
  readonly base: string;
  chat(prompt: string, o: ChatOptions): Promise<ChatReply>;
  completion(prompt: string, o: CompletionOptions): Promise<CompletionReply>;
  countTokens(content: string): Promise<number>;
  healthy(): Promise<boolean>;
  props(): Promise<{ model: string; slots: number }>;
}
