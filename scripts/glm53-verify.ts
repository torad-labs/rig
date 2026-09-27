// Run after the separate GLM tunnel and admission proxy are healthy.
export {};

const base = process.env.GLM53_VERIFY_URL ?? "http://127.0.0.1:8102";
const targetTokens = 410_000;
const timeout = 45 * 60 * 1000;
let servedModel = "";

async function post(path: string, body: unknown) {
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeout),
    timeout: false,
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status} ${text.slice(0, 500)}`);
  return JSON.parse(text);
}

async function count(content: string): Promise<number> {
  const result = await post("/tokenize", { content });
  if (!Array.isArray(result.tokens)) throw new Error("Server did not return a token count");
  return result.tokens.length;
}

async function chat(content: string, extra: Record<string, unknown> = {}) {
  const start = performance.now();
  const result = await post("/v1/chat/completions", {
    model: servedModel,
    messages: [{ role: "user", content }],
    max_tokens: 8192,
    temperature: 0,
    cache_prompt: false,
    ...extra,
  });
  if (!result.choices?.[0])
    throw new Error(`No completion: ${JSON.stringify(result).slice(0, 500)}`);
  return { result, seconds: (performance.now() - start) / 1000 };
}

const modelsResponse = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(10_000) });
if (!modelsResponse.ok) throw new Error(`Models endpoint: HTTP ${modelsResponse.status}`);
const models = (await modelsResponse.json()) as { data?: Array<{ id: string }> };
servedModel = models.data?.find((model) => model.id.includes("IQ3_XXS"))?.id ?? "";
if (!servedModel)
  throw new Error(`Unexpected model identity: ${JSON.stringify(models).slice(0, 500)}`);
console.log(
  JSON.stringify({ stage: "model", models: models.data?.map((model: { id: string }) => model.id) }),
);

const short = await chat("Reply with the single word READY.");
const shortAnswer = short.result.choices[0].message?.content ?? "";
if (!shortAnswer.includes("READY"))
  throw new Error(`Short completion failed: ${shortAnswer.slice(0, 500)}`);
console.log(
  JSON.stringify({
    stage: "short",
    answer: shortAnswer,
    usage: short.result.usage,
    timings: short.result.timings,
    seconds: short.seconds,
  }),
);

const tool = await chat("Call emit_status with status READY and count 7. Do not answer in prose.", {
  tools: [
    {
      type: "function",
      function: {
        name: "emit_status",
        description: "Report the current test status",
        parameters: {
          type: "object",
          properties: { status: { type: "string" }, count: { type: "integer" } },
          required: ["status", "count"],
        },
      },
    },
  ],
  tool_choice: "required",
});
const call = tool.result.choices[0].message?.tool_calls?.find(
  (entry: { function?: { name?: string } }) => entry.function?.name === "emit_status",
);
const args = call && JSON.parse(call.function.arguments);
const toolPassed = args?.status === "READY" && args?.count === 7;
console.log(
  JSON.stringify({
    stage: "tool",
    pass: toolPassed,
    name: call?.function.name,
    arguments: args,
    raw: toolPassed ? undefined : tool.result.choices[0].message?.content?.slice(0, 500),
    usage: tool.result.usage,
  }),
);

const bench = await chat(
  "Write a detailed explanation of how binary search works, including several examples.",
  { max_tokens: 1024 },
);
const generated = bench.result.usage?.completion_tokens ?? 0;
if (generated < 200) throw new Error(`Benchmark produced too few tokens: ${generated}`);
console.log(
  JSON.stringify({
    stage: "benchmark",
    usage: bench.result.usage,
    timings: bench.result.timings,
    seconds: bench.seconds,
    generatedTokensPerSecond: generated / bench.seconds,
  }),
);

const paragraph =
  "The field log lists calibration readings for the river station, records shifts in wind and temperature, and preserves each observation in the archive.\n";
let repetitions = Math.ceil(targetTokens / Math.max(1, await count(paragraph))) + 1000;
let prompt = "";
let counted = 0;
for (let attempt = 0; attempt < 8; attempt++) {
  const quarter = Math.floor(repetitions / 4);
  prompt =
    paragraph.repeat(quarter) +
    "NOTE: the east station code is RIDGE-4267.\n" +
    paragraph.repeat(quarter * 2) +
    "NOTE: the west station code is BIRCH-8314.\n" +
    paragraph.repeat(repetitions - 3 * quarter) +
    "What are the east and west station codes? Answer with only the two codes.";
  counted = await count(prompt);
  if (counted >= targetTokens) break;
  repetitions = Math.ceil(((repetitions * targetTokens) / counted) * 1.01);
}
if (counted < targetTokens || counted > 500_000)
  throw new Error(`Invalid long prompt token count: ${counted}`);
const long = await chat(prompt);
const answer = long.result.choices[0].message?.content ?? "";
const used = long.result.usage?.prompt_tokens ?? 0;
if (used < 400_000 || !answer.includes("RIDGE-4267") || !answer.includes("BIRCH-8314")) {
  throw new Error(
    `Long-context retrieval failed: ${JSON.stringify({ counted, used, answer: answer.slice(0, 600) })}`,
  );
}
console.log(
  JSON.stringify({
    stage: "long",
    counted,
    usage: long.result.usage,
    timings: long.result.timings,
    answer: answer.slice(0, 500),
    seconds: long.seconds,
    promptTokensPerSecond: used / long.seconds,
  }),
);
if (!toolPassed) throw new Error("GLM did not return a structured tool call");
