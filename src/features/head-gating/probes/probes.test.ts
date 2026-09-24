import { describe, expect, test } from "bun:test";
import { FakeHttp } from "../../../../test/fakes/index.ts";
import { LlamaClient } from "../llama-client.ts";
import { abaVerdict } from "./decode-probe.ts";
import { countDoubled, countRepeats } from "./fluency-probe.ts";
import { grow, plant } from "./needle-probe.ts";
import { capabilityVerdicts, countRefusals, harmVerdict, parity } from "./refusal-probe.ts";
import { firstDivergence, speculativeVerdict } from "./speculative-probe.ts";

describe("fluency counters", () => {
  test("doubled words and immediate repeats, case-insensitive, one-letter words ignored for repeats", () => {
    expect(countDoubled("the the cat sat sat. A a b")).toEqual(["the", "sat", "A"]);
    expect(countRepeats("the the cat sat sat. a a")).toBe(1); // "sat" and "sat." differ; "a" is one letter
    expect(countDoubled("venv venv is fine")).toEqual(["venv"]);
    expect(countDoubled("")).toEqual([]);
  });
});

describe("refusal counters", () => {
  const pattern = "(I cannot|I can.t|I.m sorry|cannot provide)";
  test("counts answers matching the head's pattern; parity is positional equality", () => {
    expect(
      countRefusals(["I cannot help with that", "Sure: step 1", "I'm sorry, no"], pattern),
    ).toBe(2);
    expect(parity(["a", "b", "c"], ["a", "x", "c"])).toEqual([true, false, true]);
  });
  test("a blank answer is EMPTY, never a compliance, and two blank capability answers are not a match", () => {
    const refusal = /cannot|sorry/i;
    expect(["I cannot", "Sure: step 1", "", " \n"].map((a) => harmVerdict(a, refusal))).toEqual([
      "REFUSE",
      "comply",
      "EMPTY",
      "EMPTY",
    ]);
    const set = (capability: string[]) => ({ harm: [], capability, capabilityTokens: [] });
    expect(capabilityVerdicts(set(["", "def f"]), set(["", ""]), 0.15)).toEqual([
      { verdict: "EMPTY" },
      { verdict: "EMPTY" },
    ]);
  });
  test("a divergence carries both tokens, and one the logprobs cannot locate has no index", () => {
    const tok = (token: string, top: Array<[string, number]>) => ({
      token,
      logprob: top[0]![1],
      top: top.map(([t, logprob]) => ({ token: t, logprob })),
    });
    const base = {
      harm: [],
      capability: ["a The", "x"],
      capabilityTokens: [
        [
          tok("a", [["a", -0.1]]),
          tok(" The", [
            [" The", -0.687],
            [" A", -0.708],
          ]),
        ],
        [],
      ],
    };
    const served = {
      harm: [],
      capability: ["a A", "y"],
      capabilityTokens: [[tok("a", [["a", -0.1]]), tok(" A", [[" A", -0.6]])], []],
    };
    const [located, unlocated] = capabilityVerdicts(base, served, 0.15);
    expect(located).toEqual({
      verdict: "near-tie",
      index: 1,
      base: " The",
      served: " A",
      gap: expect.closeTo(0.021, 5),
    });
    expect(unlocated).toEqual({ verdict: "DIFFERS", index: null, base: "", served: "", gap: null });
  });
});

describe("decode A-B-A verdict", () => {
  test("the served leg must sit within the base's A-A spread plus the tolerance", () => {
    expect(abaVerdict(79.4, 79.0, 79.5, 5)).toBe(true); // measured 2026-09-19
    expect(abaVerdict(80.0, 76.5, 80.1, 0)).toBe(false); // the ordered A-then-B capgate saw, judged without tolerance
    expect(abaVerdict(80.0, 76.5, 80.1, 5)).toBe(true); // inside 5%
    expect(abaVerdict(80.0, 70.0, 80.1, 5)).toBe(false); // a real cost fails
  });
});

describe("speculative verdict", () => {
  const tok = (token: string, top: Array<[string, number]> = [[token, -0.01]]) => ({
    token,
    logprob: top[0]?.[1] ?? -0.01,
    top: top.map(([t, logprob]) => ({ token: t, logprob })),
  });
  const plainTokens = [
    tok("a"),
    tok(" b", [
      [" b", -0.3],
      [" c", -0.34],
      [" d", -2.1],
    ]),
    tok(" e"),
  ];
  const plain = {
    answers: ["a b e", "x"],
    tokens: [plainTokens, [tok("x")]],
    tps: 71.8,
    draftN: 0,
    accepted: 0,
  };
  const drafted = (answer: string, tokens: ReturnType<typeof tok>[], tps = 93.5) => ({
    answers: [answer, "x"],
    tokens: [tokens, [tok("x")]],
    tps,
    draftN: 300,
    accepted: 150,
  });

  test("the first divergence carries the drafted token's gap below the plain leg's top-1", () => {
    expect(firstDivergence(plainTokens, plainTokens)).toBeNull();
    expect(firstDivergence(plainTokens, [tok("a"), tok(" c"), tok(" z")])).toEqual({
      index: 1,
      plain: " b",
      drafted: " c",
      gap: expect.closeTo(0.04, 6),
    });
    // the pack never ranked the drafted token: no gap to measure
    expect(firstDivergence(plainTokens, [tok("a"), tok(" q")])?.gap).toBeNull();
    // one leg stopped early: the missing token is the divergence
    expect(firstDivergence(plainTokens, [tok("a"), tok(" b")])).toEqual({
      index: 2,
      plain: " e",
      drafted: "",
      gap: null,
    });
  });

  test("identical, or a near-tie at the first divergence; the draft ran; not slower — each shown able to fail", () => {
    const same = speculativeVerdict(plain, drafted("a b e", plainTokens), 1.0, 0.25);
    expect(same).toEqual({
      prompts: [
        { identical: true, divergence: null, tie: false },
        { identical: true, divergence: null, tie: false },
      ],
      ran: true,
      fast: true,
    });
    const tie = speculativeVerdict(
      plain,
      drafted("a c z", [tok("a"), tok(" c"), tok(" z")]),
      1.0,
      0.25,
    );
    expect(tie.prompts[0]).toMatchObject({ identical: false, tie: true });
    expect(tie.prompts[0]?.divergence?.index).toBe(1);
    // the same divergence judged by a tighter window fails
    expect(
      speculativeVerdict(plain, drafted("a c z", [tok("a"), tok(" c"), tok(" z")]), 1.0, 0.03)
        .prompts[0]?.tie,
    ).toBe(false);
    // a token the pack rated 2.1 nats down is a wrong token, not a tie
    expect(
      speculativeVerdict(plain, drafted("a d z", [tok("a"), tok(" d"), tok(" z")]), 1.0, 0.25)
        .prompts[0]?.tie,
    ).toBe(false);
    // different text and no token logprobs to locate the divergence: fails, never passes vacuously
    expect(
      speculativeVerdict({ ...plain, tokens: [[], []] }, drafted("a c z", []), 1.0, 0.25)
        .prompts[0],
    ).toEqual({ identical: false, divergence: null, tie: false });
    expect(
      speculativeVerdict(
        plain,
        { ...drafted("a b e", plainTokens), draftN: 0, accepted: 0 },
        1.0,
        0.25,
      ).ran,
    ).toBe(false); // the flags were there, no draft ran
    expect(speculativeVerdict(plain, drafted("a b e", plainTokens, 70.0), 1.0, 0.25).fast).toBe(
      false,
    ); // a host that loses to the draft rounds
  });
});

describe("llama client", () => {
  test("asks for ranked alternatives per token only when told to, and hands them back best first", async () => {
    const http = new FakeHttp();
    const bodies: unknown[] = [];
    http.on(/\/v1\/chat\/completions$/, (_u, body) => {
      bodies.push(body);
      const asked = (body as { logprobs?: boolean }).logprobs;
      return {
        status: 200,
        text: JSON.stringify({
          choices: [
            {
              message: { content: "Hi" },
              logprobs: !asked
                ? undefined
                : {
                    content: [
                      {
                        token: "Hi",
                        logprob: -0.07,
                        top_logprobs: [
                          { token: "Hi", logprob: -0.07, bytes: [72, 105] },
                          { token: "Hello", logprob: -3.4 },
                        ],
                      },
                    ],
                  },
            },
          ],
          usage: { prompt_tokens: 3, completion_tokens: 1 },
        }),
      };
    });
    const client = new LlamaClient(http, "http://x");
    const reply = await client.chat("hey", { maxTokens: 4, topLogprobs: 2 });
    expect(reply.tokens).toEqual([
      {
        token: "Hi",
        logprob: -0.07,
        top: [
          { token: "Hi", logprob: -0.07 },
          { token: "Hello", logprob: -3.4 },
        ],
      },
    ]);
    expect(bodies[0]).toMatchObject({ logprobs: true, top_logprobs: 2 });
    const bare = await client.chat("hey", { maxTokens: 4 });
    expect(bare.tokens).toEqual([]);
    expect(bodies[1]).not.toHaveProperty("logprobs");
  });
});

describe("needle haystack", () => {
  test("grows the filler to the target by the server's count and plants needles at line depths, deepest first", async () => {
    const http = new FakeHttp();
    http.on(/\/tokenize$/, (_u, body) => ({
      status: 200,
      text: JSON.stringify({
        tokens: new Array((body as { content: string }).content.length >> 1).fill(0),
      }),
    })); // 2 chars per token
    const client = new LlamaClient(http, "http://x");
    const hay = await grow(client, "abcd\n".repeat(10), 100); // 50 chars = 25 tokens per copy -> 4 copies
    expect(hay.length).toBe(200);
    const planted = plant("l0\nl1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9", [
      { depth: 0.2, secret: "S1", question: "q1" },
      { depth: 0.8, secret: "S2", question: "q2" },
    ]);
    const lines = planted.split("\n");
    // deepest first: S2 goes in at floor(10*0.8)=8, then S1 at floor(11*0.2)=2; each note is
    // "\nNOTE…\n", three lines once joined, so S1 reads at 3 and S2 at 8+1+3
    expect(lines.indexOf("NOTE: q1 is S1. Remember it.")).toBe(3);
    expect(lines.indexOf("NOTE: q2 is S2. Remember it.")).toBe(12);
  });
});
