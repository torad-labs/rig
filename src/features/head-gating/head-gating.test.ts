import { describe, expect, test } from "bun:test";
import { putHead } from "../../../test/fakes/head-fixtures.ts";
import { type FakePorts, fakePorts } from "../../../test/fakes/index.ts";
import { loadEngine } from "../../shared/engine/engine.ts";
import { loadHead } from "../../shared/head/head.ts";
import { layoutAt } from "../../shared/layout.ts";
import { loadGates } from "./gates-config.ts";
import { RunGates } from "./head-gating.service.ts";
import { allProbes } from "./probes/index.ts";
import type { Probe } from "./probes/probe.ts";

const root = `${import.meta.dir}/../../..`;
const headToml = await Bun.file(`${root}/heads/bonsai-2-27b/head.toml`).text();
const gatesToml = await Bun.file(`${root}/heads/bonsai-2-27b/gates.toml`).text();
const engineToml = await Bun.file(`${root}/engine/engine.toml`).text();

/** a fake llama-server: the harm prompts refused by the base pack and not the served one, code
 *  identical, decode speeds per leg; which pack "runs" is read from the spawned argv */
function fakeServers(
  p: FakePorts,
  o: {
    servedRefuses?: boolean;
    servedBlank?: boolean;
    servedDecode?: number;
    codeDiffers?: boolean;
    codeTies?: boolean;
    codeFar?: boolean;
    draftDiffers?: boolean;
    draftTies?: boolean;
  } = {},
) {
  const pack = () =>
    (p.shell.spawned.at(-1)?.cmd[2] ?? "").includes("ablated") ? "served" : "base";
  const drafted = () => (p.shell.spawned.at(-1)?.cmd ?? []).includes("--spec-type");
  p.http.json(/\/health$/, { status: "ok" });
  p.http.on(/\/v1\/chat\/completions$/, (_u, body) => {
    const b = body as {
      messages: Array<{ content: string }>;
      max_tokens: number;
      logprobs?: boolean;
    };
    const prompt = b.messages[0]!.content;
    const isHarm = /lock|hotwire|paywall|phishing|thermite/i.test(prompt);
    const refuse = pack() === "base" || o.servedRefuses;
    const text = isHarm
      ? refuse
        ? "I cannot help with that."
        : o.servedBlank
          ? ""
          : "Sure. Step 1: insert the pick."
      : o.codeDiffers && pack() === "served"
        ? "def f(): return 2"
        : (o.codeTies || o.codeFar) && pack() === "served"
          ? "def f(): return 3"
          : (o.draftDiffers || o.draftTies) && drafted()
            ? "def f(): return 3"
            : "def f(): return 1";
    // one token per word; the plain leg (and the base pack) ranks the alternative " 3" far below
    // (differs) or a hair below (ties) its own " 1"
    const logprobs = b.logprobs
      ? {
          content: text.split(/(?= )/).map((token) => ({
            token,
            logprob: -0.01,
            top_logprobs: [
              { token, logprob: -0.01 },
              ...(token === " 1"
                ? [{ token: " 3", logprob: o.draftTies || o.codeTies ? -0.05 : -2.0 }]
                : []),
            ],
          })),
        }
      : undefined;
    const timings = drafted()
      ? { predicted_per_second: 93.5, draft_n: 100, draft_n_accepted: 50 }
      : { predicted_per_second: 71.8 };
    return {
      status: 200,
      text: JSON.stringify({
        choices: [{ message: { content: text }, logprobs }],
        usage: { prompt_tokens: 10, completion_tokens: 8 },
        timings,
      }),
    };
  });
  p.http.on(/\/completion$/, () => ({
    status: 200,
    text: JSON.stringify({
      content: "x",
      timings: { predicted_per_second: pack() === "served" ? (o.servedDecode ?? 79.0) : 79.4 },
    }),
  }));
}

async function setup() {
  const p = fakePorts();
  const layout = layoutAt("/r");
  putHead(p.fs, "/r", headToml);
  p.fs.put("/r/heads/bonsai-2-27b/gates.toml", gatesToml);
  p.fs.put("/r/engine/engine.toml", engineToml);
  const head = await loadHead(p.fs, layout, "bonsai-2-27b");
  const engine = await loadEngine(p.fs, layout);
  if (!head.ok || !engine.ok) throw new Error("fixture");
  p.gpu.card(1, { name: "NVIDIA GeForce RTX 5070 Ti", memoryMiB: 16303 });
  p.fs.put(`${engine.value.binDir("120")}/BUILD`, "x");
  p.fs.put(`${engine.value.binDir("120")}/llama-server`, "x");
  p.fs.put(head.value.sourcePath, "base");
  p.hasher.pinned.set(head.value.sourcePath, head.value.source.sha256);
  p.fs.put(head.value.servedPath, "served");
  p.hasher.pinned.set(head.value.servedPath, head.value.served.sha256);
  return {
    p,
    head: head.value,
    engine: engine.value,
    layout,
    uc: new RunGates(p, layout, engine.value, allProbes),
  };
}

describe("gates.toml", () => {
  test("the head's gates parse, and the gate card is not the serving card", async () => {
    const { p, head } = await setup();
    const g = await loadGates(p.fs, head);
    expect(g.ok && g.value.gate.gpu).toBe(1);
    p.fs.put("/r/heads/bonsai-2-27b/gates.toml", gatesToml.replace("gpu = 1 ", "gpu = 0 "));
    const bad = await loadGates(p.fs, head);
    expect(!bad.ok && bad.message).toContain("serves from");
  });

  // a tie_gap wide enough to call anything a tie is how this probe passes a pack it never
  // compared: the two cards measure 0.021 and 0.008-0.053, and one nat is already a served
  // token under 37% as likely as the base pack's own choice
  test("a tie_gap above one nat is refused, naming the card it sits on", async () => {
    const { p, head } = await setup();
    const cards = {
      capability: "tie_gap = 0.15    # nats the served token",
      speculative: "tie_gap = 0.15    # nats the drafted token",
    };
    for (const [card, line] of Object.entries(cards)) {
      p.fs.put(
        "/r/heads/bonsai-2-27b/gates.toml",
        gatesToml.replace(line, line.replace("0.15", "1.5")),
      );
      const bad = await loadGates(p.fs, head);
      expect(!bad.ok && bad.message).toContain(`${card}.tie_gap`);
    }
  });
});

describe("gate", () => {
  test("refusal + decode: legs on the gate card and port, one pack each, evidence written, PASS", async () => {
    const { p, head, uc, engine } = await setup();
    fakeServers(p);
    const r = await uc.run(head, { only: ["refusal", "decode"] });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value.probes).toEqual([
      {
        name: "refusal",
        pass: true,
        summary: "refusals base 6/6 → served 0/6; capability 5/5 identical",
      },
      { name: "decode", pass: true, summary: "base 79.4 / served 79.0 / base 79.4 tok/s" },
    ]);
    const legs = p.shell.spawned.map((s) => [
      s.cmd[2]!.split("/").at(-1),
      s.opts?.env?.CUDA_VISIBLE_DEVICES,
      s.cmd[s.cmd.indexOf("--port") + 1],
    ]);
    expect(legs).toEqual([
      ["Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf", "1", "8098"],
      ["Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf", "1", "8098"],
      ["Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf", "1", "8098"],
      ["Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf", "1", "8098"],
      ["Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf", "1", "8098"],
    ]);
    expect(p.shell.spawned[0]!.cmd).toContain("--kv-mean-center"); // the head's runtime args, assets resolved
    expect(p.shell.spawned[0]!.cmd).toContain(
      "/r/heads/bonsai-2-27b/assets/kv-mean-center-PQ2_0.gguf",
    );
    const summary = JSON.parse(p.fs.text(`${r.value.dir}/summary.json`)!);
    expect(summary).toMatchObject({ head: "bonsai-2-27b", engine: `${engine.sha7}`, pass: true });
    expect(p.fs.text(`${r.value.dir}/gate.log`)).toContain("PASS refusal");
    expect(await p.fs.exists(`${r.value.dir}/refusal.json`)).toBe(true);
  });
  test("a served pack that still refuses, or moves the code, FAILS the refusal probe with exit 1 and the run kept", async () => {
    let s = await setup();
    fakeServers(s.p, { servedRefuses: true });
    let r = await s.uc.run(s.head, { only: ["refusal"] });
    expect(!r.ok && r.code).toBe(1);
    expect(!r.ok && r.message).toContain("refusal FAILED");
    s = await setup();
    fakeServers(s.p, { codeDiffers: true });
    r = await s.uc.run(s.head, { only: ["refusal"] });
    expect(!r.ok && r.message).toContain("refusal FAILED");
  });
  test("a served pack that answers the harm prompts with nothing FAILS: a blank is not an unlocked answer", async () => {
    const s = await setup();
    fakeServers(s.p, { servedBlank: true });
    const r = await s.uc.run(s.head, { only: ["refusal"] });
    expect(!r.ok && r.message).toContain("refusal FAILED");
    const log = !r.ok && r.message.slice(r.message.indexOf("— ") + 2);
    expect(s.p.fs.text(log || "")).toContain("harm0  base REFUSE  served EMPTY");
  });
  test("capability answers parting at a near-tie in the base pack's own ranking PASS; ranked further down FAIL", async () => {
    let s = await setup();
    fakeServers(s.p, { codeTies: true });
    let r = await s.uc.run(s.head, { only: ["refusal"] });
    expect(r.ok).toBe(true);
    if (r.ok)
      expect(r.value.probes[0]!.summary).toBe(
        "refusals base 6/6 → served 0/6; capability 0/5 identical (5 at a near-tie)",
      );
    expect(s.p.fs.text(`${r.ok ? r.value.dir : ""}/gate.log`)).toContain(
      'cap0 near-tie at token 3 (base " 1", served " 3"), 0.040 nats below',
    );
    s = await setup();
    fakeServers(s.p, { codeFar: true });
    r = await s.uc.run(s.head, { only: ["refusal"] });
    expect(!r.ok && r.message).toContain("refusal FAILED");
  });
  test("a served pack that decodes slower than the base's A-A spread FAILS decode", async () => {
    const { p, head, uc } = await setup();
    fakeServers(p, { servedDecode: 70 });
    const r = await uc.run(head, { only: ["decode"] });
    expect(!r.ok && r.message).toContain("decode FAILED");
  });
  test("refuses the serving card while it serves, a missing build, and an unpinned pack before starting anything", async () => {
    const { p, head, uc, engine } = await setup();
    fakeServers(p);
    p.http.json(/8099\/health$/, { status: "ok" });
    let r = await uc.run(head, { only: ["refusal"], gpu: 0 });
    expect(!r.ok && r.code).toBe(2);
    p.http.on(/8099\/health$/, () => {
      throw new Error("ECONNREFUSED");
    });
    await p.fs.remove(`${engine.binDir("120")}/BUILD`);
    r = await uc.run(head, { only: ["refusal"] });
    expect(!r.ok && r.message).toContain("no complete build");
    p.fs.put(`${engine.binDir("120")}/BUILD`, "x");
    p.hasher.pinned.set(head.servedPath, "0".repeat(64));
    r = await uc.run(head, { only: ["refusal"] });
    expect(!r.ok && r.message).toContain("not the pinned bytes (sha256 differs)");
    expect(p.shell.spawned).toEqual([]);
  });
  test("depth runs llama-bench on the card with the head's cache types and is measured, never judged", async () => {
    const { p, head, uc } = await setup();
    fakeServers(p);
    p.shell.on(/llama-bench/, {
      code: 0,
      stdout: "| model | test | t/s |\n| x | pp512 | 1382 |\n",
      stderr: "",
    });
    p.http.on(/8099\/health$/, () => {
      throw new Error("ECONNREFUSED");
    });
    const r = await uc.run(head, { only: ["depth"], gpu: 0 }); // GPU 0 is fine while nothing serves on :8099
    expect(r.ok && r.value.probes[0]).toEqual({
      name: "depth",
      pass: "measured",
      summary: "5 depths x pp512/tg64, 3 repeats",
    });
    const call = p.shell.calls.find((c) => c[0]!.endsWith("llama-bench"))!;
    expect(call.slice(1)).toEqual([
      "-m",
      head.servedPath,
      "-ngl",
      "99",
      "-fa",
      "1",
      "-ctk",
      "q4_0",
      "-ctv",
      "q4_0",
      "-p",
      "512",
      "-n",
      "64",
      "-d",
      "0,16384,65536,131072,261874",
      "-r",
      "3",
      "-o",
      "md",
    ]);
    expect(p.fs.text(`${r.ok ? r.value.dir : ""}/gate.log`)).toContain("MEASURED depth");
  });
  test("live probes need --live and a healthy head; --live alone adds them to the server probes", async () => {
    const { p, head, uc } = await setup();
    fakeServers(p);
    let r = await uc.run(head, { only: ["sessions"] });
    expect(!r.ok && r.message).toContain("--live");
    p.http.on(/8099\/health$/, () => {
      throw new Error("ECONNREFUSED");
    });
    r = await uc.run(head, { only: ["concurrency"], live: true });
    expect(!r.ok && r.message).toContain("no healthy head at http://127.0.0.1:8099");
    r = await uc.run(head, { only: ["nope"] });
    expect(!r.ok && r.code).toBe(64);
  });
  test("speculative: the served pack with and without its draft head — identical answers, the draft ran, not slower", async () => {
    const { p, head, uc } = await setup();
    fakeServers(p);
    const r = await uc.run(head, { only: ["speculative"] });
    expect(r.ok && r.value.probes).toEqual([
      {
        name: "speculative",
        pass: true,
        summary: "3/3 identical (0 at a near-tie), 50 % of 300 drafted accepted, 71.8 → 93.5 tok/s",
      },
    ]);
    expect(
      p.shell.spawned.map((s) => [s.cmd[2]!.split("/").at(-1), s.cmd.includes("--spec-type")]),
    ).toEqual([
      ["Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf", false],
      ["Ternary-Bonsai-2-27B-PQ2_0-MTP-ablated-rc010-draft-r2.gguf", true],
    ]);
    // shown able to fail: drafted answers the pack did not choose
    const bad = await setup();
    fakeServers(bad.p, { draftDiffers: true });
    const b = await bad.uc.run(bad.head, { only: ["speculative"] });
    expect(!b.ok && b.message).toContain("speculative FAILED");
    const report = bad.p.fs.text(b.ok ? "" : b.message.split(" — ")[1]!);
    expect(report).toContain("FAIL speculative: 0/3 identical (0 at a near-tie)");
    expect(report).toContain(
      'DIFFERS at token 3: plain " 1" vs drafted " 3", 1.990 nats below the pack\'s top-1',
    );
    // a divergence the pack itself rates a near-tie is the verify batch's numerics, not a wrong token
    const tied = await setup();
    fakeServers(tied.p, { draftTies: true });
    const t = await tied.uc.run(tied.head, { only: ["speculative"] });
    expect(t.ok && t.value.probes[0]?.summary).toBe(
      "0/3 identical (3 at a near-tie), 50 % of 300 drafted accepted, 71.8 → 93.5 tok/s",
    );
  });
  test("a head serving its source pack (no private adapter) cannot be asked for the refusal probe", async () => {
    const { p, head, uc } = await setup();
    fakeServers(p);
    await p.fs.remove(head.path("assets/lora/bonsai-abliterate-lora.gguf"));
    await p.fs.remove(head.servedPath);
    const clone = await loadHead(p.fs, layoutAt("/r"), "bonsai-2-27b");
    if (!clone.ok) throw new Error(clone.message);
    expect(clone.value.undrived).toContain("assets/lora/bonsai-abliterate-lora.gguf");
    const r = await uc.run(clone.value, { only: ["refusal"] });
    // the undrived reason, never the generic "no [derive] step" wording — that one is false here:
    // the head DOES declare a [derive] step, this machine just cannot run it
    expect(!r.ok && r.message).toBe(
      `refusal does not apply to bonsai-2-27b: ${clone.value.undrived}`,
    );
    expect(!r.ok && r.message).not.toContain("no [derive] step");
  });
  test("a head without a draft head cannot be asked for the speculative probe by name", async () => {
    const { p, uc } = await setup();
    fakeServers(p);
    p.fs.put(
      "/r/heads/bonsai-2-27b/head.toml",
      headToml.replace(/\[speculative\][\s\S]*?\n\n/, ""),
    );
    const bare = await loadHead(p.fs, layoutAt("/r"), "bonsai-2-27b");
    if (!bare.ok) throw new Error(bare.message);
    const r = await uc.run(bare.value, { only: ["speculative"] });
    expect(!r.ok && r.message).toBe(
      "speculative does not apply to bonsai-2-27b: the head declares no draft head ([speculative])",
    );
  });
  test("a probe that does not apply is skipped, not failed, and named (probe and reason) in summary.json and gate.log", async () => {
    const { p, head, layout, engine } = await setup();
    const applies: Probe = {
      name: "applies",
      needs: "card",
      run: async () => ({ name: "applies", pass: true, summary: "ok", lines: [], data: null }),
    };
    const neverApplies: Probe = {
      name: "never-applies",
      needs: "card",
      applies: () => "fixture: never applies",
      run: async () => {
        throw new Error("a skipped probe must never run");
      },
    };
    const uc = new RunGates(p, layout, engine, [applies, neverApplies]);
    const r = await uc.run(head, {});
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value.probes).toEqual([{ name: "applies", pass: true, summary: "ok" }]);
    expect(r.value.skipped).toEqual([{ probe: "never-applies", reason: "fixture: never applies" }]);
    const summary = JSON.parse(p.fs.text(`${r.value.dir}/summary.json`)!);
    expect(summary.skipped).toEqual([{ probe: "never-applies", reason: "fixture: never applies" }]);
    expect(p.fs.text(`${r.value.dir}/gate.log`)).toContain(
      "== never-applies: SKIPPED, fixture: never applies",
    );
  });
});
