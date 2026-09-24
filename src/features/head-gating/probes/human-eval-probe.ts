// HumanEval-164 pass@1, greedy, completion mode (no chat template), base pack and served pack on
// the same card with `workers` in flight against -np workers: the capability gate the refusal
// gate cannot give. Generated programs run under python3 with a 10 s timeout in the run
// directory. The data is openai/human-eval's, fetched to local/gate-runs/ on first use.
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import type { HeadClient } from "../head-client.ts";
import type { Probe, ProbeContext } from "./probe.ts";

export interface Problem {
  task_id: string;
  prompt: string;
  test: string;
  entry_point: string;
}

export interface HumanEvalResult {
  passed: number;
  total: number;
  generationSecs: number;
  results: Array<{ task_id: string; passed: boolean }>;
}

export async function loadProblems(ctx: ProbeContext): Promise<Problem[]> {
  const jsonl = join(ctx.gateRunsDir, "HumanEval.jsonl");
  if (!(await ctx.fs.exists(jsonl))) {
    const gz = `${jsonl}.gz`;
    ctx.log.info(`  fetching ${ctx.gates.humaneval.url}`);
    const fetched = await ctx.shell.run(["curl", "-fsSL", "-o", gz, ctx.gates.humaneval.url], {
      timeoutMs: 120_000,
    });
    if (fetched.code !== 0) throw new Error(`could not fetch HumanEval: ${fetched.stderr.trim()}`);
    await ctx.fs.writeBytes(jsonl, new Uint8Array(gunzipSync(await ctx.fs.readBytes(gz))));
    await ctx.fs.remove(gz);
  }
  return (await ctx.fs.readText(jsonl))
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as Problem);
}

/** every problem completed with `workers` in flight, then each program run under python3 */
export async function evaluate(
  client: HeadClient,
  ctx: ProbeContext,
  label: string,
  problems: Problem[],
): Promise<HumanEvalResult> {
  const { workers, n_predict, stop } = ctx.gates.humaneval;
  const started = Date.now();
  const completions = new Array<string>(problems.length);
  let next = 0;
  const worker = async () => {
    for (let index = next++; index < problems.length; index = next++) {
      const problem = problems[index];
      if (!problem) break;
      const reply = await client.completion(problem.prompt, { nPredict: n_predict, stop });
      completions[index] = reply.text;
    }
  };
  await Promise.all(Array.from({ length: workers }, worker));
  const generationSecs = (Date.now() - started) / 1000;

  const dir = join(ctx.runDir, `humaneval-${label}`);
  await ctx.fs.mkdirp(dir);
  const results: HumanEvalResult["results"] = [];
  for (const [index, problem] of problems.entries()) {
    const program = `${problem.prompt}${completions[index]}\n\n${problem.test}\n\ncheck(${problem.entry_point})\n`;
    const path = join(dir, `${problem.task_id.replace("/", "_")}.py`);
    await ctx.fs.writeText(path, program);
    const run = await ctx.shell.run(["python3", path], { cwd: dir, timeoutMs: 10_000 });
    results.push({ task_id: problem.task_id, passed: run.code === 0 });
  }
  return {
    passed: results.filter((result) => result.passed).length,
    total: problems.length,
    generationSecs,
    results,
  };
}

export const humanevalProbe: Probe = {
  name: "humaneval",
  needs: "server",
  async run(ctx) {
    const problems = await loadProblems(ctx);
    const { workers, tolerance } = ctx.gates.humaneval;
    const leg = { ctx: 32768, slots: workers };
    const base = await ctx.server.leg(
      { label: "humaneval-base", pack: ctx.head.sourcePath, ...leg },
      (client) => evaluate(client, ctx, "base", problems),
    );
    const served = await ctx.server.leg(
      { label: "humaneval-served", pack: ctx.head.servedPath, ...leg },
      (client) => evaluate(client, ctx, "served", problems),
    );
    const line = (label: string, result: HumanEvalResult) =>
      `${label}: pass@1 = ${result.passed}/${result.total} = ${(result.passed / result.total).toFixed(3)}   (generation ${result.generationSecs.toFixed(0)} s)`;
    return {
      name: "humaneval",
      pass: served.passed >= base.passed - tolerance,
      summary: `pass@1 base ${base.passed}/${base.total}, served ${served.passed}/${served.total}`,
      lines: [line("base  ", base), line("served", served)],
      data: { base, served },
    };
  },
};
