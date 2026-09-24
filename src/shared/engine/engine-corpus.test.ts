import { describe, expect, test } from "bun:test";
import { fakePorts } from "../../../test/fakes/index.ts";
import { engineCorpus } from "./engine-corpus.ts";

describe("engine corpus", () => {
  test("a fixed order under headers, the skip list honoured, cut at a line boundary at the cap, the same bytes twice", async () => {
    const p = fakePorts();
    const e = "/e";
    p.fs.put(`${e}/docs/build.md`, "# build\nline\n");
    p.fs.put(`${e}/docs/ops/README.md`, "generated\n");
    p.fs.put(`${e}/src/llama.cpp`, "int a;\n");
    p.fs.put(`${e}/src/notes.txt`, "no\n");
    p.fs.put(`${e}/tools/server/server.cpp`, "int s;\n");
    p.fs.put(`${e}/tools/server/webui/index.html`, "<html>");
    const full = await engineCorpus(p.fs, e, 10_000);
    expect(full).toBe(
      "===== docs/build.md =====\n# build\nline\n\n===== tools/server/server.cpp =====\nint s;\n\n===== src/llama.cpp =====\nint a;\n\n",
    );
    expect(await engineCorpus(p.fs, e, 10_000)).toBe(full);
    const cut = await engineCorpus(p.fs, e, 45); // the second file's header would pass 45: the text ends on the first file's last line
    expect(cut).toBe("===== docs/build.md =====\n# build\nline\n\n");
    expect(await engineCorpus(p.fs, e, 80)).toBe(
      "===== docs/build.md =====\n# build\nline\n\n===== tools/server/server.cpp =====\n",
    ); // 80 cuts inside the second file, after its header line (40 + 36)
    await expect(engineCorpus(p.fs, "/nothing", 100)).rejects.toThrow("no corpus files");
  });
});
