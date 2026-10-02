import { describe, expect, test } from "bun:test";
import { commonDir, mirrorDirs, readDepFile, readNinjaDeps } from "./relink.ts";

describe("relink's readings", () => {
  test("ninja -t deps: each object's recorded files; an object with none recorded is absent", () => {
    const text = [
      "ggml/a.cu.o: #deps 2, deps mtime 1759339000000000000 (VALID)",
      "    /src/ggml/a.cu",
      "    /src/ggml/fattn-common.cuh",
      "",
      "ggml/b.cu.o: #deps 1, deps mtime 1759339000000000000 (STALE)",
      "    /src/ggml/b.cu",
      "",
      "ggml/c.cu.o: deps not found",
      "",
    ].join("\n");
    const deps = readNinjaDeps(text);
    expect([...deps.entries()]).toEqual([
      ["ggml/a.cu.o", ["/src/ggml/a.cu", "/src/ggml/fattn-common.cuh"]],
      ["ggml/b.cu.o", ["/src/ggml/b.cu"]],
    ]);
  });
  test("a dependency file's prerequisites, continued lines and escaped spaces read", () => {
    const text =
      "obj.o: /m/a.cu \\\n /m/fattn-common.cuh /m/my\\ dir/h.cuh \\\n /usr/include/x.h\n";
    expect(readDepFile(text)).toEqual([
      "/m/a.cu",
      "/m/fattn-common.cuh",
      "/m/my dir/h.cuh",
      "/usr/include/x.h",
    ]);
  });
  test("the mirror holds an include directory a chosen header is found through, and no other", () => {
    // ggml-cuda.cu has `#include "ggml-cuda/mmq.cuh"`, found through -I ggml/src (ninja's ggml/src/ggml-cuda/..)
    const ggmlCuda =
      "nvcc -I/src/ggml/src/../include -I/src/ggml/src/ggml-cuda/.. -c /src/ggml/src/ggml-cuda/ggml-cuda.cu " +
      "-o ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o";
    const instance =
      "nvcc -I/src/ggml/src/../include -c /src/ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu " +
      "-o ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q8_0.cu.o";
    const head = ["ggml/src/ggml-cuda/mmq.cuh"];
    expect(commonDir(mirrorDirs("/src", "/build", head, [ggmlCuda, instance]))).toBe("ggml/src");
    // the include directory holds no chosen path (ggml/include), or is the source root itself: the mirror stays small
    expect(commonDir(mirrorDirs("/src", "/build", head, [instance]))).toBe("ggml/src/ggml-cuda");
    expect(
      commonDir(mirrorDirs("/src", "/build", head, [`nvcc -I/src -I. ${instance.slice(5)}`])),
    ).toBe("ggml/src/ggml-cuda");
  });
  test("the deepest directory every path is in", () => {
    expect(
      commonDir([
        "ggml/src/ggml-cuda",
        "ggml/src/ggml-cuda/template-instances",
        "ggml/src/ggml-cuda",
      ]),
    ).toBe("ggml/src/ggml-cuda");
    expect(commonDir(["ggml/src/ggml-cuda", "ggml/include"])).toBe("ggml");
    expect(commonDir(["ggml", "tests"])).toBe(".");
  });
});
