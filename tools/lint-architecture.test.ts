import { describe, expect, test } from "bun:test";
import { check, type Workspace } from "./lint-architecture.ts";

const ws = (
  name: string,
  dir: string,
  files: Record<string, string>,
  dependencies: Record<string, string> = {},
  devDependencies: Record<string, string> = {},
): Workspace => ({
  name,
  dir,
  dependencies,
  devDependencies,
  files: new Map(Object.entries(files).map(([path, text]) => [`${dir}/${path}`, text])),
});
const core = ws("@rig/core", "packages/core", { "src/index.ts": "export const x = 1;" });
const testing = ws(
  "@rig/testing",
  "packages/testing",
  { "src/index.ts": "" },
  { "@rig/core": "workspace:*" },
);

describe("the architecture lint", () => {
  test("the tree as layered passes", () => {
    const head = ws(
      "@rig/head",
      "packages/head",
      {
        "src/head.ts": 'import { x } from "@rig/core";\nimport { y } from "./y.ts";',
        "src/head.test.ts": 'import { fakePorts } from "@rig/testing";',
      },
      { "@rig/core": "workspace:*" },
      { "@rig/testing": "workspace:*" },
    );
    expect(check([core, testing, head])).toEqual([]);
  });
  test("a dependency above the workspace's layer fails by name", () => {
    const head = ws("@rig/core", "packages/core", {}, { "@rig/head": "workspace:*" });
    expect(check([head, ws("@rig/head", "packages/head", {})])).toEqual([
      "@rig/core depends on @rig/head: its layer allows nothing",
    ]);
  });
  test("a workspace with no row in LAYERS fails: nobody decided what it may use", () => {
    expect(check([ws("@rig/new", "packages/new", {})])).toEqual([
      "@rig/new (packages/new) has no row in LAYERS: decide what it may depend on",
    ]);
  });
  test("a relative import out of the workspace fails", () => {
    const pack = ws("@rig/pack", "packages/pack", {
      "src/derive/x.ts": 'import { head } from "../../../head/src/head.ts";',
    });
    expect(check([pack])).toEqual([
      "packages/pack/src/derive/x.ts → ../../../head/src/head.ts: a relative import stays inside packages/pack",
    ]);
  });
  test("code that ships importing a devDependency fails; its test may", () => {
    const serve = ws(
      "@rig/serve",
      "packages/serve",
      {
        "src/serving/plan.ts": 'import { fakePorts } from "@rig/testing";',
        "src/serving/plan.test.ts": 'import { fakePorts } from "@rig/testing";',
      },
      {},
      { "@rig/testing": "workspace:*" },
    );
    expect(check([serve, testing, core])).toEqual([
      "packages/serve/src/serving/plan.ts → @rig/testing: code that ships imports only @rig/serve's dependencies (@rig/testing is a devDependency)",
    ]);
  });
  test("a devDependency beyond test support fails", () => {
    const gate = ws("@rig/gate", "packages/gate", {}, {}, { "@rig/rental": "workspace:*" });
    expect(check([gate, ws("@rig/rental", "packages/rental", {})])).toEqual([
      "@rig/gate devDepends on @rig/rental: a test leans only on @rig/testing, @rig/adapters",
    ]);
  });
  test("in the cli, a command constructing the machine fails; main.ts does it", () => {
    const cli = ws(
      "@rig/cli",
      "apps/cli",
      {
        "src/main.ts": 'import { realPorts } from "@rig/adapters";',
        "src/commands/up.command.ts": 'import { realPorts } from "@rig/adapters";',
      },
      { "@rig/adapters": "workspace:*" },
    );
    expect(
      check([
        cli,
        ws("@rig/adapters", "packages/adapters", {}, { "@rig/core": "workspace:*" }),
        core,
      ]),
    ).toEqual([
      "apps/cli/src/commands/up.command.ts → @rig/adapters: only the composition root constructs the machine",
    ]);
  });
});
