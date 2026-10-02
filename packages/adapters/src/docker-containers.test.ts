import { describe, expect, test } from "bun:test";
import { FakeShell } from "@rig/testing";
import { DockerContainers } from "./docker-containers.ts";

describe("DockerContainers", () => {
  test("a run gets the card as its CDI device and each mount read-only, and is removed after", async () => {
    const shell = new FakeShell().on(/^docker run/, { code: 0, stdout: "", stderr: "" });
    await new DockerContainers(shell).run("r:t", ["rig", "build", "h"], {
      gpu: 1,
      mounts: { "/host/smoke.gguf": "/smoke.gguf" },
    });
    expect(shell.calls[0]).toEqual([
      "docker",
      "run",
      "--rm",
      "--device",
      "nvidia.com/gpu=1",
      "-v",
      "/host/smoke.gguf:/smoke.gguf:ro",
      "r:t",
      "rig",
      "build",
      "h",
    ]);
  });
  test("an image is saved to the tarball named", async () => {
    const shell = new FakeShell().on(/^docker save/, { code: 0, stdout: "", stderr: "" });
    await new DockerContainers(shell).save("r:t", "/w/image.tar");
    expect(shell.calls[0]).toEqual(["docker", "save", "-o", "/w/image.tar", "r:t"]);
  });
});
