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
  test("a run as the caller, capped, with its environment and a writable mount beside the read-only ones", async () => {
    let timeoutMs: number | undefined;
    const shell = new FakeShell().on(/^docker run/, (_cmd, opts) => {
      timeoutMs = opts?.timeoutMs;
      return { code: 0, stdout: "", stderr: "" };
    });
    await new DockerContainers(shell).run("r:t", ["rig", "build"], {
      gpu: 0,
      asCaller: true,
      limits: { memory: "14g", cpus: 6 },
      env: { HOME: "/tmp", RIG_ROOT: "/rig" },
      mounts: { "/repo": "/rig" },
      writable: { "/repo/local/prebuilt": "/rig/local" },
      timeoutMs: 7_200_000,
    });
    expect(shell.calls[0]).toEqual([
      "docker",
      "run",
      "--rm",
      "--device",
      "nvidia.com/gpu=0",
      "--user",
      `${process.getuid?.()}:${process.getgid?.()}`,
      "--memory",
      "14g",
      "--memory-swap",
      "14g",
      "--cpus",
      "6",
      "-e",
      "HOME=/tmp",
      "-e",
      "RIG_ROOT=/rig",
      "-v",
      "/repo:/rig:ro",
      "-v",
      "/repo/local/prebuilt:/rig/local",
      "r:t",
      "rig",
      "build",
    ]);
    expect(timeoutMs).toBe(7_200_000);
  });
  test("an image is saved to the tarball named", async () => {
    const shell = new FakeShell().on(/^docker save/, { code: 0, stdout: "", stderr: "" });
    await new DockerContainers(shell).save("r:t", "/w/image.tar");
    expect(shell.calls[0]).toEqual(["docker", "save", "-o", "/w/image.tar", "r:t"]);
  });
});
