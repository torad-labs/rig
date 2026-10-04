import { describe, expect, test } from "bun:test";
import { FakeShell } from "@rig/testing";
import { OpenSsh } from "./open-ssh.ts";

const target = {
  host: "203.0.113.7",
  port: 40174,
  user: "root",
  knownHosts: "/r/local/rented-box/boxes/1000/known_hosts",
};
/** OpenSSH 10.0p2's stderr, exit 255, for a host whose key is not the one the known_hosts file holds, run with rig's ssh
 *  options against a local sshd whose key was swapped (Oct 3); the banner's middle lines are cut */
const REFUSED = [
  "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
  "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @",
  "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
  "Offending ED25519 key in /r/local/rented-box/boxes/1000/known_hosts:1",
  "Host key for [203.0.113.7]:40174 has changed and you have requested strict checking.",
  "Host key verification failed.",
].join("\n");

describe("OpenSsh", () => {
  test("a refusal for a changed host key is told apart from every other failure", async () => {
    const shell = new FakeShell();
    const ssh = new OpenSsh(shell);
    shell.on(/^ssh /, { code: 255, stdout: "", stderr: REFUSED });
    expect(await ssh.run(target, "true")).toMatchObject({ code: 255, hostKeyChanged: true });
    for (const stderr of [
      "ssh: connect to host 203.0.113.7 port 40174: Connection refused",
      "root@203.0.113.7: Permission denied (publickey).",
    ]) {
      shell.on(/^ssh /, { code: 255, stdout: "", stderr });
      expect((await ssh.run(target, "true")).hostKeyChanged).toBeUndefined();
    }
    // a command of the box's own that prints the words exits as itself: only ssh's 255 is ssh's refusal
    shell.on(/^ssh /, { code: 1, stdout: "", stderr: REFUSED });
    expect((await ssh.run(target, "cat notes")).hostKeyChanged).toBeUndefined();
    shell.on(/^ssh /, { code: 0, stdout: "ok\n", stderr: "" });
    expect(await ssh.run(target, "true")).toEqual({ code: 0, stdout: "ok\n", stderr: "" });
  });
  test("strict checking stays on, against this box's own known_hosts", async () => {
    const shell = new FakeShell();
    shell.on(/^(ssh|scp) /, { code: 0, stdout: "", stderr: "" });
    const ssh = new OpenSsh(shell);
    await ssh.run(target, "true");
    await ssh.push(target, "/r/local/rented-box/payload.tar.gz", "/workspace/rig/payload.tar.gz");
    for (const argv of shell.calls) {
      expect(argv).toContain("StrictHostKeyChecking=accept-new");
      expect(argv).toContain(`UserKnownHostsFile=${target.knownHosts}`);
    }
  });
});
