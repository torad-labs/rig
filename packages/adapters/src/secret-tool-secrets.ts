import type { Secrets, Shell } from "@rig/core";

/** libsecret's keyring through `secret-tool lookup service <service> key <key>`: its exit is 1 with nothing found, and
 *  a machine without it (a rented box, CI) has no keyring */
export class SecretToolSecrets implements Secrets {
  constructor(private readonly shell: Shell) {}
  async lookup(service: string, key: string): Promise<string | null> {
    if (!(await this.shell.which("secret-tool"))) return null;
    const read = await this.shell.run(["secret-tool", "lookup", "service", service, "key", key], {
      timeoutMs: 30_000,
    });
    const value = read.stdout.trim();
    return read.code === 0 && value !== "" ? value : null;
  }
}
