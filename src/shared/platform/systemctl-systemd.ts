import { userInfo } from "node:os";
import { join } from "node:path";
import type { Shell, Systemd } from "../ports/index.ts";

/** systemd --user, the scope a workstation head runs in (the unit survives logout via linger). */
export class SystemctlSystemd implements Systemd {
  constructor(
    private readonly shell: Shell,
    private readonly home = process.env.HOME ?? "",
    private readonly user = userInfo().username,
  ) {}
  unitDir() {
    return join(this.home, ".config", "systemd", "user");
  }
  private async ctl(...args: string[]) {
    const result = await this.shell.run(["systemctl", "--user", ...args], { timeoutMs: 120_000 });
    if (result.code !== 0)
      throw new Error(
        `systemctl --user ${args.join(" ")}: ${result.stderr.trim() || result.stdout.trim()}`,
      );
    return result.stdout;
  }
  async daemonReload() {
    await this.ctl("daemon-reload");
  }
  async enable(unit: string) {
    await this.ctl("enable", unit);
  }
  async disable(unit: string) {
    await this.ctl("disable", unit);
  }
  async restart(unit: string) {
    await this.ctl("restart", unit);
  }
  async stop(unit: string) {
    await this.ctl("stop", unit);
  }
  async isActive(unit: string) {
    return (
      (await this.shell.run(["systemctl", "--user", "is-active", unit])).stdout.trim() === "active"
    );
  }
  async lastResult(unit: string) {
    const shown = await this.shell.run([
      "systemctl",
      "--user",
      "show",
      "-p",
      "Result",
      "--value",
      unit,
    ]);
    const value = shown.stdout.trim();
    return shown.code === 0 && value !== "" ? value : null;
  }
  async mainPid(unit: string) {
    const pid = Number(
      (
        await this.shell.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit])
      ).stdout.trim(),
    );
    return pid > 0 ? pid : null;
  }
  async linger() {
    const shown = await this.shell.run([
      "loginctl",
      "show-user",
      this.user,
      "--property=Linger",
      "--value",
    ]);
    const value = shown.stdout.trim();
    return shown.code === 0 && (value === "yes" || value === "no") ? value === "yes" : null;
  }
  async enableLinger() {
    return (await this.shell.run(["loginctl", "enable-linger", this.user])).code === 0;
  }
}
