import { describe, expect, test } from "bun:test";
import { FakeShell } from "@rig/testing";
import { VastAiRental } from "./vast-ai-rental.ts";

/** one page of `vastai show instances-v1 --raw`: an object, its rows under "instances" and the next page's token */
const listing = (ids: number[], nextToken: string | null = null) => ({
  code: 0,
  stdout: JSON.stringify({
    success: true,
    instances: ids.map((id) => ({ id, actual_status: "running", label: "rig" })),
    next_token: nextToken,
    total_instances: ids.length,
  }),
  stderr: "",
});

describe("VastAiRental.list", () => {
  test("every page of instances-v1, followed by its token: a box on the second page is listed", async () => {
    // instances-v1 pages 25 at a time, and its -a gathers pages only for its tables: under --raw it returns the
    // first page with the next one's token (vastai 1.0.12, cli/commands/instances.py:1270)
    const shell = new FakeShell();
    const vast = new VastAiRental(shell);
    shell.on(/^vastai show instances-v1 --raw$/, listing([51000001, 54000001], "page-2"));
    shell.on(/^vastai show instances-v1 --next-token page-2 --raw$/, listing([54030694]));
    expect((await vast.list()).map((instance) => instance.id)).toEqual([
      51000001, 54000001, 54030694,
    ]);
    expect(shell.calls.map((call) => call.join(" "))).toEqual([
      "vastai show instances-v1 --raw",
      "vastai show instances-v1 --next-token page-2 --raw",
    ]);
  });
  test("a listing that is not instances-v1's object, or whose tokens never end, is not read as a listing", async () => {
    const shell = new FakeShell();
    const vast = new VastAiRental(shell);
    // the old command's bare array
    shell.on(/^vastai show instances-v1 --raw$/, {
      code: 0,
      stdout: JSON.stringify([{ id: 1000 }]),
      stderr: "",
    });
    await expect(vast.list()).rejects.toThrow("no instances");
    shell.on(/^vastai show instances-v1/, listing([1000], "again"));
    await expect(vast.list()).rejects.toThrow("pages");
  });
});

describe("VastAiRental.show: the host's own ssh endpoint", () => {
  // the shape read off box 53930876 (a KVM VM), with a documentation address: the proxy at ssh8.vast.ai refused for
  // 13 minutes while the host's own address and the port mapped to 22 answered the first time it was tried
  const row = {
    id: 1000,
    actual_status: "running",
    label: "rig",
    ssh_host: "ssh8.vast.ai",
    ssh_port: 33272,
    public_ipaddr: "203.0.113.47",
    ports: { "22/tcp": [{ HostIp: "0.0.0.0", HostPort: "40174" }] },
  };
  const show = async (raw: Record<string, unknown>) => {
    const shell = new FakeShell();
    shell.on(/^vastai show instance 1000 --raw$/, {
      code: 0,
      stdout: JSON.stringify(raw),
      stderr: "",
    });
    return new VastAiRental(shell).show(1000);
  };
  test("the public address and the host port mapped to 22 are carried beside the proxy's", async () => {
    expect(await show(row)).toMatchObject({
      sshHost: "ssh8.vast.ai",
      sshPort: 33272,
      directSsh: { host: "203.0.113.47", port: 40174 },
    });
  });
  test("an instance that maps no port 22, or has no public address, has none", async () => {
    const { ports: _ports, ...unmapped } = row;
    expect((await show(unmapped))?.directSsh).toBeUndefined();
    const { public_ipaddr: _ip, ...unaddressed } = row;
    expect((await show(unaddressed))?.directSsh).toBeUndefined();
    expect(
      (await show({ ...row, ports: { "22/tcp": [{ HostPort: "nope" }] } }))?.directSsh,
    ).toBeUndefined();
  });
});

describe("VastAiRental.show", () => {
  test("a failed show falls back to the listing: its row when listed, null when the listing confirms the instance is gone, thrown when the listing fails too", async () => {
    const shell = new FakeShell();
    const vast = new VastAiRental(shell);
    shell.on(/^vastai show instance 1000 --raw$/, {
      code: 0,
      stdout: JSON.stringify({
        id: 1000,
        actual_status: "running",
        label: "rig",
        gpu_util: 97,
        image_uuid: "registry.torad.ai/rig:glm-5.3-flash-sm120-2a9d696-efb7b816",
      }),
      stderr: "",
    });
    expect(await vast.show(1000)).toMatchObject({
      id: 1000,
      status: "running",
      gpuUtil: 97,
      image: "registry.torad.ai/rig:glm-5.3-flash-sm120-2a9d696-efb7b816",
    });
    // a 429 on show while the listing still has the box: the box is not gone, and the listing's
    // row is the market read, its card reading included
    shell.on(/^vastai show instance 1000 --raw$/, {
      code: 1,
      stdout: "",
      stderr: "429 Too Many Requests",
    });
    shell.on(/^vastai show instances-v1 --raw$/, {
      code: 0,
      stdout: JSON.stringify({
        instances: [{ id: 1000, actual_status: "running", label: "rig", gpu_util: 97 }],
        next_token: null,
      }),
      stderr: "",
    });
    expect(await vast.show(1000)).toMatchObject({ id: 1000, status: "running", gpuUtil: 97 });
    // the listing unreadable too (an expired key): nothing confirms the box is gone
    shell.on(/^vastai show instances-v1 --raw$/, {
      code: 1,
      stdout: "",
      stderr: "401 Unauthorized",
    });
    await expect(vast.show(1000)).rejects.toThrow("401 Unauthorized");
    // destroyed: show fails and the listing has no such box
    shell.on(/^vastai show instances-v1 --raw$/, listing([56]));
    expect(await vast.show(1000)).toBeNull();
  });
});

describe("VastAiRental.searchOffers and create", () => {
  test("offers are priced with the box's own disk, and their download and disk prices are read", async () => {
    const shell = new FakeShell();
    const vast = new VastAiRental(shell);
    // a row as vast answers `--storage 160` (2026-09-30): dph_total includes the 160 GB disk's storage_total_cost
    shell.on(/^vastai search offers .* --storage 160 -o dph_total --raw$/, {
      code: 0,
      stdout: JSON.stringify([
        {
          id: 50138870,
          gpu_name: "RTX PRO 6000 S",
          num_gpus: 2,
          gpu_ram: 97887,
          compute_cap: 1200,
          dph_total: 0.261,
          inet_down_cost: 0.0027,
          storage_total_cost: 0.0741,
          machine_id: 56409,
        },
      ]),
      stderr: "",
    });
    expect(await vast.searchOffers("num_gpus=2", 160)).toMatchObject([
      { id: 50138870, dph: 0.261, downCostPerGb: 0.0027, storagePerHour: 0.0741, machineId: 56409 },
    ]);
  });

  test("a box from a template takes the template's image, login and launch; one from an image says them", async () => {
    const shell = new FakeShell();
    const vast = new VastAiRental(shell);
    shell.on(/^vastai create instance/, {
      code: 0,
      stdout: JSON.stringify({ success: true, new_contract: 1000 }),
      stderr: "",
    });
    await vast.create(7, { templateHash: "a79f7a77", diskGb: 160, label: "rig" });
    await vast.create(7, { image: "nvidia/cuda:13", diskGb: 40, label: "rig" });
    // a KVM box's on-start repairs the key vast wrote; it travels as one argument, whatever it contains
    await vast.create(7, {
      image: "docker.io/vastai/kvm:x",
      diskGb: 50,
      label: "rig",
      onstart: "a; b",
    });
    const creates = shell.calls
      .filter((call) => call[1] === "create")
      .map((call) => call.join(" "));
    expect(creates).toEqual([
      "vastai create instance 7 --template_hash a79f7a77 --disk 160 --label rig --cancel-unavail --raw",
      "vastai create instance 7 --image nvidia/cuda:13 --ssh --direct --disk 40 --label rig --cancel-unavail --raw",
      "vastai create instance 7 --image docker.io/vastai/kvm:x --ssh --direct --onstart-cmd a; b --disk 50 --label rig --cancel-unavail --raw",
    ]);
  });
});
