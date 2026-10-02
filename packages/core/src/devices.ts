// The cards a command may put a head on: the nvidia-smi indices named (`--gpu 1`, `--gpu 0,1`), or every card the
// machine has (`--gpu auto`), from which the head's first profile the cards hold takes what it needs. CUDA numbers the
// cards as nvidia-smi does under CUDA_DEVICE_ORDER=PCI_BUS_ID, which serve sets.

export type Devices = readonly number[] | "auto";

/** `auto`, or distinct indices in the order given; undefined for anything else */
export function parseDevices(text: string): Devices | undefined {
  if (text === "auto") return "auto";
  if (!/^\d+(,\d+)*$/.test(text)) return undefined;
  const indices = text.split(",").map(Number);
  return new Set(indices).size === indices.length ? indices : undefined;
}

/** the spelling parseDevices reads back */
export function formatDevices(devices: Devices): string {
  return devices === "auto" ? "auto" : devices.join(",");
}
