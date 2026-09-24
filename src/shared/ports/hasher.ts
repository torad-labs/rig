// Hasher: one seam between rig and the machine. A port names a capability, never a tool.
export interface Hasher {
  sha256File(path: string): Promise<string>;
}
