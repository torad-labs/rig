// ObjectStore: one seam between rig and the machine. A port names a capability, never a tool.
export interface Bucket {
  endpoint: string;
  bucket: string;
  accessKeyId: string;
  secretAccessKey: string;
}
/** one bucket of an S3-compatible store */
export interface ObjectStore {
  exists(key: string): Promise<boolean>;
  /** a file's bytes, or the bytes given, under `key`; a large file goes up in parts */
  put(
    key: string,
    from: { file: string } | { bytes: Uint8Array | string },
    contentType?: string,
  ): Promise<void>;
}
export interface ObjectStores {
  open(bucket: Bucket): ObjectStore;
}
