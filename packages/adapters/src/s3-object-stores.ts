import type { Bucket, ObjectStore, ObjectStores } from "@rig/core";

/** S3-compatible buckets (R2) through Bun's client, which uploads a large file in parts */
export class S3ObjectStores implements ObjectStores {
  open(bucket: Bucket): ObjectStore {
    const client = new Bun.S3Client(bucket);
    return {
      exists: (key) => client.exists(key),
      async put(key, from, contentType) {
        const body = "file" in from ? Bun.file(from.file) : from.bytes;
        await client.write(key, body, contentType ? { type: contentType } : {});
      },
    };
  }
}
