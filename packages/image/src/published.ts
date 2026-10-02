// The images `rig image` built for a head, as its records say: one directory per head and sm under
// local/images/, each with the image.json the build wrote last. Another package reads them through
// here, never the files.
import { join } from "node:path";
import { ExitCode, type FileSystem, fail, type Layout, ok, type Result } from "@rig/core";
import type { ImageReport } from "./image-build.service.ts";

export interface PublishedImage {
  record: ImageReport;
  /** the image's directory, where a consumer of the image keeps what it made of it */
  dir: string;
}

/** where the image of `head` for sm_`cap` is built and recorded */
export const imageDir = (layout: Layout, head: string, cap: string) =>
  join(layout.imagesDir, `${head}-sm${cap}`);

/** the image `rig image --push` last published for the head: one sm's, or the refusal names them */
export async function publishedImage(
  fs: FileSystem,
  layout: Layout,
  head: string,
): Promise<Result<PublishedImage>> {
  const images = layout.imagesDir;
  const dirs = (await fs.exists(images))
    ? (await fs.list(images)).filter((name) => name.startsWith(`${head}-sm`))
    : [];
  const pushed: PublishedImage[] = [];
  for (const name of dirs) {
    const file = join(images, name, "image.json");
    if (!(await fs.exists(file))) continue;
    const record = JSON.parse(await fs.readText(file)) as ImageReport;
    if (record.pushed) pushed.push({ record, dir: join(images, name) });
  }
  if (pushed.length === 1) return ok(pushed[0]!);
  if (pushed.length === 0)
    return fail(ExitCode.Failure, `no pushed image of ${head}: run rig image ${head} --push first`);
  return fail(
    ExitCode.Failure,
    `${head} has pushed images for ${pushed.map((each) => `sm_${each.record.cap}`).join(", ")}: one template per sm is not chosen for you yet`,
  );
}
