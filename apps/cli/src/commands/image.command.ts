import type { Log, Result } from "@rig/core";
import type { Head } from "@rig/head";
import type { BuildImage, ImageReport } from "@rig/image";
import { type Args, flagBool, flagInt, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "image <head> [--gpu N] [--from-tarball FILE] [--push] [--json]   the head as a container image a rented box runs with no script: the rig CLI, the head and its engine for this card's sm, through the public repo's rules, proven on the card; --push publishes it to registry.toml's registry (the pack is fetched at the first start) and saves the head's vast template on it";

/** what a push is followed by: the head's template saved on the image just pushed (`rig vast template`), null where
 *  this rig has no engine to render one */
export type PublishAfterPush =
  | ((head: Head) => Promise<Result<{ name: string; hashId?: string | undefined }>>)
  | null;

export function buildImageCommand(
  service: BuildImage,
  loadHead: LoadHead,
  log: Log,
  publish: PublishAfterPush,
): Command {
  return {
    name: "image",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    run(args: Args) {
      return withHead(args.positionals[0], "image <head>", loadHead, log, async (head) => {
        const result = await service.run(head, {
          gpu: flagInt(args, "gpu") ?? 0,
          fromTarball: flagStr(args, "from-tarball"),
          push: flagBool(args, "push"),
        });
        // a pushed image no template names is one no box rents: the template follows it, edited in place
        if (result.ok && result.value.pushed && publish) {
          const saved = await publish(head);
          if (!saved.ok) {
            log.error(
              `pushed ${result.value.digest}, but its template was not saved: ${saved.message}`,
            );
            return saved.code;
          }
          log.info(`template ${saved.value.name}: ${saved.value.hashId}`);
        }
        return reportLine(log, args, result, describe);
      });
    },
  };
}

function describe(report: ImageReport): string {
  return report.pushed
    ? `pushed ${report.digest ?? report.image}`
    : `built ${report.image} (proven on the card; --push publishes it)`;
}
