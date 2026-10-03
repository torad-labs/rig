import { ExitCode, type Log, type Result } from "@rig/core";
import type { Head } from "@rig/head";
import type { BuildImage, ImageReport } from "@rig/image";
import { type Args, flagBool, flagInt, flagStr } from "../cli/args.ts";
import { type Command, type LoadHead, reportLine, withHead } from "../cli/command.ts";

const USAGE =
  "image <head> [--gpu N] [--from-tarball FILE] [--push] | image <head> --cap N --build FILE [--from-tarball FILE] | image <head> --prove RECEIPT [--gpu N] | image <head> --push-proven RECEIPT, each with [--json]   the head as a container image a rented box runs with no script: the rig CLI, the head and its engine for one sm, through the public repo's rules, proven on a card; the first form does it all on this machine's card and --push publishes it to registry.toml's registry (the pack is fetched at the first start) and saves the head's vast template on it; the other three are that work in the parts that belong on different machines: --build makes the image for sm N with no card and keeps it as a tarball, its image ID and this commit recorded in the image.json beside it; --prove, on the machine with the card and any checkout, requires the image a `docker load` of that tarball put there to be that ID and the card to be of its sm, runs it as a rented box will, and writes the card and the time into the receipt; --push-proven publishes the image a load put back here, with no card and no build, refusing anything that is not the ID a card proved — the rules live here, the card is rented, and the registry credential never leaves this machine";

/** the flags each part takes beyond the part's own; the usage line is main's list of what is allowed at all */
const TAKES = {
  build: ["cap", "from-tarball"],
  prove: ["gpu"],
  "push-proven": [],
  run: ["gpu", "from-tarball", "push"],
} as const;
const PART_FLAGS = ["build", "prove", "push-proven"] as const;

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
      const parts = PART_FLAGS.filter((flag) => args.flags[flag] !== undefined);
      const part = parts[0] ?? "run";
      const refusal =
        parts.length > 1
          ? `${parts.map((flag) => `--${flag}`).join(" and ")} are different runs on different machines: pick one`
          : misplaced(args, part);
      if (refusal) {
        log.error(refusal);
        return Promise.resolve(ExitCode.Usage);
      }
      return withHead(args.positionals[0], "image <head>", loadHead, log, async (head) => {
        const result = await dispatch(service, head, args, part);
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

/** a flag that belongs to another part, or a part missing the flag it is made of, as the sentence that refuses it */
function misplaced(args: Args, part: keyof typeof TAKES): string | undefined {
  const own = new Set<string>([...TAKES[part], ...(part === "run" ? [] : [part])]);
  const wrong = [...PART_FLAGS, "cap", "gpu", "from-tarball", "push"].filter(
    (flag) => args.flags[flag] !== undefined && !own.has(flag),
  );
  if (wrong.length > 0)
    return part === "run"
      ? `${wrong.map((flag) => `--${flag}`).join(", ")} ${wrong.length > 1 ? "belong" : "belongs"} to a part of the image build that runs on its own: the one-shot form takes --gpu, --from-tarball and --push`
      : `--${part} is its own run and does not take ${wrong.map((flag) => `--${flag}`).join(", ")}`;
  if (part !== "run" && flagStr(args, part) === undefined)
    return `--${part} takes a ${part === "build" ? "file to keep the image in" : "receipt (the image.json --build wrote)"}`;
  if (part === "build" && flagStr(args, "cap") === undefined)
    return "--build is for an sm the image is made for: it takes --cap N, since there is no card here to read it from";
  return undefined;
}

function dispatch(
  service: BuildImage,
  head: Head,
  args: Args,
  part: keyof typeof TAKES,
): Promise<Result<ImageReport>> {
  switch (part) {
    case "build":
      return service.build(head, {
        cap: flagStr(args, "cap") ?? "",
        fromTarball: flagStr(args, "from-tarball"),
        out: flagStr(args, "build") ?? "",
      });
    case "prove":
      return service.prove(head, {
        receipt: flagStr(args, "prove") ?? "",
        gpu: flagInt(args, "gpu") ?? 0,
      });
    case "push-proven":
      return service.pushProven(head, { receipt: flagStr(args, "push-proven") ?? "" });
    case "run":
      return service.run(head, {
        gpu: flagInt(args, "gpu") ?? 0,
        fromTarball: flagStr(args, "from-tarball"),
        push: flagBool(args, "push"),
      });
  }
}

function describe(report: ImageReport): string {
  if (report.pushed) return `pushed ${report.digest ?? report.image}`;
  if (!report.proven)
    return `built ${report.image} (image ID ${report.id}; --prove it on a card of sm_${report.cap}, then --push-proven)`;
  return `${report.image} proven on ${report.proven.card} (--push publishes it, or --push-proven where the credentials are)`;
}
