import { type Args, flagInt } from "../../shared/cli/args.ts";
import { type Command, type LoadHead, printJson, withHead } from "../../shared/cli/command.ts";
import type { Log } from "../../shared/ports/index.ts";
import type { DescribeHead } from "./head-description.service.ts";

const USAGE =
  "describe [<head>] [--gpu N]   one JSON object on stdout: what the head is and what this machine has of it; without a head, the heads";

export function describeHeadCommand(
  service: DescribeHead,
  loadHead: LoadHead,
  listHeads: () => Promise<string[]>,
  log: Log,
): Command {
  return {
    name: "describe",
    usage: USAGE,
    async run(args: Args) {
      const name = args.positionals[0];
      if (!name) {
        printJson({ heads: await listHeads() });
        return 0;
      }
      return withHead(name, "describe [<head>]", loadHead, log, async (head) => {
        printJson(await service.run(head, flagInt(args, "gpu") ?? head.gpu));
        return 0;
      });
    },
  };
}
