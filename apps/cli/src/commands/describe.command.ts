import type { Log } from "@rig/core";
import type { DescribeHead } from "@rig/serve";
import { type Args, flagDevices } from "../cli/args.ts";
import { type Command, type LoadHead, printJson, withHead } from "../cli/command.ts";

const USAGE =
  "describe [<head>] [--gpu N[,M…]|auto]   one JSON object on stdout: what the head is and what this machine has of it; without a head, the heads";

export function describeHeadCommand(
  service: DescribeHead,
  loadHead: LoadHead,
  listHeads: () => Promise<string[]>,
  log: Log,
): Command {
  return {
    name: "describe",
    usage: USAGE,
    headName: (args) => args.positionals[0],
    async run(args: Args) {
      const name = args.positionals[0];
      if (!name) {
        printJson({ heads: await listHeads() });
        return 0;
      }
      return withHead(name, "describe [<head>]", loadHead, log, async (head) => {
        printJson(await service.run(head, flagDevices(args, "gpu")));
        return 0;
      });
    },
  };
}
