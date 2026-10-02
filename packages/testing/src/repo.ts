import { resolve } from "node:path";

/** the checkout's root, for a test that reads what is committed there: heads/, engine/, vast.toml */
export const repoRoot = resolve(import.meta.dir, "../../..");
