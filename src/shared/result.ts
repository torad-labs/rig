// Use cases return a Result, never throw for expected outcomes: a caller (the CLI, `up`, a test)
// decides what an outcome means. Exit codes are part of the contract a wizard reads.
export type Result<T = void> =
  | { ok: true; value: T }
  | { ok: false; code: ExitCode; message: string };

export const ExitCode = {
  Ok: 0,
  Failure: 1, // something needed is missing or wrong (named in the message)
  Busy: 2, // refused because the head is serving live traffic
  Unsupported: 3, // the card is not one this format is measured on
  Driver: 4, // the driver cannot run the toolkit that would build the kernels
  Usage: 64, // bad arguments (sysexits EX_USAGE)
} as const;
export type ExitCode = (typeof ExitCode)[keyof typeof ExitCode];

export const ok = <T>(value: T): Result<T> => ({ ok: true, value });
export const fail = (code: ExitCode, message: string): Result<never> => ({
  ok: false,
  code,
  message,
});
