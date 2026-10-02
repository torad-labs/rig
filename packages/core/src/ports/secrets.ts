/** The operator's keyring: a secret by the service and key it is stored under, null when it holds none (a rented box
 *  has no keyring). A value read is handed to the one call that needs it, never logged or written. */
export interface Secrets {
  lookup(service: string, key: string): Promise<string | null>;
}
