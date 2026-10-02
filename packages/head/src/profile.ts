// Which of the head's profiles a machine serves, and on which of its cards. The profiles are
// measurements, not a formula — utilization runs 77–96% across the measured cards — listed most
// capable first; a machine gets the first one whose cards it has: that many cards of one compute
// capability, each with the profile's VRAM free for the head. [sizing], when the head declares it,
// is what each single-card profile is checked against at load (head-config.ts).
import type { Devices, Gpu, GpuInfo, Host } from "@rig/core";
import { ExitCode, fail, ok, type Result } from "@rig/core";
import type { Head } from "./head.ts";
import type { Profile } from "./head-config.ts";

/** a card as a profile sees it: where it is, what it runs, and the VRAM the head can have on it */
export interface CardRoom {
  index: number;
  cap: string;
  vramMiB: number;
}

/** the profile a machine serves and the cards it serves it on, lowest index first */
export interface Placement {
  profile: Profile;
  cards: CardRoom[];
}

export function pickProfile(
  head: Pick<Head, "profiles">,
  cards: readonly CardRoom[],
): Result<Placement> {
  for (const profile of head.profiles) {
    const roomy = cards.filter((card) => card.vramMiB >= profile.min_vram_mib);
    for (const cap of new Set(roomy.map((card) => card.cap))) {
      const same = roomy.filter((card) => card.cap === cap);
      if (same.length >= profile.devices)
        return ok({ profile, cards: same.slice(0, profile.devices) });
    }
  }
  const have = cards.map((card) => `${card.vramMiB} MiB`).join(" + ") || "no card";
  const least = new Map<number, number>();
  for (const { devices, min_vram_mib } of head.profiles)
    least.set(devices, Math.min(least.get(devices) ?? min_vram_mib, min_vram_mib));
  const wants = [...least]
    .sort(([a], [b]) => a - b)
    .map(([n, mib]) =>
      n === 1 ? `one card with ${mib} MiB` : `${n} cards of one sm with ${mib} MiB each`,
    )
    .join(", or ");
  return fail(
    ExitCode.Unsupported,
    `${have} of VRAM for it fits no profile this head declares (the least: ${wants})`,
  );
}

/** The VRAM a head can have on `card`: its total less what every other process holds, since a
 *  card that also drives a desktop keeps its compositor's and browser's share (2.3 GB of a 16 GB
 *  card here, where the 16 GB profile then failed to allocate). What holds the head's port is the
 *  head itself, already serving, and its share is its own. */
export async function headVramMiB(
  deps: { gpu: Gpu; host: Host },
  card: GpuInfo,
  port: number,
): Promise<number> {
  const pid = await deps.host.listeningPid(port);
  const own = pid === null ? 0 : await deps.gpu.processMiB(card.index, pid);
  return card.memoryMiB - Math.max(0, card.usedMiB - own);
}

/** a card's room for the head, as pickProfile takes it */
export async function cardRoom(
  deps: { gpu: Gpu; host: Host },
  card: GpuInfo,
  port: number,
): Promise<CardRoom> {
  return { index: card.index, cap: card.computeCap, vramMiB: await headVramMiB(deps, card, port) };
}

/** a card of this machine and the room the head has on it */
export interface MachineCard {
  info: GpuInfo;
  room: CardRoom;
}

/** the cards `devices` names (every card nvidia-smi lists, for auto), each with the head's room on it */
export async function machineCards(
  deps: { gpu: Gpu; host: Host },
  devices: Devices,
  port: number,
): Promise<Result<MachineCard[]>> {
  const infos: GpuInfo[] = [];
  if (devices === "auto") {
    infos.push(...(await deps.gpu.list()));
    if (infos.length === 0)
      return fail(ExitCode.Failure, "no CUDA card on this machine (nvidia-smi lists none)");
  } else {
    for (const index of devices) {
      const info = await deps.gpu.query(index);
      if (!info) return fail(ExitCode.Failure, `no CUDA card at nvidia-smi index ${index}`);
      infos.push(info);
    }
  }
  const cards: MachineCard[] = [];
  for (const info of infos) cards.push({ info, room: await cardRoom(deps, info, port) });
  return ok(cards);
}

/** the profile the head serves on the cards `devices` names and the cards it takes of them; a refusal names each card
 *  and the share of it other processes hold (a desktop's, another model's), since that share is why a card that is
 *  big enough on paper is not */
export async function placeHead(
  deps: { gpu: Gpu; host: Host },
  head: Pick<Head, "name" | "port" | "profiles">,
  devices: Devices,
): Promise<Result<Placement & { machine: MachineCard[] }>> {
  const machine = await machineCards(deps, devices, head.port);
  if (!machine.ok) return machine;
  const placed = pickProfile(
    head,
    machine.value.map((card) => card.room),
  );
  if (placed.ok) return ok({ ...placed.value, machine: machine.value });
  const cards = machine.value;
  const names = cards.map(({ info }) => `${info.name} at index ${info.index}`).join(", ");
  const held = cards
    .map(({ info, room }) => ({ info, mib: info.memoryMiB - room.vramMiB }))
    .filter(({ mib }) => mib > 0);
  const shares =
    cards.length === 1
      ? held.map(
          ({ info, mib }) => `${mib} of its ${info.memoryMiB} MiB are held by other processes`,
        )
      : [
          `other processes hold ${held.map(({ info, mib }) => `${mib} of index ${info.index}'s ${info.memoryMiB} MiB`).join(", ")}`,
        ];
  const others = held.length > 0 ? ` (${shares.join("")}: a desktop, another model)` : "";
  return fail(placed.code, `${names} cannot serve ${head.name}${others}: ${placed.message}`);
}

/** how a refusal names where a placement serves: the card's (or the cards') profile and the VRAM the head has there */
export function placementLabel(placement: Placement): string {
  const mib = placement.cards.map((card) => card.vramMiB).join(" + ");
  return placement.cards.length === 1
    ? `this card's profile (${mib} MiB for the head)`
    : `these ${placement.cards.length} cards' profile (${mib} MiB for the head)`;
}
