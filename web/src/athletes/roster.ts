// Deterministic per-player looks (skin tone, hair, boots) and kit choices from MatchMeta.
import { hashString } from "../coords.ts";
import type { KitColors, MatchMeta, PlayerMeta, TeamMeta } from "../types.ts";
import { HAIR_COLORS, HAIR_STYLES, type HairStyle } from "./hair.ts";

export const SKIN_TONES = ["#f0c8a4", "#ddaa80", "#c48a5f", "#a16b45", "#7b4b2d", "#53321e"];
export const BOOT_COLORS = ["#101114", "#f2f2f0", "#d7ff3a", "#ff7a1a", "#4cc9f0", "#c8102e", "#101114", "#f2f2f0"];

export const REFEREE_KIT: KitColors = {
  primary: "#23262d",
  secondary: "#23262d",
  accent: "#d7ff3a",
  number: "#d7ff3a",
  pattern: "plain",
};

export interface PlayerLook {
  skin: string;
  hair: HairStyle;
  hairColor: string;
  boot: string;
  seed: number;
}

export function lookFor(id: string, name: string): PlayerLook {
  const h = hashString(`${id}:${name}`);
  const skinIdx = h % SKIN_TONES.length;
  const hair = HAIR_STYLES[(h >>> 4) % HAIR_STYLES.length]!;
  // darker hair is more common, and lighter shades only appear with lighter skin tones
  const hcIdx = skinIdx >= 3 ? (h >>> 9) % 2 : (h >>> 9) % HAIR_COLORS.length;
  return {
    skin: SKIN_TONES[skinIdx]!,
    hair,
    hairColor: HAIR_COLORS[hcIdx]!,
    boot: BOOT_COLORS[(h >>> 13) % BOOT_COLORS.length]!,
    seed: h,
  };
}

export function teamOf(meta: MatchMeta, side: "home" | "away"): TeamMeta {
  return side === "home" ? meta.home : meta.away;
}

export function kitFor(team: TeamMeta, p: PlayerMeta): KitColors {
  return p.position === "GK" ? team.gk_kit : team.kit;
}

export function allPlayers(meta: MatchMeta): PlayerMeta[] {
  return [...meta.home.players, ...meta.away.players];
}
