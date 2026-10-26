// PlayerState.anim -> Mixamo clips (names = FBX basenames, see scripts/assets/convert_mixamo.mjs).
// Pure data + timing functions so they can be unit tested.
import type { PlayerAnim } from "../types.ts";

/**
 * Renderer convention for strike animations (passes, crosses, shots, headers, drop kicks, tackles):
 * the ball is struck CONTACT_S seconds after anim_t = 0. Each clip is time-warped so its measured contact frame
 * (athlete_clips.json `contact_s`) lands there; after contact the clip plays at its own speed.
 */
export const CONTACT_S = 0.4;

/** Cross-fade between consecutive animation states. */
export const FADE_S = 0.18;

export type AnimKind = "locomotion" | "loop" | "oneshot" | "strike";

export interface AnimSpec {
  kind: AnimKind;
  /** Candidate clips; `anim_variant` picks one (modulo length). */
  clips: string[];
  /** Playback rate for one-shots (1 = authored speed). */
  rate?: number;
  /** Start offset into the clip (seconds) for one-shots with long lead-ins. */
  offset?: number;
  /** The clip already contains vertical body motion: ignore PlayerState.z for it. */
  ownsHeight?: boolean;
}

export const ANIMS: Record<PlayerAnim, AnimSpec> = {
  idle: { kind: "locomotion", clips: ["Breathing_Idle"] },
  jog: { kind: "locomotion", clips: ["Jog_Forward"] },
  run: { kind: "locomotion", clips: ["Run_forward"] },
  sprint: { kind: "locomotion", clips: ["Sprint"] },
  turn: { kind: "locomotion", clips: ["Run_forward"] },
  jog_back: { kind: "loop", clips: ["Running_Backward"] },
  strafe_left: { kind: "loop", clips: ["Jog_Strafe_Left"] },
  strafe_right: { kind: "loop", clips: ["Jog_Strafe_Right"] },
  dribble: { kind: "loop", clips: ["Dribble"] },
  gk_idle: { kind: "loop", clips: ["Goalkeeper_Idle"] },
  receive_ground: { kind: "oneshot", clips: ["Receive_ground_pass"], offset: 0.25, rate: 1.4 },
  receive_aerial: { kind: "oneshot", clips: ["Receive_aerial_pass"], rate: 1.1 },
  pass_short: { kind: "strike", clips: ["Soccer_Pass"] },
  pass_long: { kind: "strike", clips: ["long_pass"] },
  cross: { kind: "strike", clips: ["long_pass"] },
  shoot: { kind: "strike", clips: ["Strike_Foward"] },
  chip: { kind: "strike", clips: ["Soccer_Pass"] },
  volley: { kind: "strike", clips: ["Strike_Foward"] },
  header_standing: { kind: "strike", clips: ["standing_Header"] },
  header_jumping: { kind: "strike", clips: ["jumping_Header"], ownsHeight: true },
  tackle: { kind: "strike", clips: ["Soccer_Tackle"] },
  slide_tackle: { kind: "oneshot", clips: ["Slide_Tackle"], rate: 1.1, ownsHeight: true },
  fall: { kind: "oneshot", clips: ["Soccer_Trip"], ownsHeight: true },
  get_up: { kind: "oneshot", clips: ["Getting_Up"], rate: 1.6, offset: 0.4, ownsHeight: true },
  throw_in: { kind: "oneshot", clips: ["Throw_In"] },
  gk_dive_left: { kind: "oneshot", clips: ["Goalkeeper_Diving_Save_left"], offset: 0.35, rate: 1.25, ownsHeight: true },
  gk_dive_right: { kind: "oneshot", clips: ["Goalkeeper_Diving_Save_Right"], offset: 0.35, rate: 1.25, ownsHeight: true },
  gk_block_left: { kind: "oneshot", clips: ["Goalkeeper_Body_Block_left"], ownsHeight: true },
  gk_block_right: { kind: "oneshot", clips: ["Goalkeeper_Body_Block_right"], ownsHeight: true },
  gk_catch_high: { kind: "oneshot", clips: ["Goalkeeper_Catch_highshot"], ownsHeight: true },
  gk_catch_low: { kind: "oneshot", clips: ["Goalkeeper_Catch_lowshot"], ownsHeight: true },
  gk_scoop: { kind: "oneshot", clips: ["Goalkeeper_Scoop"] },
  gk_throw: { kind: "oneshot", clips: ["Goalkeeper_Overhand_Throw"] },
  gk_drop_kick: { kind: "strike", clips: ["Goalkeeper_Drop_Kick"] },
  celebrate: { kind: "oneshot", clips: ["celebration_Cheering", "celebration_Fist_Pump", "celebration_Victory"] },
  disappointed: { kind: "oneshot", clips: ["Disappointed"] },
};

export interface ClipInfo {
  kind: string;
  duration_s: number;
  contact_s?: number;
  natural_speed_mps?: number;
  phase0_s?: number;
}

export function clipFor(anim: PlayerAnim, variant: number): string {
  const spec = ANIMS[anim] ?? ANIMS.idle;
  return spec.clips[variant % spec.clips.length]!;
}

/**
 * Clip time for a one-shot or strike state `animT` seconds after it started.
 * Strikes: [0, CONTACT_S] maps onto [lead-in start, contact]; afterwards authored speed. Holds the last frame.
 */
export function oneShotTime(anim: PlayerAnim, clip: ClipInfo, animT: number): number {
  const spec = ANIMS[anim];
  const dur = clip.duration_s;
  let t: number;
  if (spec.kind === "strike" && clip.contact_s !== undefined) {
    const contact = clip.contact_s;
    // keep at most 0.75 s of lead-in so long wind-ups (drop kick) stay snappy
    const lead = Math.min(contact, 0.75);
    const start = contact - lead;
    t = animT < CONTACT_S ? start + (animT / CONTACT_S) * lead : contact + (animT - CONTACT_S) * (spec.rate ?? 1);
  } else {
    t = (spec.offset ?? 0) + animT * (spec.rate ?? 1);
  }
  return Math.min(Math.max(0, t), dur - 1e-3);
}

/** Forward locomotion blend: speed (m/s) -> weights of idle, jog, run, sprint. */
export const LOCO_CLIPS = ["Breathing_Idle", "Jog_Forward", "Run_forward", "Sprint"] as const;
const LOCO_SPEEDS = [0, 2.2, 5.0, 7.4];

export function locomotionWeights(speed: number): [number, number, number, number] {
  const w: [number, number, number, number] = [0, 0, 0, 0];
  if (speed <= LOCO_SPEEDS[0]!) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < 3; i++) {
    const a = LOCO_SPEEDS[i]!;
    const b = LOCO_SPEEDS[i + 1]!;
    if (speed <= b) {
      const f = (speed - a) / (b - a);
      // idle -> jog blends fast so slow walks do not float
      const s = i === 0 ? Math.min(1, f * 1.8) : f;
      w[i] = 1 - s;
      w[i + 1] = s;
      return w;
    }
  }
  w[3] = 1;
  return w;
}

/**
 * Cycles per second of the shared locomotion phase at `speed`: the weighted average of each moving clip's
 * natural cadence, scaled by speed / natural speed (limited so feet do not visibly race or crawl).
 */
export function locomotionCadence(speed: number, info: Record<string, ClipInfo | undefined>): number {
  const w = locomotionWeights(speed);
  let num = 0;
  let den = 0;
  LOCO_CLIPS.forEach((name, i) => {
    const wi = w[i] ?? 0;
    if (i === 0 || wi === 0) return;
    const c = info[name];
    if (!c) return;
    const nat = c.natural_speed_mps ?? speed;
    const rate = Math.min(1.35, Math.max(0.7, speed / Math.max(0.1, nat)));
    num += wi * (rate / c.duration_s);
    den += wi;
  });
  return den > 0 ? num / den : 0;
}
