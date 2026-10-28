// Pure overlay and Rewind logic (no DOM, no Three.js): thresholds, card content, verdict framing, distribution bars.
// Everything here is unit tested in tests/ui.test.ts.
import type {
  DecisionMoment,
  MatchEvent,
  OptionEvaluation,
  RolloutDistribution,
  RolloutOutcome,
} from "../types/generated/contracts.ts";

export const KMH = 3.6;
/** docs/event_taxonomy.md: sprint > 7.0 m/s (25.2 km/h); top-speed badge at >= 9.0 m/s. */
export const SPRINT_MPS = 7.0;
export const TOP_SPEED_MPS = 9.0;
export const GOAL_X = 52.5;

export type Outcome = RolloutOutcome["label"];
export type Framing = "beat_the_odds" | "unlucky" | "as_expected" | "bold_and_paid_off" | "same_call_as_reality";

export function kmh(mps: number): number {
  return mps * KMH;
}

/** Broadcast clock "mm:ss" from continuous match time; second half counts from 45:00 at its kick-off. */
export function formatClock(t: number, period: number, secondHalfStart: number | null): string {
  const base = period >= 2 ? 45 * 60 : 0;
  const start = period >= 2 ? (secondHalfStart ?? 45 * 60) : 0;
  const elapsed = Math.max(0, t - start);
  const regulation = 45 * 60;
  const mmss = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  if (elapsed > regulation) {
    const extra = Math.ceil((elapsed - regulation) / 60);
    return `${mmss(base + regulation)} +${extra}`;
  }
  return mmss(base + elapsed);
}

export interface SpeedBadge {
  kind: "sprint" | "top_speed";
  kmh: number;
}

/**
 * Badge for a player at `speed` m/s, given their match top speed so far (before this sample).
 * A top-speed badge needs >= 9.0 m/s and a new personal best; otherwise a sprint badge above 7.0 m/s.
 */
export function speedBadge(speed: number, topSoFar: number): SpeedBadge | null {
  if (speed >= TOP_SPEED_MPS && speed >= topSoFar) return { kind: "top_speed", kmh: kmh(speed) };
  if (speed > SPRINT_MPS) return { kind: "sprint", kmh: kmh(speed) };
  return null;
}

type Q = Record<string, unknown>;
const q = (e: MatchEvent): Q => (e.qualifiers ?? {}) as Q;
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function passDifficulty(xpass: number | null): string | null {
  if (xpass === null) return null;
  if (xpass >= 0.85) return "Routine";
  if (xpass >= 0.6) return "Testing";
  if (xpass >= 0.35) return "Difficult";
  return "Audacious";
}

export interface PassCard {
  kind: "pass";
  t: number;
  passer: string;
  recipient: string | null;
  distanceM: number;
  xpass: number | null;
  difficulty: string | null;
  technique: string | null;
}

export interface ShotCard {
  kind: "shot";
  t: number;
  shooter: string;
  xg: number | null;
  speedKmh: number | null;
  outcome: string;
  distanceM: number | null;
}

/** A pass card for a completed pass of at least `minM` metres, else null (short passes never get a card). */
export function passCard(e: MatchEvent, minM = 18): PassCard | null {
  if (e.type !== "pass") return null;
  const Q = q(e);
  if (Q.outcome !== "complete") return null;
  const a = e.location ?? [0, 0, 0];
  const b = e.end_location ?? a;
  const len = num(Q.length_m) ?? Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (len < minM && Q.is_key_pass !== true && Q.is_assist !== true) return null;
  const xpass = num(Q.xpass);
  return {
    kind: "pass",
    t: e.t,
    passer: e.player ?? "",
    recipient: typeof Q.recipient === "string" ? Q.recipient : null,
    distanceM: len,
    xpass,
    difficulty: passDifficulty(xpass),
    technique: typeof Q.technique === "string" && Q.technique !== "standard" ? Q.technique : null,
  };
}

export function shotCard(e: MatchEvent): ShotCard | null {
  if (e.type !== "shot") return null;
  const Q = q(e);
  const speed = num(Q.ball_speed_mps);
  return {
    kind: "shot",
    t: e.t,
    shooter: e.player ?? "",
    xg: num(Q.xg),
    speedKmh: speed === null ? null : kmh(speed),
    outcome: typeof Q.outcome === "string" ? Q.outcome : "shot",
    distanceM: num(Q.distance_to_goal_m),
  };
}

export function eventCard(e: MatchEvent): PassCard | ShotCard | null {
  return e.type === "shot" ? shotCard(e) : passCard(e);
}

/** Position of a marker on the timeline, 0..1. */
export function markerFraction(t: number, durationS: number): number {
  if (durationS <= 0) return 0;
  return Math.min(1, Math.max(0, t / durationS));
}

/** The real outcome after a moment, as a rollout outcome label (from the Scout's consequence). */
export function realOutcome(m: Pick<DecisionMoment, "consequence">): Outcome {
  switch (m.consequence.kind) {
    case "goal":
      return "goal";
    case "shot":
      return "shot_saved";
    case "big_chance":
      return "chance_created";
    case "turnover_to_chance_against":
      return "turnover";
    default:
      return "possession_retained";
  }
}

export const OUTCOME_TEXT: Record<Outcome, string> = {
  goal: "Goal",
  shot_saved: "Shot saved",
  shot_off_target: "Shot off target",
  shot_blocked: "Shot blocked",
  woodwork: "Hit the woodwork",
  chance_created: "Chance created",
  possession_retained: "Possession kept",
  turnover: "Turnover",
  out_of_play: "Out of play",
  foul_won: "Foul won",
  offside: "Offside",
};

export const FRAMING_TEXT: Record<Framing, string> = {
  beat_the_odds: "Beat the odds",
  unlucky: "Unlucky",
  as_expected: "As expected",
  bold_and_paid_off: "Bold, and it paid off",
  same_call_as_reality: "Same call as reality",
};

/** The option the player actually chose: `is_real`, else the option matching `real_action`'s intent. */
export function realOption(m: Pick<DecisionMoment, "options" | "real_action">): OptionEvaluation | null {
  const flagged = m.options.find((o) => o.is_real);
  if (flagged) return flagged;
  const ri = m.real_action.intent;
  return (
    m.options.find((o) => o.intent.kind === ri.kind && (ri.kind !== "pass_player" || o.intent.target_player === ri.target_player)) ??
    null
  );
}

export function sameCall(a: OptionEvaluation["intent"], b: OptionEvaluation["intent"]): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "pass_player") return a.target_player === b.target_player;
  if (a.kind === "shot") return true;
  if (!a.target || !b.target) return !a.target && !b.target;
  return Math.hypot(a.target[0] - b.target[0], a.target[1] - b.target[1]) < 3;
}

/**
 * Verdict framing (judgement-free until here). "Good" means the fan's action itself succeeded.
 * A low-probability success beats the odds, a risky one (under 65 %) was bold and paid off, and a likely action
 * that failed was unlucky.
 */
export function verdictFraming(
  user: Pick<OptionEvaluation, "p_success" | "intent">,
  real: Pick<OptionEvaluation, "intent"> | null,
  outcome: Pick<RolloutOutcome, "primary_success" | "label">,
): Framing {
  if (real && sameCall(user.intent, real.intent)) return "same_call_as_reality";
  const good = outcome.primary_success || outcome.label === "goal";
  if (good && user.p_success < 0.4) return "beat_the_odds";
  if (good && user.p_success < 0.65) return "bold_and_paid_off";
  if (!good && user.p_success >= 0.5) return "unlucky";
  return "as_expected";
}

export interface Bar {
  label: Outcome;
  text: string;
  count: number;
  fraction: number;
}

const GOODNESS: Outcome[] = [
  "goal",
  "woodwork",
  "shot_saved",
  "chance_created",
  "shot_off_target",
  "shot_blocked",
  "foul_won",
  "possession_retained",
  "out_of_play",
  "offside",
  "turnover",
];

/** Outcome bars for the "1,000 realities" chart, best outcomes first, zero counts dropped. */
export function distributionBars(d: Pick<RolloutDistribution, "outcome_counts" | "n">): Bar[] {
  const total = Math.max(1, d.n);
  return d.outcome_counts
    .filter((c) => c.count > 0)
    .sort((a, b) => GOODNESS.indexOf(a.label) - GOODNESS.indexOf(b.label))
    .map((c) => ({ label: c.label, text: OUTCOME_TEXT[c.label], count: c.count, fraction: c.count / total }));
}

/**
 * The "open teammate" for the scripted demo: among viable passes (not the real choice) that are at least even money,
 * the one that gains the most ground (value if it works); falls back to the best decision value.
 */
export function openTeammate(m: Pick<DecisionMoment, "options" | "real_action">): OptionEvaluation | null {
  const real = realOption(m);
  const passes = m.options.filter((o) => o.intent.kind === "pass_player" && o.viable && !(real && sameCall(o.intent, real.intent)));
  const even = passes.filter((o) => o.p_success >= 0.5).sort((a, b) => b.value_if_success - a.value_if_success);
  if (even[0]) return even[0];
  passes.sort((a, b) => b.decision_value - a.decision_value);
  return passes[0] ?? null;
}

/** One entry per distinct choice (the engine scores several carry directions; the panel shows the best). */
export function dedupeOptions(options: readonly OptionEvaluation[]): OptionEvaluation[] {
  const best = new Map<string, OptionEvaluation>();
  for (const o of options) {
    const k = o.intent.kind;
    const key = k === "pass_player" ? `p:${o.intent.target_player}` : k === "carry" || k === "shot" ? k : o.label;
    const cur = best.get(key);
    if (!cur || o.decision_value > cur.decision_value) best.set(key, o);
  }
  return [...best.values()];
}

export function pct(p: number): string {
  return `${Math.round(p * 100)}%`;
}

/** Most recent on-ball event (pass or shot) at or before t, within `windowS`: the fan-initiated rewind point. */
export function lastOnBall(events: readonly MatchEvent[], t: number, windowS = 20): MatchEvent | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i]!;
    if (e.t > t + 0.05) continue;
    if (t - e.t > windowS) return null;
    if ((e.type === "pass" || e.type === "shot") && !e.rollout_id && e.player) return e;
  }
  return null;
}
