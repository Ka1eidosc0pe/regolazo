import { describe, expect, it } from "vitest";
import type { DecisionMoment, MatchEvent, OptionEvaluation } from "../src/types/generated/contracts.ts";
import {
  dedupeOptions,
  distributionBars,
  eventCard,
  formatClock,
  lastOnBall,
  markerFraction,
  openTeammate,
  passCard,
  passDifficulty,
  realOption,
  realOutcome,
  shotCard,
  speedBadge,
  verdictFraming,
} from "../src/ui/logic.ts";
import { captionHoldS } from "../src/ui/narration.ts";

function ev(type: MatchEvent["type"], t: number, qualifiers: Record<string, unknown>, extra: Partial<MatchEvent> = {}): MatchEvent {
  return {
    schema_version: "1.0.0",
    id: `ev_${t}`,
    index: 0,
    source: "ground_truth",
    confidence: null,
    type,
    period: 1,
    t,
    tick: Math.round(t * 25),
    duration_s: 0,
    team: "home",
    player: "H08",
    track_id: null,
    secondary_track_id: null,
    rollout_id: null,
    location: [0, 0, 0],
    end_location: [30, 0, 0],
    possession_id: 1,
    possession_team: "home",
    play_pattern: "regular_play",
    phase_of_play: "progression",
    under_pressure: false,
    counterpress: false,
    related_events: [],
    qualifiers: qualifiers as unknown as MatchEvent["qualifiers"],
    ...extra,
  } as MatchEvent;
}

function opt(id: string, kind: OptionEvaluation["intent"]["kind"], p: number, dv: number, extra: Partial<OptionEvaluation> = {}, target: string | null = null): OptionEvaluation {
  return {
    option_id: id,
    intent: { kind, player: "A07", target_player: target, target: null, path: [], curve: 0, power: 0.5, pass_technique_override: null, shot_technique_override: null, source: "ai" },
    resolved: null,
    label: id,
    label_key: `option.${kind}`,
    label_args: {},
    p_success: p,
    p_success_baseline: null,
    value_if_success: dv * 2,
    xg: null,
    xt_gain: null,
    decision_value: dv,
    viable: true,
    is_real: false,
    ...extra,
  };
}

describe("speed badges (docs/event_taxonomy.md thresholds)", () => {
  it("shows a sprint badge only above 7.0 m/s = 25.2 km/h", () => {
    expect(speedBadge(7.0, 10)).toBeNull();
    const b = speedBadge(7.01, 10);
    expect(b?.kind).toBe("sprint");
    expect(b?.kmh).toBeCloseTo(25.236, 2);
  });
  it("shows top speed at >= 9 m/s and a new personal best", () => {
    expect(speedBadge(9.2, 9.1)?.kind).toBe("top_speed");
    expect(speedBadge(9.2, 9.5)?.kind).toBe("sprint");
    expect(speedBadge(8.9, 0)?.kind).toBe("sprint");
  });
});

describe("cards", () => {
  it("pass card: completed long passes with distance and xPass difficulty", () => {
    const c = passCard(ev("pass", 10, { outcome: "complete", length_m: 37.4, xpass: 0.42, recipient: "H09", technique: "lofted" }));
    expect(c).toMatchObject({ kind: "pass", passer: "H08", recipient: "H09", difficulty: "Difficult", technique: "lofted" });
    expect(c?.distanceM).toBeCloseTo(37.4);
  });
  it("no card for short or incomplete passes, unless key passes", () => {
    expect(passCard(ev("pass", 1, { outcome: "complete", length_m: 8 }))).toBeNull();
    expect(passCard(ev("pass", 1, { outcome: "incomplete", length_m: 40 }))).toBeNull();
    expect(passCard(ev("pass", 1, { outcome: "complete", length_m: 8, is_key_pass: true }))).not.toBeNull();
  });
  it("shot card: xG and shot speed in km/h", () => {
    const c = shotCard(ev("shot", 5464.88, { outcome: "goal", xg: 0.05, ball_speed_mps: 14.14 }));
    expect(c?.xg).toBeCloseTo(0.05);
    expect(c?.speedKmh).toBeCloseTo(50.9, 1);
    expect(c?.outcome).toBe("goal");
    expect(eventCard(ev("duel", 1, {}))).toBeNull();
  });
  it("difficulty bands", () => {
    expect(passDifficulty(0.9)).toBe("Routine");
    expect(passDifficulty(0.7)).toBe("Testing");
    expect(passDifficulty(0.2)).toBe("Audacious");
    expect(passDifficulty(null)).toBeNull();
  });
});

describe("clock and timeline", () => {
  it("formats first half, second half and stoppage time", () => {
    expect(formatClock(65, 1, null)).toBe("01:05");
    expect(formatClock(2800, 1, null)).toBe("45:00 +2");
    expect(formatClock(2900 + 125, 2, 2900)).toBe("47:05");
  });
  it("places markers proportionally and clamps", () => {
    expect(markerFraction(2796.4, 5592.8)).toBeCloseTo(0.5);
    expect(markerFraction(-1, 100)).toBe(0);
    expect(markerFraction(10, 0)).toBe(0);
  });
});

describe("rewind logic", () => {
  const real = opt("carry", "carry", 0.95, 0.014, { is_real: true });
  const barnes = opt("barnes", "pass_player", 0.55, -0.0, {}, "A09");
  const hernandez = opt("hern", "pass_player", 0.99, 0.016, {}, "A10");
  hernandez.value_if_success = 0.016;
  barnes.value_if_success = 0.026;
  const longshot = opt("dj", "pass_player", 0.18, -0.02, { viable: false }, "A11");
  const moment = { options: [real, barnes, hernandez, longshot], real_action: { intent: real.intent } } as unknown as DecisionMoment;

  it("finds the real option by flag, else by the real intent", () => {
    expect(realOption(moment)?.option_id).toBe("carry");
    const unflagged = { options: [{ ...hernandez }, { ...barnes }], real_action: { intent: barnes.intent } } as unknown as DecisionMoment;
    expect(realOption(unflagged)?.option_id).toBe("barnes");
  });
  it("picks the open teammate: even money or better, most ground gained", () => {
    expect(openTeammate(moment)?.option_id).toBe("barnes");
  });
  it("frames the verdict without judging the choice up front", () => {
    expect(verdictFraming(hernandez, real, { primary_success: true, label: "possession_retained" })).toBe("as_expected");
    expect(verdictFraming(barnes, real, { primary_success: true, label: "possession_retained" })).toBe("bold_and_paid_off");
    expect(verdictFraming(longshot, real, { primary_success: true, label: "goal" })).toBe("beat_the_odds");
    expect(verdictFraming(hernandez, real, { primary_success: false, label: "turnover" })).toBe("unlucky");
    expect(verdictFraming(real, real, { primary_success: true, label: "goal" })).toBe("same_call_as_reality");
  });
  it("maps the Scout consequence to the real outcome", () => {
    expect(realOutcome({ consequence: { kind: "goal", xt_swing: 0, horizon_s: 10, event_ids: [] } })).toBe("goal");
    expect(realOutcome({ consequence: { kind: "turnover_to_chance_against", xt_swing: 0, horizon_s: 10, event_ids: [] } })).toBe("turnover");
  });
  it("dedupes carry directions to the best one", () => {
    const c2 = opt("carry2", "carry", 0.95, 0.02);
    const out = dedupeOptions([real, c2, barnes]);
    expect(out.map((o) => o.option_id).sort()).toEqual(["barnes", "carry2"]);
  });
  it("sorts distribution bars best first and drops zeros", () => {
    const bars = distributionBars({
      n: 1000,
      outcome_counts: [
        { label: "turnover", count: 290 },
        { label: "goal", count: 98 },
        { label: "offside", count: 0 },
        { label: "shot_saved", count: 264 },
      ],
    });
    expect(bars.map((b) => b.label)).toEqual(["goal", "shot_saved", "turnover"]);
    expect(bars[0]!.fraction).toBeCloseTo(0.098);
  });
  it("fan-initiated rewind uses the last on-ball event", () => {
    const evs = [ev("pass", 10, {}), ev("duel", 11, {}), ev("shot", 12, {}), ev("pass", 30, {})];
    expect(lastOnBall(evs, 13)?.type).toBe("shot");
    expect(lastOnBall(evs, 11.5)?.t).toBe(10);
    expect(lastOnBall(evs, 100)).toBeNull();
  });
  it("captions hold long enough to read", () => {
    expect(captionHoldS("Goal!")).toBe(3.5);
    expect(captionHoldS(Array(40).fill("word").join(" "))).toBe(9);
  });
});
