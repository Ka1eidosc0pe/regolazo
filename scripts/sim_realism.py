"""Realism gate (PLAN.md Step 3): simulate many matches and report against real-data targets.

    uv run python scripts/sim_realism.py --matches 200 --workers 9 [--out docs/sim_realism.md] [--json PATH]

Writes docs/sim_realism.md: target bands from analytics/artifacts/realism_targets.json (StatsBomb / Metrica), the
simulated means with pass / fail, event coverage, displayed-vs-realized calibration and performance.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.realism import MatchSummary, calibration, core_values, run_many
from sim.stats import aggregate, targets

ROWS = [
    ("goals_per_match", "Goals per match", "{:.2f}"),
    ("shots_per_team", "Shots per team", "{:.1f}"),
    ("pass_completion", "Pass completion", "{:.1%}"),
    ("passes_per_team", "Passes per team (excl. throw-ins)", "{:.0f}"),
    ("header_share_of_shots", "Header share of shots", "{:.1%}"),
    ("mean_shot_distance_m", "Mean shot distance (m)", "{:.1f}"),
    ("distance_per_outfield_player_km", "Distance per outfield player (km)", "{:.2f}"),
    ("possession_share_sd", "Possession split, match-to-match SD (pp)", "{:.1f}"),
]


def report(res: list[MatchSummary], wall: float, workers: int) -> tuple[str, dict[str, object]]:
    agg = aggregate([r.stats for r in res])
    agg["possession_share_sd"] = float(np.std([100 * r.possession_share_home for r in res]))
    tg = targets()
    lines = [
        "# Simulator realism (Step 3 gate)",
        "",
        f"{len(res)} simulated full matches (seeds {res[0].seed}-{res[-1].seed}), trained analytics models, "
        f"generated squads. Targets: `analytics/artifacts/realism_targets.json` (StatsBomb Open Data men's matches, "
        "Metrica tracking for distance); bands are the file's suggested tolerances for a many-match mean.",
        "",
        "| Metric | Simulated | Target band | |",
        "|---|---|---|---|",
    ]
    passed = {}
    for key, label, fmt in ROWS:
        v = float(agg.get(key, float("nan")))
        lo, hi = tg.get(key, (float("nan"), float("nan")))
        ok = bool(lo <= v <= hi)
        passed[key] = ok
        lines.append(
            f"| {label} | {fmt.format(v)} | {fmt.format(lo)} - {fmt.format(hi)} | {'pass' if ok else 'FAIL'} |"
        )
    lines += [
        "",
        f"Also per match: xG {agg['xg_per_match']:.2f}, fouls {agg['fouls_per_match']:.1f}, "
        f"corners {agg['corners_per_match']:.1f}, offsides {agg['offsides_per_match']:.1f}.",
        "",
        "## Displayed vs realized probabilities",
        "",
        "Every AI pass and shot is executed with the player-adjusted probability the engine would show in Rewind; "
        "the bins below compare it with what happened (open-play passes; all shots).",
        "",
        "| Action | Mean displayed | Mean model (avg player) | Realized | n |",
        "|---|---|---|---|---|",
    ]
    for name, rows in (("Pass", [p for r in res for p in r.passes]), ("Shot", [s for r in res for s in r.shots])):
        for disp, model, real, n in calibration(rows):
            lines.append(f"| {name} | {disp:.3f} | {model:.3f} | {real:.3f} | {n} |")
    seen = set().union(*[r.values for r in res])
    core = core_values()
    missing = sorted(core - seen)
    lines += [
        "",
        "## Event coverage",
        "",
        f"{len(core) - len(missing)} of {len(core)} core values (event types, play patterns, phases and every "
        "qualifier enum value outside `OPTIONAL_FOR_COVERAGE`) occur.",
        "",
        ("Missing: " + ", ".join(f"`{m}`" for m in missing)) if missing else "Nothing missing.",
        "",
        "## Performance",
        "",
        f"Mean wall time per 90-minute match (single core, one of {workers} parallel workers): "
        f"{np.mean([r.wall_s for r in res]):.1f} s for {np.mean([r.sim_s for r in res]) / 60:.1f} simulated minutes. "
        f"Whole run: {wall / 60:.1f} min.",
        "",
    ]
    data = {"aggregate": agg, "passed": passed, "missing": missing, "n": len(res)}
    return "\n".join(lines), data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "sim_realism.md")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    t0 = time.time()
    res = run_many(list(range(args.start, args.start + args.matches)), args.workers)
    md, data = report(res, time.time() - t0, args.workers)
    args.out.write_text(md)
    if args.json:
        args.json.write_text(json.dumps(data, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
