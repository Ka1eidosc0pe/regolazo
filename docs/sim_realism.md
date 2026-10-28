# Simulator realism (Step 3 gate)

200 simulated full matches (seeds 1000-1199), trained analytics models, generated squads. Targets: `analytics/artifacts/realism_targets.json` (StatsBomb Open Data men's matches, Metrica tracking for distance); bands are the file's suggested tolerances for a many-match mean.

| Metric | Simulated | Target band | |
|---|---|---|---|
| Goals per match | 3.12 | 2.27 - 3.08 | FAIL |
| Shots per team | 12.8 | 10.7 - 14.5 | pass |
| Pass completion | 81.0% | 75.1% - 83.0% | pass |
| Passes per team (excl. throw-ins) | 522 | 405 - 548 | pass |
| Header share of shots | 14.7% | 12.5% - 20.9% | pass |
| Mean shot distance (m) | 17.1 | 14.8 - 18.1 | pass |
| Distance per outfield player (km) | 10.68 | 9.32 - 11.39 | pass |
| Possession split, match-to-match SD (pp) | 12.0 | 8.1 - 15.1 | pass |

Also per match: xG 3.08, fouls 21.6, corners 6.5, offsides 2.4.

## Displayed vs realized probabilities

Every AI pass and shot is executed with the player-adjusted probability the engine would show in Rewind; the bins below compare it with what happened (open-play passes; all shots).

| Action | Mean displayed | Mean model (avg player) | Realized | n |
|---|---|---|---|---|
| Pass | 0.503 | 0.653 | 0.475 | 39783 |
| Pass | 0.819 | 0.857 | 0.791 | 39783 |
| Pass | 0.904 | 0.915 | 0.868 | 39783 |
| Pass | 0.956 | 0.960 | 0.941 | 39783 |
| Pass | 0.989 | 0.988 | 0.985 | 39784 |
| Shot | 0.021 | 0.024 | 0.035 | 1023 |
| Shot | 0.049 | 0.053 | 0.062 | 1022 |
| Shot | 0.079 | 0.084 | 0.080 | 1023 |
| Shot | 0.133 | 0.138 | 0.108 | 1022 |
| Shot | 0.308 | 0.305 | 0.319 | 1023 |

## Event coverage

230 of 231 core values (event types, play patterns, phases and every qualifier enum value outside `OPTIONAL_FOR_COVERAGE`) occur.

Missing: `line_step_up.outcome:trap_success`

## Performance

Mean wall time per 90-minute match (single core, one of 8 parallel workers): 117.6 s for 95.8 simulated minutes. Whole run: 50.0 min.
## Known limits (accepted for the submission)

- Goals per match 3.12 sit 1% above the band (xG per match 3.08; realized shot conversion tracks the displayed
  probability).
- `line_step_up.outcome:trap_success` never occurs: offside calls come from passers misreading the line, not from
  coordinated step-ups, so no pass flagged offside falls inside a step-up window.
- Matches are more one-sided than real football: on a 30-match sample shots per team have an SD of 9.8 (real 5.5),
  home shot share an SD of 0.32 (real 0.17) and goal difference an SD of 2.7 (real 1.8). Team style (directness) and the
  team level spread drive it; a milder style term plus a narrower level range brought these down to 8.0, 0.25 and 2.1 in
  a 27-match test but was not gated in time, so it is not in this build.
- Measured in-match pass calibration: displayed probability runs about 2-3 points above the realized rate in the
  middle bins; about 3 % of passes drawn to succeed get a noise realization no teammate can reach. The controlled
  consistency tests (500 realities per scene) pass.

Engine for the server: `REGOLAZO_ENGINE=sim.engine:create_engine` (recordings and fork snapshots under
`$REGOLAZO_DATA_DIR/matches/<match_id>/`).
