# ReGolazo Event Taxonomy (frozen with schema v1.0.0)

> Implemented in `shared/schemas/events.py` and `shared/schemas/enums.py`. Where this document and the schema differ, the schema wins. Pydantic field names: `PassQ.length_class`, `ShotQ.situation`, `ball_speed_mps` for speeds. The common envelope `id` is an `ev_…` string, not a uuid. Directions: convert a world-frame `angle_rad` with `shared.geometry.angle_to_attacking_frame` before applying the back-pass / switch rules.

Ground truth is emitted by the simulator; the Spotter reproduces Tier A (video) and Tier B (tracking) as `source=detected`; the Analyst derives Tier C.

**Source basis:** StatsBomb Open Data Events spec v4.0.0 (read directly from the PDF in `statsbomb/open-data/doc`), cross-checked with SPADL (socceraction), Wyscout/FBref progressive-action conventions and FIFA/Catapult speed zones.
**Marking:** `[SB]` = taken from StatsBomb spec. `[SPADL]`/`[FBref]`/`[Wyscout]`/`[FIFA]` = taken from that source. `[CONV]` = our convention (where no source defines it, or the source gives only an image/qualitative definition). `[INVENTED]` = a numeric threshold we made up; tune against sim output and mark clearly in config (all thresholds live in one `thresholds.py`/YAML, never inlined).
Note: the SB spec defines Cross (App. 6) and Cutback (App. 5) zones only as images, and Pressure (App. 7) with no distance rule; those are `[CONV]` below.

---
## 1. Coordinates, units, time

| Item | Convention | Why |
|---|---|---|
| Pitch | 105 x 68 m (FIFA recommended; SPADL uses same) | matches SPADL, easy half-lengths 52.5/34 |
| Origin | **Pitch centre (centre spot)** | symmetric: flipping team direction is `(x,y)->(-x,-y)`; goals at x = ±52.5 |
| Axes (attacking frame, used by thresholds below) | `x` toward the acting team's target goal, `y` toward the acting team's **left** (when facing +x), `z` up (m). Right-handed. World frame: x toward the goal at +52.5, y = +34 is the far touchline from the main camera. | handedness-preserving mapping from SB |
| Event frame | **Contracts store world-frame coordinates** (same frame as `SimState`; see docs/ARCHITECTURE.md §3.1). Every threshold in this document is written in the **acting team's attacking frame** (attacking +x, +y = left); convert with `shared.geometry.to_attacking_frame` before applying it. | one frame across sim, renderer, Spotter and overlays; models and thresholds get the attacking frame from one tested helper |
| Ball height | `z` of ball centre; ball radius 0.11 m (resting z≈0.11) | |
| Units | metres, seconds, m/s, m/s², radians | km/h only in UI |
| Time | `t` = **continuous match seconds since the first kickoff** (float) + `period` (1, 2) + `tick` (sim index @25 Hz, ground truth only). Windows in this document ("within 5 s") are differences of `t`. | one timeline for streaming, seek and replay |
| Pitch landmarks (m, centre origin, attacking goal at +x) | goal line x=52.5; goal posts y=±3.66; crossbar z=2.44; penalty area x≥36.0, \|y\|≤20.16; goal area x≥47.0, \|y\|≤9.16; penalty spot (41.5,0); centre circle r 9.15; thirds: defensive x<-17.5, middle ±17.5, final x>17.5 | FIFA laws |
| Distance to goal | Euclidean to goal centre `(52.5, 0)` | used in progressive tests |

### StatsBomb (120 x 80, origin top-left, y down) -> ReGolazo
```
x_m = (x_sb - 60) * 105/120          # = (x_sb-60)*0.875
y_m = (40 - y_sb) * 68/80            # = (40-y_sb)*0.85 ; y_sb=0 is the attacker's LEFT touchline => +y_m = left
z_m = z_sb * 0.9144                  # shot end_location z is in yards-like units (goal height 2.67 -> 2.44 m)
```
- SB `location` is already normalised so the acting team attacks toward x=120 -> no flipping needed. Freeze-frame players are in the same frame.
- Distortion: the uniform scale makes goal width 6.8 m (vs 7.32) and penalty-box width 37.4 m (vs 40.32). Fine for model training; for shot end locations use the goal-plane mapping `y_goal_m = (40 - y_sb) * 7.32/8`, `z_m = z_sb * 2.44/2.67`. Optional: piecewise-linear mapping via pitch landmarks if finer fidelity is needed.
- Recompute lengths/angles in metres from converted coordinates (SB `pass.length` is in "yards"; don't reuse). SB `pass.angle` is clockwise-positive (right-positive); ours `atan2(dy,dx)` is counter-clockwise (left-positive): `angle_ours = -angle_sb`.
- Metrica tracking: normalised 0–1 coordinates, x along length, y down: `x_m=(x-0.5)*105`, `y_m=(0.5-y)*68`, then flip per period/team.

---
## 2. Common event envelope (all events)

| Field | Type | Notes |
|---|---|---|
| `id` | uuid | |
| `index` | int | order within match [SB] |
| `period`, `t`, `tick?` | int, float, int | see §1 |
| `type` | `EventType` | §7 |
| `team` | team_id | acting team |
| `player` | player_id | primary actor (null for team-level events, or when the Spotter only has a track) |
| `track_id`, `secondary_track_id` | int? | Spotter track ids of the primary / second actor (detected events only) |
| `rollout_id` | str? | set on events generated inside a Rewind rollout; ids `ev_r_<rollout>_<n>` |
| `location` | `[x,y,z]` | actor/ball start position, **world frame** (see §1); ball z if known |
| `end_location` | `[x,y,z]?` | pass/carry/shot end; shot = 3D |
| `duration` | float s? | |
| `possession_id` | int | §4.7 [SB] |
| `play_pattern` | enum | regular_play, from_corner, from_free_kick, from_throw_in, from_goal_kick, from_keeper, from_kick_off, from_counter, other [SB] |
| `under_pressure` | bool | §4.6 [SB] |
| `counterpress` | bool | on pressure/duel/block/interception/foul/dribbled-past etc. [SB] |
| `related_events` | list[id] | e.g. shot<->block/save, pass<->receipt |
| `source` | `ground_truth|detected` | |
| `confidence` | float? | detected only |
| `outcome` | per-type enum | below |
| `qualifiers` | per-type model (`PassQ`, `ShotQ`...) | discriminated union on `type` |

**Coverage gates.** `shared.schemas.enums.OPTIONAL_FOR_COVERAGE` lists the rare values (e.g. `wayward`, `overhead_kick`, `six_seconds`) that the Step 3 event-coverage gate and the Step 6 per-qualifier gate need not hit; every other value is core.

**Detected events.** Fields the Spotter may not see are nullable in the schema (`PassQ.height/body_part/technique/pass_type`, `ReceiptQ.touch_quality/body_orientation`, `ShotQ.body_part/technique/situation`, `ClearanceQ.body_part`, `PressureQ.pressed_player`) and required for ground truth (`events.GROUND_TRUTH_REQUIRED`).

Defaults: `qualifiers` fields are `Optional` unless marked **req**. "req" = must be set on every ground-truth event of that type; the Spotter may leave non-visual ones null with lower confidence.

---
## 3. Tier A: on-ball events

### 3.1 `pass`
Ball intentionally played by one player toward a teammate or space (including restarts and GK distribution) [SB Pass].

| Item | Spec |
|---|---|
| Actors | `player`=passer **req**; `recipient` (intended receiver; null for clearances/passes into space) **req** for ground truth; `end_location.z` = ball z at arrival/first ground contact |
| Locations | `location` start (xyz at release), `end_location` (x,y,z at receipt, or at out/interception point if not received) **req** |
| Metrics | `length_m` (start->end), `angle_rad` (§1), `ball_speed_ms` (peak within 0.2 s of release), `peak_height_m`, `duration_s` (flight) |
| `outcome` **req** | `complete` (reaches a teammate who controls it), `incomplete` (not intercepted, ball dead/uncontrolled in play, or recipient fails), `intercepted` (opponent touches and gains/redirects, incl. blocks), `out_of_play`, `offside` |
| `height` **req** | `ground` / `low` / `high` (§3.1.1) |
| `length_class` **req** | `short` / `medium` / `long` |
| `body_part` **req** | `right_foot`, `left_foot`, `head`, `other`, `hands` (throw-in, GK throw), `no_touch` (dummy/let-run) |
| `technique` **req** | `standard` (side-foot/push), `driven`, `lofted`, `chipped`, `curled`, `outside_of_boot`, `backheel` (extends [SB] technique: SB only has inswinging/outswinging/straight/through_ball) |
| `pass_type` **req** | `standard`, `cross`, `cutback`, `through_ball`, `layoff`, `flick_on`, `back_pass` (single value, precedence in §3.1.3) |
| `restart` | `kick_off`, `throw_in`, `goal_kick`, `corner`, `free_kick_direct`, `free_kick_indirect`, null=open play |
| `corner_delivery` | `inswinging`, `outswinging`, `straight` (corners only) [SB] |
| `gk_distribution` | `throw`, `punt`, `drop_kick`, `goal_kick`, `pass` (only if passer is GK) |
| Flags (bool) | `is_switch`, `is_progressive`, `is_line_breaking` (+`lines_broken:int`), `is_key_pass`, `is_assist`, `is_shot_assist` (=key pass, alias), `first_time`, `deflected`, `recipient_offside` |
| Links | `assisted_shot_id`; `receipt_id` |

#### 3.1.1 Height, length, angle bands

| Quantity | Value | Basis |
|---|---|---|
| `ground` | peak ball height ≤ **0.30 m** over the whole flight (allows small bobbles; ball diameter 0.22 m) | [SB]: "doesn't come off the ground"; 0.3 m is [INVENTED] tolerance |
| `low` | 0.30 < peak ≤ **1.5 m** (≈ shoulder height of a 1.75–1.85 m player) | [SB]: "under shoulder level at peak"; 1.5 m [CONV] |
| `high` | peak > 1.5 m | [SB] "above shoulder level" |
| `short` | length < **15 m** | [FBref]: 5–15 yd short, 15–30 yd medium, >30 yd long, rounded to metres [CONV] |
| `medium` | 15 ≤ length ≤ **30 m** | |
| `long` | length > **30 m** | |
| Min logged pass length | none (a 1 m tap is a pass); `back_pass` below needs ≥ 3 m | |

Spotter reality check: vision can only estimate height bands coarsely; train/evaluate `height` as 3-class with `low` expected to be the weakest.

#### 3.1.2 Technique operational rules (from ball kinematics; sim emits from intent, Spotter infers or classifies)

| Technique | Rule | Status |
|---|---|---|
| `outside_of_boot` | foot surface = outside; curve bends away from kicking foot | sim ground truth only (not visible at broadcast res); Spotter may emit null |
| `chipped` | launch elevation ≥ **35°**, length ≤ **25 m**, backspin | [INVENTED] |
| `lofted` | launch elevation ≥ **20°** (and not chipped), peak > 1.5 m | [INVENTED] |
| `curled` | max perpendicular deviation of ball path from the straight chord ≥ **1.5 m** (and ≥ 4 % of length) | [INVENTED] |
| `driven` | launch elevation ≤ **12°** and initial speed ≥ **18 m/s** | [INVENTED] |
| `backheel` | contact with heel | [SB backheel flag] |
| `standard` | otherwise | |

Precedence: `outside_of_boot > backheel > chipped > curled > lofted > driven > standard`.

#### 3.1.3 Pass type definitions (precedence: `cutback > cross > through_ball > layoff > flick_on > back_pass > standard`)

| Type | Definition | Numbers |
|---|---|---|
| `cutback` | Ground/low pass played backwards from near the opponent's goal line into the penalty area [SB: "passed low backwards, within the opposition's penalty box"; zone image App. 5] | Origin: `x ≥ 46.0` and `\|y\| ≥ 9.0`; destination: `36.0 ≤ x ≤ origin.x - 3.0` and `\|y_end\| ≤ 14`; peak height ≤ 1.5 m [CONV zones] |
| `cross` | Ball delivered from a wide area into the box area [SB App. 6 zones] | Origin: `\|y\| ≥ 18.0` and `x ≥ 20.0`; destination: `x ≥ 33.0` and `\|y_end\| ≤ 20.16`; ball moves inward ≥ 8 m laterally (`\|y_end\| ≤ \|y_start\| - 8`); length ≥ 8 m; not a cutback. Corner and free-kick deliveries qualify if the destination test passes (origin test waived) [CONV zones] |
| `through_ball` | Pass that cuts the last line of defence [SB technique "Through Ball"] and puts the receiver beyond it | Ball path crosses the line x = x of the last outfield opponent; recipient (at arrival) has `x ≥ last_defender.x + 1.0`; height ≤ `low` unless lofted through-ball (then `lofted` technique + this type); length ≥ 10 m; recipient not offside at release (otherwise outcome `offside` but type still `through_ball`) [CONV numbers] |
| `layoff` | One-touch pass played backward/lateral by a player who just received a forward/aerial ball, to a trailing teammate | `first_time` true; pass bearing differs from incoming ball bearing by ≥ 90°; receiver within 15 m and ≥ 2 m further from the opponent goal than the layoff player [INVENTED] |
| `flick_on` | Header or one-touch deflection that redirects an incoming aerial ball forward, typically from goal kicks/long balls | `first_time`, incoming ball peak height > 1.5 m, outgoing bearing within ±60° of +x, change of bearing ≤ 60° from incoming [INVENTED] |
| `back_pass` | Pass directed backward | `\|angle_rad\| ≥ 135°` w.r.t. +x and length ≥ 3 m [CONV; SB `angle` π = straight back] |
| `standard` | Everything else | |

#### 3.1.4 Pass flags

| Flag | Definition | Numbers |
|---|---|---|
| `is_switch` | Pass changing sides of the pitch [SB: ≥ 50 % of pitch width, >40 yd of 80 yd] | `\|y_end - y_start\| ≥ 34.0 m` (50 % of 68 m) [SB scaled] |
| `is_progressive` | Moves ball substantially closer to opponent goal centre. Distance-to-goal reduction between start and end, by start location [Wyscout] | start & end in own half: ≥ **30 m** closer; start own half, end opp half: ≥ **15 m**; both in opp half: ≥ **10 m**. Also true for any pass ending in the opponent penalty area that started outside it [FBref]. Excludes restarts `goal_kick`, `kick_off`. Computed for all attempts; Analyst filters `complete` when counting. [Wyscout/FBref, metres rounded] |
| `is_line_breaking` | Ball path passes through an opposition defensive line to a receiver located beyond it. Lines = back/mid/front groups from opponent formation detection; line x = mean x of the group; line lateral span = [min y -1 m, max y +1 m] | pass start `x < line.x` and receiver at arrival `x > line.x`, and path crosses the line's lateral span. `lines_broken` = number of lines passed (0–3). Complete passes only for `true` (Analyst); sim stores for all. [CONV] |
| `is_key_pass` (= `is_shot_assist`) | Last pass before a shot by a teammate, same possession | shot ≤ **5 s** after pass end, no intervening opponent control; shooter may take ≤ 1 dribble/carry [Opta-like; 5 s INVENTED] |
| `is_assist` | Key pass whose shot is a goal (not own goal) | [Opta/SB goal-assist] |
| `first_time` | Release within 0.4 s of receiving (touch-to-release) | [INVENTED] |

### 3.2 `ball_receipt`
Receipt (or attempted receipt) of a pass [SB Ball Receipt: only emits `incomplete` outcome in SB; we log all].

| Item | Spec |
|---|---|
| Actors | `player`=receiver; `passer` |
| `outcome` **req** | `complete` (controlled), `incomplete` (receiver fails/does not reach) |
| `touch_quality` **req** | `clean` (ball within 1.5 m of player 0.5 s after first touch, not lost), `loose` (1.5–3 m), `miscontrol` (> 3 m, out of play within 1 s, or opponent controls within 1 s) [INVENTED thresholds] |
| `body_orientation` **req** | by body facing angle θ w.r.t. +x (own attacking direction): `open` \|θ\| < 60°; `half_turn` 60° ≤ \|θ\| ≤ 120°; `back_to_goal` \|θ\| > 120° [INVENTED]. Spotter: from player heading/torso orientation estimated from motion and detector pose cue |
| Other | `first_touch_body_part`, `pressure_distance_m` (nearest opponent at touch), `aerial` (bool) |

Receipt is only emitted for a pass whose `recipient` makes a touch within 1.0 s of ball arrival, or when the pass is `incomplete` with a nearby intended recipient. `miscontrol` also emits a stand-alone `miscontrol` event (SB Miscontrol).

### 3.3 `carry`
Player moves with the ball under close control [SB Carry].

| Item | Spec |
|---|---|
| Rule | Same player controls the ball (ball within **2.0 m** of the player and no other player's touch in between) from receipt/take-on end until the next on-ball action. Emitted when path length ≥ **3.0 m** [SPADL `min_dribble_length`=3 m; 2.0 m ball radius INVENTED] |
| Fields | `location`/`end_location` (x,y,z≈0), `distance_m` (net), `path_m`, `duration`, `max_speed`, `is_progressive` |
| `is_progressive` | Wyscout-style bands as pass (own-half end: ≥ 30; across halves: ≥ 15; opp half: ≥ 10 m closer to goal) **or** enters the opp penalty area from outside. Carries ending in own half do not qualify [FBref excludes defending-half carries] |
| `outcome` | `retained` (followed by pass/shot/take-on by same player), `lost` (dispossessed/miscontrol/out), `end_of_period` |
| Note | SB v4.0.0 derives carries between consecutive on-ball events of the same player (spec states no minimum length); we apply a 3 m floor to avoid noise |

### 3.4 `take_on` (SB "Dribble")
Attempt by the ball carrier to beat an opponent.

| Item | Spec |
|---|---|
| Trigger | An opponent defender within **2.5 m** of the carrier and in front (projection onto carrier's heading ≥ 0) at the start of a challenge/feint [INVENTED] |
| Actors | `player`=dribbler; `opponent`=defender beaten (nearest defender) |
| `outcome` **req** | `success` (carrier passes the defender: defender's projection onto the carrier heading becomes ≤ -1.0 m, **and** carrier retains the ball ≥ 1.0 s or completes a pass) / `fail` (tackled, dispossessed, ball lost within 1.5 s) [SB: Complete/Incomplete] |
| Qualifiers | `nutmeg` (ball passes between defender's legs), `overrun` (carrier loses ball past defender to another), `no_touch` (ball pushed past defender, player runs after it) [SB] |
| Linked | success also emits `dribbled_past` (defender-side; may be derived), fail emits `duel(tackle)` or `dispossessed` |

### 3.5 `shot`
Attempt to score with any legal body part [SB Shot].

| Item | Spec |
|---|---|
| Actor | `player`=shooter; `keeper` (opposing GK), `blocker?` |
| Locations | `location` (x,y,z at contact); `end_location` 3D: for on-target = point where ball crosses the goal plane (x=52.5), `y ∈ [-3.66,3.66]`, `z ∈ [0,2.44]`; off target = extrapolated goal-plane crossing (outside frame, ground-trajectory); blocked = block point; woodwork = contact point on frame [SB shot end_location x,y,z] |
| `outcome` **req** | `goal`, `saved`, `saved_to_post`, `off_target`, `blocked`, `woodwork` (hit post/bar and not in), `wayward` (> 5 m wide/high or mis-hit) [SB: Goal, Saved, Saved To Post, Off T, Blocked, Post, Wayward; Saved Off T folded into `saved`+`on_target=false`] |
| `on_target` (derived) | goal plane crossed inside frame (goal, saved, saved_to_post). `woodwork` is not on target [Opta] |
| `body_part` **req** | `right_foot`, `left_foot`, `head`, `other` [SB] |
| `technique` **req** | `driven`, `placed`, `finesse`, `chip`, `volley`, `half_volley`, `header`, `diving_header`, `toe_poke`, `backheel`, `overhead_kick` (§3.5.1) |
| `shot_situation` **req** | `open_play`, `counter_attack`, `corner_phase`, `free_kick_phase`, `throw_in_phase`, `direct_free_kick`, `penalty`, `kick_off` (§3.5.2) |
| Metrics | `ball_speed_ms` (peak ball speed within 0.2 s of contact, **req**), `distance_to_goal_m`, `angle_rad` (shooting angle to goal posts), `peak_height_m` |
| Models | `xg`, `psxg` (post-shot xG; null for off-target/blocked) — computed by Analyst, stored on event for ground truth |
| Context flags | `first_time`, `one_on_one`, `open_goal` [SB], `follows_dribble` [SB], `aerial_won` [SB], `deflected` [SB], `under_pressure`, `n_defenders_in_cone`, `gk_position` (`set`,`moving`,`prone` at shot [SB goalkeeper.position]), `freeze_frame` (ref to SimState snapshot) |
| Links | `key_pass_id`, `block_event`, `gk_event` |

#### 3.5.1 Shot technique rules (SB: Normal, Lob, Volley, Half Volley, Backheel, Overhead Kick, Diving Header; we split Normal and add toe-poke/finesse)

| Technique | Rule | Status |
|---|---|---|
| `volley` | Ball airborne at contact (z > 0.3 m), no bounce before contact | [SB] |
| `half_volley` | Contact ≤ 0.25 s after a bounce, ball z ≤ 0.5 m | [SB] numbers [INVENTED] |
| `header` / `diving_header` | Body part head; diving = shooter's feet off ground and body horizontal toward ball [SB] | |
| `overhead_kick`, `backheel` | Back to goal at contact / heel contact [SB] | |
| `chip` (= SB `Lob`) | launch elevation ≥ 30° and ball passes over keeper at height > 1.8 m, shot distance ≤ 25 m | [INVENTED] |
| `finesse` | curved path, max perpendicular deviation from chord ≥ 1.0 m, placed to far corner, speed < 25 m/s | [INVENTED] |
| `driven` | ball speed ≥ **25 m/s** (90 km/h), no significant curve | [INVENTED] |
| `toe_poke` | contact with toe, short backlift, speed < 15 m/s, often under pressure | sim ground truth; Spotter classifier |
| `placed` | default foot shot: speed < 25 m/s, not curved/chipped | [SB "Normal"] |
Precedence: `overhead_kick > backheel > diving_header > header > volley > half_volley > chip > finesse > toe_poke > driven > placed`.

#### 3.5.2 Shot situation
| Value | Rule |
|---|---|
| `penalty` | Shot is a penalty kick [SB type Penalty] |
| `direct_free_kick` | Shot taken directly from a free kick [SB type Free Kick] |
| `kick_off` | Shot directly from kickoff [SB] |
| `corner_phase` / `free_kick_phase` / `throw_in_phase` | Shot ≤ **10 s** after the restart, same possession (SB `play_pattern` From Corner/Free Kick/Throw In), not a direct shot [window INVENTED] |
| `counter_attack` | Possession qualifies as a counter-attack [SB play_pattern "From Counter"]: began with open-play turnover outside the attacking team's final third; ≥ **75 %** direct toward goal (net displacement toward goal / ball path length); advanced ≥ **16.5 m** (18 yd) toward goal [SB]; shot within **15 s** of the turnover [15 s INVENTED] |
| `open_play` | else |

### 3.6 Defensive events

| `type` | Definition | Actors | Qualifiers / outcome |
|---|---|---|---|
| `duel` | 50/50 contest between opposing players [SB Duel] | `player`, `opponent` | `duel_type` **req**: `tackle` (ground duel to dispossess ball carrier), `aerial` (header contest), `fifty_fifty` (both challenge a loose ball) [SB Duel types Tackle, Aerial Lost, 50/50]. `outcome` **req**: `won` (player ends with possession or knocks ball to teammate), `lost`. `possession_after`: `tackler`, `teammate`, `opponent`, `out_of_play`, `loose` [SB outcomes collapsed: Success In Play/Out, Lost In Play/Out]. `counterpress` flag. Aerial: `aerial_won` loser emitted as separate duel row with `outcome=lost`. Aerial duel fires when ≥ 2 opponents within 2.5 m of ball landing point at ball z ≥ 1.0 m and ≥ 1 jumps/contests. |
| `interception` | Defender reads a pass and gets the ball / redirects it by moving into the lane [SB] | `player`; `passer`; `pass_id` | `outcome`: `won` (gains possession), `success_in_play`, `success_out`, `lost` (touch but ball to opponent). Rule: opponent-pass in flight, defender is first to touch before intended recipient |
| `block` | Defender blocks ball in its path with body [SB] | `player`; `blocked_event` | `block_type`: `shot_block`, `pass_block`, `cross_block`; flags `deflection` (ball direction change < 30°), `save_block` (outfield player blocks a shot that was on target) [SB] |
| `clearance` | Defensive action to remove danger without targeting a teammate [SB] | `player` | `body_part`, `aerial_won`, `outcome`: `out_of_play`, `in_play_safe`, `in_play_danger`; `under_pressure`. Not a `pass` unless a recipient is intended. |
| `ball_recovery` | Gains control of a loose ball (no opponent pass in flight) [SB] | `player` | `offensive` (recovering a ball lost by a teammate [SB]), `recovery_failure` [SB]; loose = no player control for ≥ 0.3 s |
| `dispossessed` | Carrier loses ball to an opponent tackle without attempting a dribble [SB] | `player`, `opponent` | Pair of `duel(tackle,won)` for the tackler |
| `miscontrol` | Player loses ball through a bad touch [SB]| `player` | `aerial_won` [SB]; see §3.2 thresholds |
| `deflection` | Touch by any player (not intended as a pass/block/clearance) that changes ball bearing by ≥ **10°** [CONV] | `player` | `deflection_target`: shot, pass, cross; flag `deflected=true` on the parent pass/shot |
| `foul_committed` | Infringement penalised by referee (offside is not a foul) [SB] | `player`=offender, `opponent`=fouled | `foul_type`: `foul`, `handball`, `dangerous_play`, `dive`, `six_seconds`, `backpass_pickup`; `card`: `none`, `yellow`, `second_yellow`, `red`; `advantage` (bool), `penalty` (bool), `offensive` (in possession) [SB] |
| `foul_won` | Mirror of `foul_committed` (derived, optional) | `player` | `defensive` flag |

### 3.7 `gk_action` (goalkeeper)
`type=gk_action`, `player`=GK, `gk_action` **req** in:

| Value | Definition | Outcome values |
|---|---|---|
| `save` | Stops an on-target shot [SB Shot Saved] | `catch` (collected), `parry_safe` (to teammate/safe area), `parry_danger` (to opponent / rebound in danger area), `parry_out` (out of play), `to_post`, `fumble_saved_twice`; `goal_conceded` is NOT here (shot outcome = goal; GK event `goal_conceded` emitted for stats) |
| `claim` | Catches a cross/high ball in the air or after bounce [SB Collected] | `success`, `fail`, `collected_twice` |
| `punch` | Punches a cross/ball away [SB Punch] | `in_play_safe`, `in_play_danger`, `out_of_play` |
| `sweeper` | Leaves the box area / line to clear or intercept a through ball [SB Keeper Sweeper] | `claim`, `clear`, `fail` |
| `rush_out` | Comes off the line to dispossess a carrier in 1v1 [SB Smother] | `won`, `lost_in_play`, `lost_out` |
| `pick_up` | Collects a back pass / loose ball with hands (rare; legal only non-teammate-kicked) | `success` |
| `goal_conceded` | Marker for a goal against | `no_touch`, `touched_in` |

Qualifiers: `technique`: `standing`, `diving` [SB]; `body_part`: `both_hands`, `left_hand`, `right_hand`, `chest`, `head`, `left_foot`, `right_foot` [SB]; `shot_id`; `position_before`: `set`, `moving`, `prone` [SB]; `dive_distance_m`; `save_xg_prevented` (psxg - 0 for saves; Analyst).
**Distribution** is emitted as `pass` with `gk_distribution` (throw / punt / drop_kick / goal_kick / pass) and `body_part=hands` for throws, as in SB (Keeper Arm, Drop Kick); not a `gk_action`.

### 3.8 Restarts and stoppages

| `type` | Definition | Qualifiers |
|---|---|---|
| `restart` | Ball put back in play by a restart law | `restart_type` **req**: `kick_off`, `throw_in`, `goal_kick`, `corner`, `free_kick_direct`, `free_kick_indirect`, `penalty`, `drop_ball`; `taker` (player), `location`, `awarded_to` (team), `cause_event_id`, `wall_size` (FK), `corner_delivery` (see pass) |
| `ball_out_of_play` | Ball fully crosses touchline or goal line [SB `out`] | `line`: `touchline`, `goal_line`; `last_touch_team`, `last_touch_player`, `location` (x,y at crossing); `result_restart_type` |
| `offside` | Offside infringement; non-pass cases [SB Offside] plus pass cases via pass `outcome=offside` | `offender`, `passer`, `offside_line_x`, `margin_m`, `location` (offender), `restart` = indirect FK for defending team |
| `period_start`/`period_end` | Whistle | `period` |
| `own_goal` | Ball into own net by defending player [SB Own Goal For/Against] | `player` |

**Offside rule (ground truth):** at the instant of pass release, an attacker in the opponent half is offside if any playable body part is nearer the goal line than both the ball and the second-last opponent (GK counts as an opponent), tolerance **0.0 m** in sim; Spotter reports `margin_m` and treats |margin| < 0.3 m as uncertain [0.3 INVENTED].

---
## 4. Numeric definitions shared by Tier A/B/C

### 4.1 Progressive pass / carry
See §3.1.4 and §3.3 (Wyscout distance-to-goal-centre bands, with FBref's "into penalty area" clause). [FBref's own definition = ≥ 10 yd closer than furthest point of last 6 passes; we chose Wyscout bands because they are stateless and per-event.]

### 4.2 Switch of play: §3.1.4 (≥ 34 m lateral).
### 4.3 Cross/cutback/through ball/key pass/assist: §3.1.3–3.1.4.

### 4.4 Carry minimum: §3.3 (3.0 m, ball ≤ 2.0 m from player).

### 4.5 Speed zones [FIFA / Premier League Catapult-style]
| Zone | km/h | m/s |
|---|---|---|
| walk/jog | < 19.8 | < 5.5 |
| `high_intensity_run` | 19.8 – 25.2 | 5.5 – 7.0 |
| `sprint` | > 25.2 | > 7.0 |
(PLAN says > 25 km/h; we use 25.2 = 7.0 m/s exactly, [FIFA].) Speed from tracking = central difference of position at 25 Hz smoothed by Savitzky–Golay/0.5 s moving average [CONV]; physically capped at 12 m/s (discard outliers).

### 4.6 Under pressure & pressure
- `under_pressure` on an on-ball event = the event's time interval overlaps a `pressure` event's `[t, t+duration]` on that player (receipt: if pressure begins before ball arrival ends) [SB App. 7]. For ground truth w/o pressure event (e.g. duels, take-ons), it is true by definition [SB v4].
- `pressure` event (Tier B): see §5.1.

### 4.7 Possession sequence [SB `possession`; SPADL has no equivalent]
A possession starts when a team gains **control** (a pass, carry, shot, or ≥ 1.0 s uninterrupted touch/control of the ball) and ends when: the opponent gains control (a controlled touch as above; a single deflection/block/clearance touch by the opponent after which the original team touches again within **2.0 s** does NOT end it), ball goes out of play (a new possession starts on the restart, even for the same team [SB]), foul/offside whistle, a shot resolves (goal, save held, out), or period ends. Store `possession_id` (int) and `possession_team`. Control timeout: ball uncontrolled > 2.0 s with nobody touching is `loose` and belongs to no team until next control [thresholds [INVENTED]; structure [SB]].

### 4.8 Phase of play (Tier C, per possession-team label every tick; attached at event time)
Priority top-to-bottom; first match wins. `phase_of_play` ∈:

| Phase | Rule |
|---|---|
| `set_piece` | ≤ **10 s** after a corner, free kick, throw-in in the attacking half (x ≥ -10), or penalty, same possession [INVENTED window] |
| `counter_attack` | Possession matches the SB counter definition (§3.5.2) |
| `transition` | ≤ **8 s** after an open-play turnover (either team gains/loses ball), not yet `counter_attack` [8 s INVENTED; counterpress window 5 s is [SB]]. In-possession team labelled `attacking_transition`; out-of-possession `defensive_transition` (see `transition_side`) |
| `final_third` | In possession, ball x ≥ 17.5 |
| `progression` | In possession, ball -17.5 ≤ x < 17.5 |
| `build_up` | In possession, ball x < -17.5 (typically GK/defenders start; includes goal-kick phase) |
| `out_of_possession` | team does not have possession (use `defensive_block`) |
Out-of-possession `defensive_block` ∈ `high_press` (defensive line mean x > 0 and ≥ 3 pressures in last 10 s), `mid_block` (mean line x in [-25, 0]), `low_block` (mean line x < -25 m), with block height measured as mean x of the 4 deepest outfield players [CONV; thresholds INVENTED].

### 4.9 Team shape (Tier C; per frame, outfield players only, excludes GK)
| Metric | Definition |
|---|---|
| `width_m` | max y − min y (also report `width_sd` = std of y) |
| `length_m` | max x − min x (depth, back line to front line), excluding GK |
| `compactness_m2` | area of convex hull of outfield players (smaller = more compact); `stretch_index` = mean distance of outfield players to team centroid |
| `defensive_line_height_m` | mean x of the 4 deepest outfield players (own-goal-origin frame: x + 52.5 = distance from own goal line) |
| `centroid` | mean (x,y) of outfield players |
Formation detection: cluster players into lines (k=3, 1-D k-means on x of outfield players, or Hungarian-match to template formations) separately in possession and out of possession, windowed over ≥ 10 s of that state [CONV].

### 4.10 PPDA and field tilt (per team, per window)
```
PPDA = opp_passes_in_zone / defensive_actions_in_zone      (lower = more intense press)
zone  = opponent's own 60% of pitch from the opponent's goal line, i.e. attacking-team-frame x_pressing >= -10.5 m  (pressing team's attacking 60%)
opp_passes_in_zone = opponent open-play passes (any outcome) with start in zone
defensive_actions  = pressing team's tackles (duel tackle), interceptions, fouls committed, challenges (duel lost) in zone
```
(Opta convention [StatsBomb variant: replaces tackles/interceptions/fouls with `pressure` count].) Also report `PPDA_sb` using `pressure` events. `field_tilt = team_final_third_passes / (team_final_third_passes + opp_final_third_passes)` using passes whose start is in that team's attacking third (`x ≥ 17.5` in own frame) [Opta; touches variant optional].

### 4.11 Possession value (xT)
Tier C. Karun Singh 16x12 grid; `xt_delta = xT(end) - xT(start)` for complete passes/carries; stored on Analyst output, not on `MatchEvent`.

---
## 5. Tier B: off-ball and pressure (derived from tracking)

All Tier B events use player tracks at 25 Hz in world frame, converted to acting team frame. `source=detected` events come from the Spotter's tracking; ground truth from sim.

### 5.1 `pressure` / `counterpress` / `pressing_sequence`

| Item | Spec |
|---|---|
| `pressure` definition | Defender applies pressure to an opponent who is receiving, carrying or releasing the ball [SB]. Opponent distance rule is **not specified** by SB (human-collected); our operational rule: **defender (non-GK) within 5.0 m of the pressed player AND closing** (rate of distance change ≤ -0.5 m/s, or distance ≤ 2.0 m), **and** defender speed ≥ 1.0 m/s toward the pressed player or being within 2.0 m; sustained ≥ **0.4 s**; event spans the interval; `duration`=interval length. Pressed player is the ball carrier, a recipient within 1.0 s before/after ball arrival, or the releaser within 0.5 s after release. [5.0 m from brief; others INVENTED] |
| Actors | `player`=presser; `pressed_player`; `pressed_role`: `carrier`, `receiver`, `releaser` |
| Qualifiers | `min_distance_m`, `closing_speed_ms` (mean), `counterpress` (bool), `n_pressers` (concurrent pressers within 5 m on same target), `pressure_outcome`: `forced_turnover` (opponent loses ball ≤ 3 s), `forced_hurried_action` (pass/clearance/out within 2 s), `none` |
| `counterpress` | Pressure (or duel/block/interception/foul/dribbled past) by the team that just lost the ball, within **5 s** of an **open-play turnover** [SB] |
| `pressing_sequence` | ≥ 2 pressure events in the same possession by ≥ 2 distinct pressers with inter-pressure gaps ≤ **3 s**, spanning the pressed team's possession; ends at: turnover (`success`), pressed team plays ≥ 10 m forward past the press (`broken`), ball out (`out`), or whistle. Fields: `n_pressures`, `duration`, `pressing_team_players`, `outcome` [INVENTED] |

### 5.2 `off_ball_run`

Candidate run (attacking team in possession; runner is not the ball carrier): speed ≥ **4.0 m/s** held ≥ **1.0 s** (extend until speed < 2.5 m/s), net displacement ≥ **5 m**, runner distance to ball ≥ 3 m throughout [INVENTED]. `run_type` classification (first match):

| `run_type` | Operational rule from tracking | Extra qualifiers |
|---|---|---|
| `run_in_behind` | Heading within ±45° of +x; starts with `x ≤ last_defender.x + 1.0` (level or behind line) and ends with `x ≥ last_defender.x + 2.0` where `last_defender` = last outfield opponent at run start; ≥ 8 m displacement | `targeted` (ball played to runner within 3 s of run end), `offside_at_release` |
| `overlap` | Ball carrier (teammate) on the same lateral side; at run start runner is level with or behind the carrier (`x_run ≤ x_carrier + 2`) and **outside** the carrier (`\|y_run\| ≥ \|y_carrier\| + 2` toward the touchline); runner passes the carrier's x by ≥ 3 m ahead while staying outside | `carrier_id`, `targeted`, `lateral_gap_m` |
| `underlap` | Same as overlap but runner passes **inside** the carrier (`\|y_run\| ≤ \|y_carrier\| - 2` toward the centre) | `carrier_id`, `targeted` |
| `checking_run` | Runner moves toward the ball carrier: distance to ball decreases by ≥ 4 m over the run, heading within 45° of the line to the ball, start distance ≥ 8 m, end distance ≤ 20 m | `targeted`, `receives_ball` |
| `decoy` | Intent-based; ground truth emits when AI marks the run as a decoy. Detection rule: not `targeted`, speed ≥ **5.5 m/s** (HIR) for ≥ 1 s, ≥ 1 defender (nearest, within 4 m) follows the runner (moves ≥ 3 m in a direction within 60° of the runner's) **and** a pass is released ≤ 2 s after the run begins into the space the follower left (pass end within 12 m of follower's start position) | `dragged_defender_id`, `space_exploited` |
| `support` | none of above, runs toward ball side within 15 m | |
Common qualifiers: `start/end location`, `max_speed`, `distance_m`, `duration`, `ball_possession_team`, `outcome`: `targeted_received`, `targeted_missed`, `not_targeted`.

### 5.3 `sprint`, `high_intensity_run`, `acceleration`

| Event | Rule | Fields |
|---|---|---|
| `sprint` | Speed > 7.0 m/s (25.2 km/h) for ≥ **1.0 s**, merging gaps < 0.5 s [FIFA/Catapult; 1.0 s INVENTED] | `player`, `t_start`, `duration`, `distance_m`, `peak_speed_ms`, `start/end location` |
| `high_intensity_run` | Speed 5.5–7.0 m/s for ≥ **1.0 s**, peak < 7.0 m/s (else it's a sprint) [FIFA zone; 1.0 s INVENTED] | same |
| `acceleration` | `a ≥ +3.0 m/s²` held ≥ **0.5 s** (or `deceleration`: `a ≤ -3.0`), Δspeed ≥ 1.5 m/s [Catapult-style 3 m/s²; rest INVENTED] | `accel_type`: `acceleration`, `deceleration`; `peak_accel_ms2`, `speed_start/end` |
| `top_speed`, `distance_covered` | Running aggregates from tracking, not events (`PlayerPhysicalSummary` per match / per minute). Badge triggers: top speed ≥ 9.0 m/s, distance milestones every 1 km [badge thresholds INVENTED] | |

### 5.4 `line_step_up` (defensive line step-up / offside trap)

| Item | Rule |
|---|---|
| `line_step_up` | Back-line (≥ 3 defenders, lateral spread ≤ 25 m) mean x advances ≥ **4.0 m** within **2.0 s**, with ≥ 3 defenders each moving ≥ 1.5 m/s toward +x, speed std among them < 1.0 m/s (coordinated) [INVENTED] |
| `offside_trap` (flag/outcome) | A `line_step_up` that completes ≤ 1.5 s before an opponent pass release, with an attacker beyond the new line at release => `outcome=trap_success` if `offside` called; `trap_beaten` if receiver gets beyond line onside or the pass is delivered behind (attackers onside). `line_step_up` without an attacker at risk = `outcome=reset` |
Fields: `line_player_ids`, `line_height_before/after`, `ball_carrier_pressure`, `outcome`: `trap_success`, `trap_beaten`, `reset`.

---
## 6. Tier assignment

| Event / state | Tier | Produced by | Notes for Spotter |
|---|---|---|---|
| pass (+height, length, technique, type, flags) | A | Spotter from video | height from 3D ball track; technique partly classifier; `outside_of_boot`, `toe_poke` likely null |
| ball_receipt, carry, take_on | A | Spotter | orientation = coarse (3 classes) |
| shot (+body part, technique, situation, speed, outcome) | A | Spotter | xG/psxg from Analyst |
| duel (tackle/aerial/50-50), interception, block, clearance, ball_recovery, dispossessed, miscontrol, deflection, foul_committed/won | A | Spotter | |
| gk_action (save, claim, punch, sweeper, rush_out), GK distribution (pass) | A | Spotter | |
| restart, ball_out_of_play, offside, own_goal, period_start/end | A | Spotter | restarts from ball/player-position state machine |
| pressure, counterpress | B | Spotter from tracking | rule in §5.1 |
| pressing_sequence | B | Spotter from tracking | |
| off_ball_run (all run types) | B | Spotter from tracking | `decoy` is the weakest; report P/R separately |
| sprint, high_intensity_run, acceleration, distance, top speed | B | Spotter from tracking | metrics, not events (aggregates) |
| line_step_up / offside_trap | B | Spotter from tracking | |
| possession_sequence, phase_of_play, defensive_block | C | Analyst | from events + tracking |
| formation (in/out of possession), team width/length/compactness, defensive line height | C | Analyst | per-frame measures |
| PPDA, field tilt, xT flow, momentum, Control-vs-Chaos | C | Analyst | |
| xG, psxg, xPass, pass difficulty, pitch control | C (models) | Analyst | |

Tier C is also emitted in ground truth (as time series / `PhaseState` records), not as `MatchEvent`s, except `possession_id` and `phase_of_play`, which are stamped on every event.

---
## 7. Shared enums (compact; snake_case)

- **`event_type`**: `pass`, `ball_receipt`, `carry`, `take_on`, `shot`, `duel`, `interception`, `block`, `clearance`, `ball_recovery`, `dispossessed`, `miscontrol`, `deflection`, `foul_committed`, `foul_won`, `gk_action`, `restart`, `ball_out_of_play`, `offside`, `own_goal`, `period_start`, `period_end`, `pressure`, `pressing_sequence`, `off_ball_run`, `sprint`, `high_intensity_run`, `acceleration`, `line_step_up`
- **`team_side`**: `home`, `away`
- **`body_part`**: `right_foot`, `left_foot`, `head`, `other`, `hands`, `no_touch` (GK: `left_hand`, `right_hand`, `both_hands`, `chest`)
- **`pass_height`**: `ground`, `low`, `high`
- **`pass_length`**: `short`, `medium`, `long`
- **`pass_technique`**: `standard`, `driven`, `lofted`, `chipped`, `curled`, `outside_of_boot`, `backheel`
- **`pass_type`**: `standard`, `cross`, `cutback`, `through_ball`, `layoff`, `flick_on`, `back_pass` (+ boolean `is_switch`)
- **`corner_delivery`**: `inswinging`, `outswinging`, `straight`
- **`gk_distribution`**: `throw`, `punt`, `drop_kick`, `goal_kick`, `pass`
- **`pass_outcome`**: `complete`, `incomplete`, `intercepted`, `out_of_play`, `offside`
- **`receipt_outcome`**: `complete`, `incomplete`
- **`touch_quality`**: `clean`, `loose`, `miscontrol`
- **`body_orientation`**: `open`, `half_turn`, `back_to_goal`
- **`carry_outcome`**: `retained`, `lost`, `end_of_period`
- **`take_on_outcome`**: `success`, `fail`
- **`shot_technique`**: `driven`, `placed`, `finesse`, `chip`, `volley`, `half_volley`, `header`, `diving_header`, `toe_poke`, `backheel`, `overhead_kick`
- **`shot_situation`**: `open_play`, `counter_attack`, `corner_phase`, `free_kick_phase`, `throw_in_phase`, `direct_free_kick`, `penalty`, `kick_off`
- **`shot_outcome`**: `goal`, `saved`, `saved_to_post`, `off_target`, `blocked`, `woodwork`, `wayward`
- **`gk_position`** (at shot): `set`, `moving`, `prone`
- **`duel_type`**: `tackle`, `aerial`, `fifty_fifty`
- **`duel_outcome`**: `won`, `lost`
- **`possession_after`**: `tackler`, `teammate`, `opponent`, `out_of_play`, `loose`
- **`block_type`**: `shot_block`, `pass_block`, `cross_block`
- **`clearance_outcome`**: `out_of_play`, `in_play_safe`, `in_play_danger`
- **`foul_type`**: `foul`, `handball`, `dangerous_play`, `dive`, `six_seconds`, `backpass_pickup`
- **`card`**: `none`, `yellow`, `second_yellow`, `red`
- **`gk_action`**: `save`, `claim`, `punch`, `sweeper`, `rush_out`, `pick_up`, `goal_conceded`
- **`gk_outcome`**: `catch`, `parry_safe`, `parry_danger`, `parry_out`, `to_post`, `success`, `fail`, `collected_twice`, `claim`, `clear`, `won`, `lost_in_play`, `lost_out`, `in_play_safe`, `in_play_danger`, `out_of_play`, `no_touch`, `touched_in`
- **`gk_technique`**: `standing`, `diving`
- **`restart_type`**: `kick_off`, `throw_in`, `goal_kick`, `corner`, `free_kick_direct`, `free_kick_indirect`, `penalty`, `drop_ball`
- **`out_line`**: `touchline`, `goal_line`
- **`pressed_role`**: `carrier`, `receiver`, `releaser`
- **`pressure_outcome`**: `forced_turnover`, `forced_hurried_action`, `none`
- **`press_sequence_outcome`**: `success`, `broken`, `out`, `whistle`
- **`run_type`**: `run_in_behind`, `overlap`, `underlap`, `checking_run`, `decoy`, `support`
- **`run_outcome`**: `targeted_received`, `targeted_missed`, `not_targeted`
- **`accel_type`**: `acceleration`, `deceleration`
- **`step_up_outcome`**: `trap_success`, `trap_beaten`, `reset`
- **`phase_of_play`**: `build_up`, `progression`, `final_third`, `attacking_transition`, `defensive_transition`, `counter_attack`, `set_piece`, `out_of_possession`
- **`defensive_block`**: `high_press`, `mid_block`, `low_block`
- **`play_pattern`**: `regular_play`, `from_corner`, `from_free_kick`, `from_throw_in`, `from_goal_kick`, `from_keeper`, `from_kick_off`, `from_counter`, `other`
- **`period`**: `1`, `2`, `3`, `4`, `5`

---
## 8. Threshold index (single config; flag review items)

| Name | Value | Status |
|---|---|---|
| pitch | 105 x 68 m | [SPADL/FIFA] |
| pass ground / low / high | ≤0.3 / ≤1.5 / >1.5 m peak | 0.3 [INVENTED], 1.5 [CONV from SB shoulder] |
| pass short / medium / long | <15 / 15–30 / >30 m | [FBref yd->m] |
| switch | ≥ 34 m lateral | [SB scaled] |
| progressive pass/carry | 30 / 15 / 10 m closer by zone, or into box | [Wyscout, FBref] |
| carry min | 3.0 m; ball ≤ 2.0 m | [SPADL 3 m]; 2.0 m [INVENTED] |
| take-on | defender ≤ 2.5 m; beaten at −1.0 m; retain 1.0 s | [INVENTED] |
| key pass window | ≤ 5 s | [INVENTED] |
| first_time | ≤ 0.4 s | [INVENTED] |
| cross / cutback zones | §3.1.3 | [CONV approximating SB App. 5/6] |
| through ball | recipient ≥ 1 m beyond last defender, length ≥ 10 m | [CONV] |
| counter-attack | 75 % direct, ≥ 16.5 m, from outside final third, shot ≤ 15 s | [SB]; 15 s [INVENTED] |
| counterpress window | 5 s | [SB] |
| pressure | ≤ 5.0 m, closing, ≥ 0.4 s | 5 m [brief]; rest [INVENTED] |
| HIR / sprint | 5.5–7.0 / >7.0 m/s, ≥ 1.0 s | speeds [FIFA]; 1.0 s [INVENTED] |
| acceleration | ±3.0 m/s², ≥ 0.5 s | [Catapult]; 0.5 s [INVENTED] |
| off-ball run candidate | ≥ 4.0 m/s, ≥ 1 s, ≥ 5 m | [INVENTED] |
| run_in_behind / overlap / underlap / checking / decoy | §5.2 | [INVENTED] |
| line step-up | ≥ 4 m in ≤ 2 s, ≥ 3 defenders | [INVENTED] |
| transition window | 8 s | [INVENTED] |
| set-piece phase window | 10 s | [INVENTED] |
| possession break | 2.0 s regain rule, 1.0 s control | [INVENTED] |
| PPDA zone | attacking 60 % (x ≥ -10.5 m) | [Opta] |
| shot techniques speed/angles | §3.5.1 | [INVENTED] |
| corner inswing/outswing | path bulges toward (in) / away from (out) goal line by ≥ 1.0 m vs chord midpoint; else `straight` | [INVENTED] |
| offside tolerance | 0.0 m sim; 0.3 m Spotter uncertainty | [INVENTED] |
