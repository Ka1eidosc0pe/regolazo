import numpy as np

from vision.events import controls, detect, kinematics, owners, sprints, track_teams
from vision.tracking import Tracks

FPS = 25.0


def timeline(seconds: float) -> np.ndarray:
    return np.arange(round(seconds * FPS)) / FPS + 100.0


def still(n: int, x: float, y: float) -> np.ndarray:
    return np.tile([x, y], (n, 1)).astype(np.float64)


def move(t: np.ndarray, t0: float, t1: float, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """Position at rest at a until t0, linear to b by t1, at rest after."""
    f = np.clip((t - t[0] - t0) / (t1 - t0), 0, 1)[:, None]
    out: np.ndarray = np.asarray(a) + f * (np.asarray(b) - np.asarray(a))
    return out


def test_kinematics_speed_of_constant_motion() -> None:
    t = timeline(4)
    xy = np.column_stack([5.0 * (t - t[0]), np.zeros_like(t)])
    k = kinematics(xy, t)
    assert np.allclose(k.speed[20:-20], 5.0, atol=1e-6)


def test_kinematics_fills_short_gaps_only() -> None:
    t = timeline(4)
    xy = np.column_stack([t - t[0], np.zeros_like(t)])
    xy[30:35] = np.nan  # 0.2 s gap: filled
    xy[60:90] = np.nan  # 1.2 s gap: stays missing
    k = kinematics(xy, t, max_gap_s=0.6, win_s=0.04)
    assert not np.isnan(k.pos[32, 0])
    assert np.isnan(k.pos[75, 0])


def test_owner_is_nearest_player_within_radius() -> None:
    ball = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    players = {1: still(3, 0.5, 0.0), 2: still(3, 1.0, 0.0), 3: still(3, 21.2, 0.0)}
    teams = {1: "home", 2: "away", 3: "away"}
    assert owners(ball, players, teams).tolist() == [1, -1, 3]  # type: ignore[arg-type]


def test_controls_merge_runs_and_drop_short_contacts() -> None:
    owner = np.array([1, 1, 1, -1, 1, 1, 2, -1, 3, 3, 3])
    runs = controls(owner, {1: "home", 2: "away", 3: "home"})
    assert [(r.track, r.i0, r.i1) for r in runs] == [(1, 0, 5), (3, 8, 10)]


def test_track_teams_assigns_goalkeeper_by_goal() -> None:
    t = timeline(1)
    tracks = Tracks(t, still(len(t), 0, 0), {1: still(len(t), -50, 0), 2: still(len(t), 50, 0), 3: still(len(t), 0, 0)})
    tracks.role = {1: "goalkeeper", 2: "goalkeeper", 3: "referee"}
    # period 1, home attacks +x: home keeper stands in the -x goal
    assert track_teams(tracks, 1, True) == {1: "home", 2: "away"}
    assert track_teams(tracks, 2, True) == {1: "away", 2: "home"}


def two_pass_scene() -> Tracks:
    """Home 1 passes to home 2 (20 m, completed), home 2 passes to away 3 (intercepted)."""
    t = timeline(6)
    n = len(t)
    p1, p2, p3 = still(n, 0, 0), still(n, 20, 0), still(n, 20, 15)
    ball = np.vstack([move(t, 1.0, 2.0, (0, 0), (20, 0))[: round(3 * FPS)], move(t, 4.0, 5.0, (20, 0), (20, 15))[75:]])
    return Tracks(t, ball, {1: p1, 2: p2, 3: p3}, {1: "home", 2: "home", 3: "away"})


def test_passes_completed_and_intercepted() -> None:
    events, runs = detect(two_pass_scene(), 1, True)
    passes = [e for e in events if e.type == "pass"]
    assert [r.track for r in runs] == [1, 2, 3]
    assert len(passes) == 2
    first, second = passes
    assert first.qualifiers is not None and first.qualifiers.kind == "pass"
    assert first.qualifiers.outcome == "complete" and first.team == "home"
    assert abs(first.qualifiers.length_m - 20.0) < 2.5
    assert abs(first.qualifiers.ball_speed_mps - 20.0) < 3.0
    assert first.track_id == 1 and first.secondary_track_id == 2
    assert abs(first.t - 101.0) < 0.2
    assert second.qualifiers is not None and second.qualifiers.kind == "pass"
    assert second.qualifiers.outcome == "intercepted"
    # the intercepted pass is still the passer's possession; the interceptor starts a new one
    assert second.team == "home" and second.possession_id == first.possession_id
    assert runs[2].team == "away"
    assert all(e.source == "detected" for e in events)


def test_fast_ball_toward_goal_is_a_shot() -> None:
    t = timeline(4)
    n = len(t)
    shooter = still(n, 30, 2)
    ball = move(t, 1.0, 1.9, (30, 2), (52, 0))  # ~24 m/s toward the +x goal
    ball[round(2 * FPS) :] = np.nan  # ball leaves the picture
    tracks = Tracks(t, ball, {1: shooter}, {1: "home"})
    events, _ = detect(tracks, 1, True)
    shots = [e for e in events if e.type == "shot"]
    assert len(shots) == 1 and shots[0].team == "home"
    # the same ball struck by the away team (attacking -x in period 1) is not a shot
    tracks.role = {1: "away"}
    assert not [e for e in detect(tracks, 1, True)[0] if e.type == "shot"]


def test_sprint_threshold_and_duration() -> None:
    t = timeline(6)
    speed = np.full(len(t), 5.0)
    speed[25:75] = 8.0  # 2 s above 7 m/s
    speed[100:110] = 8.0  # 0.4 s: too short
    found = sprints(speed, t)
    assert len(found) == 1
    i0, i1, peak, dist = found[0]
    assert (i0, i1) == (25, 74) and peak == 8.0
    assert abs(dist - 8.0 * 49 / FPS) < 1e-6
