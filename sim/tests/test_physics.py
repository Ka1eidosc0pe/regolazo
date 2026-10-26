"""Milestone 1 gate: ball physics, kick model and player kinematics (PLAN.md Step 3 physics unit tests)."""

import numpy as np
import pytest

from sim import kick as K
from sim import kinematics, physics
from sim.constants import BALL_RADIUS, DT, GRAVITY, HALF_GOAL, HALF_LENGTH, RESTITUTION


def _launch(
    speed: float, elev_deg: float, az: float = 0.0, z0: float = BALL_RADIUS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.array([[0.0, 0.0, z0]])
    vel = K.launch_velocity(np.array([speed]), np.radians(np.array([elev_deg])), np.array([az]))
    return pos, vel, np.zeros((1, 3))


def _first_landing(speed: float, elev_deg: float, p: physics.BallParams) -> tuple[float, float, np.ndarray]:
    """Exact landing point: step until the bounce, then close the last parabola analytically."""
    pos, vel, spin = _launch(speed, elev_deg)
    t = 0.0
    trail = [pos[0].copy()]
    for _ in range(2000):
        x0, v0 = pos.copy(), vel.copy()
        flags = physics.step_ball(pos, vel, spin, DT, p, frame=False)
        if flags.bounced[0]:
            z, vz = x0[0, 2] - BALL_RADIUS, v0[0, 2]
            tau = (vz + np.sqrt(vz**2 + 2 * GRAVITY * z)) / GRAVITY
            return x0[0, 0] + v0[0, 0] * tau, t + tau, np.array(trail)
        t += DT
        trail.append(pos[0].copy())
    raise AssertionError("ball never landed")


def test_projectile_matches_analytic_without_drag() -> None:
    v, th = 20.0, 40.0
    rng, t_land, trail = _first_landing(v, th, physics.NO_DRAG)
    r = np.radians(th)
    assert rng == pytest.approx(v**2 * np.sin(2 * r) / GRAVITY, abs=1e-6)
    assert t_land == pytest.approx(2 * v * np.sin(r) / GRAVITY, abs=1e-6)
    # every sampled tick lies on the analytic parabola (RK2 is exact for constant acceleration)
    ts = np.arange(len(trail)) * DT
    np.testing.assert_allclose(trail[:, 0], v * np.cos(r) * ts, atol=1e-9)
    np.testing.assert_allclose(trail[:, 2], BALL_RADIUS + v * np.sin(r) * ts - 0.5 * GRAVITY * ts**2, atol=1e-9)


def test_drag_shortens_range_realistically() -> None:
    no_drag, _, _ = _first_landing(25.0, 30.0, physics.NO_DRAG)
    with_drag, _, _ = _first_landing(25.0, 30.0, physics.DEFAULT)
    assert with_drag < no_drag
    # a 90 km/h lofted ball loses roughly a fifth to a third of its vacuum range
    assert 0.65 < with_drag / no_drag < 0.85


def test_bounce_loses_energy() -> None:
    pos = np.array([[0.0, 0.0, 2.0]])
    vel = np.zeros((1, 3))
    spin = np.zeros((1, 3))
    zs, bounces = [], 0
    for _ in range(200):
        f = physics.step_ball(pos, vel, spin, frame=False)
        bounces += int(f.bounced[0])
        zs.append(pos[0, 2])
        if bounces == 1 and vel[0, 2] < 0 and zs[-1] < max(zs):
            break
    apex = max(zs[np.argmin(zs[:40]) :]) if bounces else 0.0
    drop = 2.0 - BALL_RADIUS
    rebound = apex - BALL_RADIUS
    assert 0 < rebound < drop
    assert rebound / drop == pytest.approx(RESTITUTION**2, rel=0.1)


def test_rolling_ball_stops_where_closed_form_says() -> None:
    pos = np.array([[0.0, 0.0, BALL_RADIUS]])
    vel = np.array([[12.0, 0.0, 0.0]])
    spin = np.zeros((1, 3))
    for _ in range(1000):
        physics.step_ball(pos, vel, spin, frame=False)
        if np.all(vel == 0):
            break
    assert np.all(vel == 0)
    assert pos[0, 2] == BALL_RADIUS
    assert pos[0, 0] == pytest.approx(float(K.ground_range(np.array([12.0]))[0]), rel=0.02)


def test_bouncing_ball_eventually_rolls_and_stops() -> None:
    pos, vel, spin = _launch(18.0, 35.0)
    for _ in range(1500):
        physics.step_ball(pos, vel, spin, frame=False)
    assert np.all(vel == 0) and pos[0, 2] == BALL_RADIUS


def test_magnus_curl_direction() -> None:
    pos, vel, _ = _launch(22.0, 8.0)
    left = np.array([[0.0, 0.0, 40.0]])  # +z spin curls to the left (+y) for a ball travelling +x
    xs, _ = physics.predict(pos, vel, left, 40)
    assert xs[-1, 0, 1] > 1.5
    xs, _ = physics.predict(pos, vel, -left, 40)
    assert xs[-1, 0, 1] < -1.5


def test_backspin_lifts_and_topspin_dips() -> None:
    pos, vel, _ = _launch(22.0, 20.0)
    az = np.zeros(1)
    back = K.launch_spin(az, np.zeros(1), np.array([30.0]))
    top = -back
    peak = {
        name: physics.predict(pos, vel, s, 60)[0][:, 0, 2].max()
        for name, s in (("none", np.zeros((1, 3))), ("back", back), ("top", top))
    }
    assert peak["back"] > peak["none"] > peak["top"]


def test_determinism_same_seed_same_trajectory() -> None:
    def run(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        origin = np.tile([0.0, 0.0, BALL_RADIUS], (64, 1))
        target = np.column_stack([rng.uniform(10, 40, 64), rng.uniform(-20, 20, 64), np.zeros(64)])
        kicks = rng.integers(0, 6, 64)
        plan = K.plan_kicks(origin, target, kicks, np.full(64, 0.5), rng.uniform(-1, 1, 64))
        noisy = K.perturb(plan.launch, kicks, np.ones(64), rng)
        xs, _ = physics.predict(origin, noisy.velocity(), noisy.spin, 80)
        return xs

    np.testing.assert_array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


def test_ground_pass_arrives_with_requested_speed() -> None:
    dist = np.array([8.0, 15.0, 25.0, 35.0])
    v0 = K.ground_speed_for(dist, np.full(4, 6.0))
    pos = np.column_stack([np.zeros(4), np.zeros(4), np.full(4, BALL_RADIUS)])
    vel = np.column_stack([v0, np.zeros(4), np.zeros(4)])
    xs, vs = physics.predict(pos, vel, np.zeros((4, 3)), 200)
    for j in range(4):
        k = np.argmax(xs[:, j, 0] >= dist[j])
        assert np.linalg.norm(vs[k, j]) == pytest.approx(6.0, abs=0.35)
    t = K.ground_time(v0, dist)
    assert np.all(np.isfinite(t)) and np.all(np.diff(t) > 0)


@pytest.mark.parametrize("kick", [K.K_LOFTED, K.K_CHIPPED, K.K_CROSS, K.K_PUNT, K.K_GK_THROW])
def test_lofted_kicks_land_near_target_after_refine(kick: int) -> None:
    n = 5
    origin = np.tile([0.0, 0.0, BALL_RADIUS], (n, 1))
    d = np.linspace(14, {K.K_CHIPPED: 22, K.K_GK_THROW: 32}.get(kick, 42), n)
    h = 1.7 if kick == K.K_CROSS else 0.2
    target = np.column_stack([d * np.cos(0.3), d * np.sin(0.3), np.full(n, h)])
    kicks = np.full(n, kick)
    plan = K.refine(K.plan_kicks(origin, target, kicks, np.full(n, 0.5), np.zeros(n)))
    xs, _ = physics.predict(origin, plan.launch.velocity(), plan.launch.spin, 120)
    for j in range(n):
        along = xs[:, j, 0] * np.cos(0.3) + xs[:, j, 1] * np.sin(0.3)
        k = int(np.argmax(along >= d[j]))
        assert abs(xs[k, j, 2] - h) < 0.9
        lateral = -xs[k, j, 0] * np.sin(0.3) + xs[k, j, 1] * np.cos(0.3)
        assert abs(lateral) < 0.6


def test_shot_elevation_hits_goal_mouth_height() -> None:
    n = 4
    origin = np.column_stack([np.full(n, 34.0), np.linspace(-8, 8, n), np.full(n, BALL_RADIUS)])
    target = np.column_stack(
        [np.full(n, HALF_LENGTH), np.array([-2.5, 2.5, -1.0, 3.0]), np.array([0.4, 2.0, 1.2, 0.8])]
    )
    kicks = np.array([K.K_SHOT_DRIVEN, K.K_SHOT_PLACED, K.K_SHOT_DRIVEN, K.K_SHOT_FINESSE])
    plan = K.refine(K.plan_kicks(origin, target, kicks, np.full(n, 0.6), np.array([0.0, 0.0, 0.0, 0.7])))
    xs, _ = physics.predict(origin, plan.launch.velocity(), plan.launch.spin, 60)
    for j in range(n):
        k = int(np.argmax(xs[:, j, 0] >= HALF_LENGTH))
        assert abs(xs[k, j, 2] - target[j, 2]) < 0.5
        assert abs(xs[k, j, 1] - target[j, 1]) < 0.5


def test_curled_shot_bends_more_than_driven() -> None:
    origin = np.array([[30.0, -10.0, BALL_RADIUS]] * 2)
    target = np.array([[HALF_LENGTH, 2.5, 1.0]] * 2)
    kicks = np.array([K.K_SHOT_DRIVEN, K.K_SHOT_FINESSE])
    plan = K.refine(K.plan_kicks(origin, target, kicks, np.full(2, 0.5), np.array([0.0, -0.8])))
    xs, _ = physics.predict(origin, plan.launch.velocity(), plan.launch.spin, 40)
    chord = target[0, :2] - origin[0, :2]
    u = chord / np.linalg.norm(chord)
    dev = [np.max(np.abs(-(xs[:, j, 0] - origin[j, 0]) * u[1] + (xs[:, j, 1] - origin[j, 1]) * u[0])) for j in (0, 1)]
    assert dev[1] > 1.0 > dev[0]


def test_noise_multiplier_monotone() -> None:
    q = np.array([0.2, 0.5, 0.9])
    m = K.noise_multiplier(q, np.zeros(3), np.full(3, 0.5), np.zeros(3))
    assert m[0] > m[1] > m[2] and m[1] == pytest.approx(1.0, abs=0.01)
    pressured = K.noise_multiplier(np.full(2, 0.5), np.array([0.0, 1.0]), np.full(2, 0.5), np.zeros(2))
    assert pressured[1] > pressured[0]
    calm = K.noise_multiplier(np.full(2, 0.5), np.ones(2), np.array([0.1, 0.95]), np.zeros(2))
    assert calm[1] < calm[0]
    weak = K.noise_multiplier(np.full(2, 0.5), np.zeros(2), np.full(2, 0.5), np.array([0.0, 1.0]))
    assert weak[1] > weak[0]


def test_post_and_crossbar_rebound() -> None:
    pos = np.array([[HALF_LENGTH - 3.0, HALF_GOAL, 1.0], [HALF_LENGTH - 3.0, 0.0, 2.44]])
    vel = np.array([[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    spin = np.zeros((2, 3))
    after = np.zeros((2, 3))
    hit = np.zeros(2, dtype=bool)
    for _ in range(10):
        h = physics.step_ball(pos, vel, spin).woodwork
        after[h] = vel[h]
        hit |= h
    assert hit.all()
    assert after[0, 0] < 0  # straight onto the post: back into play
    assert after[1, 2] < -3.0  # just under the bar: off the underside and down


def test_ball_in_net_is_caught() -> None:
    pos = np.array([[HALF_LENGTH - 2.0, 0.0, 1.0]])
    vel = np.array([[25.0, 0.0, 0.0]])
    spin = np.zeros((1, 3))
    for _ in range(50):
        physics.step_ball(pos, vel, spin)
    assert HALF_LENGTH < pos[0, 0] <= HALF_LENGTH + 2.0
    assert np.linalg.norm(vel) < 3.0


def _run_to(
    target: np.ndarray, steps: int, start_vel: tuple[float, float] = (0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((1, 2))
    vel = np.array([start_vel], dtype=float)
    ms, ma, md = np.array([8.6]), np.array([3.9]), np.array([5.5])
    path, vels = [], []
    for _ in range(steps):
        pos, vel = kinematics.steer(pos, vel, target, np.array([9.9]), ms, ma, md)
        path.append(pos[0].copy())
        vels.append(vel[0].copy())
    return np.array(path), np.array(vels)


def test_player_respects_speed_and_accel_limits() -> None:
    path, vels = _run_to(np.array([[60.0, 0.0]]), 400)
    speed = np.linalg.norm(vels, axis=1)
    assert speed.max() <= 8.6 + 1e-9
    acc = np.linalg.norm(np.diff(vels, axis=0), axis=1) / DT
    assert acc.max() <= 5.5 + 1e-6
    # sprint 0 -> 30 m takes roughly 4-5 s for an average professional
    t30 = np.argmax(path[:, 0] >= 30.0) * DT
    assert 3.8 < t30 < 5.6
    # arrives and stops
    assert np.linalg.norm(path[-1] - [60.0, 0.0]) < 0.3 and speed[-1] < 0.3


def test_turning_at_speed_has_a_radius() -> None:
    # running +x at 7 m/s, target behind-left: the player cannot reverse instantly
    path, vels = _run_to(np.array([[-10.0, 10.0]]), 30, start_vel=(7.0, 0.0))
    assert path[:, 0].max() > 2.0  # carried forward by momentum while turning
    speed = np.linalg.norm(vels, axis=1)
    assert speed.min() < 6.0


def test_jump_and_turn() -> None:
    z, vz = np.zeros(1), kinematics.jump_speed(np.array([80.0]))
    peak = 0.0
    for _ in range(40):
        z, vz = kinematics.step_jump(z, vz)
        peak = max(peak, float(z[0]))
    assert 0.4 < peak < 0.7 and z[0] == 0.0
    f = np.zeros(1)
    for _ in range(5):
        f = kinematics.turn_toward(f, np.array([np.pi / 2]), np.zeros(1))
    assert 0 < f[0] <= np.pi / 2
