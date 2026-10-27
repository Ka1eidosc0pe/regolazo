"""Kick model: intent (target, technique, power, curve) + skill -> ball launch with attribute-scaled noise.

Launch planning inverts precomputed flight tables (drag included, built once per process) so that hundreds of
kicks can be planned per tick without iterative simulation. ``refine`` then corrects the plan against the exact
physics (spin included) with one forward simulation, which is what the preview arc and real kicks use.
"""

import functools
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from sim.constants import BALL_RADIUS, DRAG_K, GRAVITY, ROLL_DECEL
from sim.physics import DEFAULT, BallParams, predict

F = npt.NDArray[np.float64]
IA = npt.NDArray[np.int64]

# --- Kick techniques (integer codes so they vectorize) ------------------------------------------------------
K_GROUND: Final = 0
K_DRIVEN: Final = 1
K_LOFTED: Final = 2
K_CHIPPED: Final = 3
K_CROSS: Final = 4
K_CURLED: Final = 5
K_OUTSIDE: Final = 6
K_SHOT_DRIVEN: Final = 7
K_SHOT_PLACED: Final = 8
K_SHOT_FINESSE: Final = 9
K_SHOT_CHIP: Final = 10
K_VOLLEY: Final = 11
K_HALF_VOLLEY: Final = 12
K_HEADER: Final = 13
K_TOE_POKE: Final = 14
K_CLEARANCE: Final = 15
K_THROW_IN: Final = 16
K_GK_THROW: Final = 17
K_PUNT: Final = 18
K_DROP_KICK: Final = 19
K_HEADER_PASS: Final = 20
K_CROSS_LOW: Final = 21
N_KICKS: Final = 22

SHOT_KICKS: Final = frozenset(
    {K_SHOT_DRIVEN, K_SHOT_PLACED, K_SHOT_FINESSE, K_SHOT_CHIP, K_VOLLEY, K_HALF_VOLLEY, K_HEADER, K_TOE_POKE}
)

# Per technique: base noise (azimuth rad, elevation rad, relative speed), launch mode and fixed elevation.
MODE_GROUND, MODE_FIXED_ELEV, MODE_FIXED_SPEED = 0, 1, 2


@dataclass(frozen=True)
class KickProfile:
    mode: int
    elev_deg: float
    """Fixed elevation (MODE_FIXED_ELEV) or a fallback guess."""
    speed_lo: float
    speed_hi: float
    """MODE_FIXED_SPEED: speed at power 0 and 1. Other modes: hard cap (speed_hi)."""
    sig_az: float
    sig_el: float
    sig_v: float
    spin: float = 0.0
    """Side-spin magnitude at |curve| = 1 (rad/s); negative = backspin magnitude for chips."""
    backspin: float = 0.0


PROFILES: Final[dict[int, KickProfile]] = {
    K_GROUND: KickProfile(MODE_GROUND, 0.0, 3.0, 24.0, 0.030, 0.0, 0.055),
    K_DRIVEN: KickProfile(MODE_FIXED_SPEED, 3.0, 17.0, 28.0, 0.030, 0.018, 0.045),
    K_LOFTED: KickProfile(MODE_FIXED_ELEV, 27.0, 5.0, 33.0, 0.030, 0.025, 0.045, backspin=6.0),
    K_CHIPPED: KickProfile(MODE_FIXED_ELEV, 47.0, 5.0, 22.0, 0.040, 0.040, 0.060, backspin=22.0),
    K_CROSS: KickProfile(MODE_FIXED_ELEV, 17.0, 8.0, 30.0, 0.034, 0.030, 0.050, spin=30.0),
    K_CURLED: KickProfile(MODE_GROUND, 5.0, 3.0, 26.0, 0.034, 0.012, 0.055, spin=42.0),
    K_OUTSIDE: KickProfile(MODE_GROUND, 4.0, 3.0, 26.0, 0.045, 0.015, 0.065, spin=34.0),
    K_SHOT_DRIVEN: KickProfile(MODE_FIXED_SPEED, 4.0, 24.0, 32.0, 0.040, 0.034, 0.045),
    K_SHOT_PLACED: KickProfile(MODE_FIXED_SPEED, 3.0, 15.0, 22.0, 0.030, 0.028, 0.045),
    K_SHOT_FINESSE: KickProfile(MODE_FIXED_SPEED, 5.0, 18.0, 24.0, 0.034, 0.030, 0.045, spin=55.0),
    K_SHOT_CHIP: KickProfile(MODE_FIXED_ELEV, 36.0, 6.0, 20.0, 0.040, 0.045, 0.060, backspin=14.0),
    K_VOLLEY: KickProfile(MODE_FIXED_SPEED, 4.0, 20.0, 30.0, 0.070, 0.065, 0.070),
    K_HALF_VOLLEY: KickProfile(MODE_FIXED_SPEED, 4.0, 19.0, 28.0, 0.060, 0.055, 0.065),
    K_HEADER: KickProfile(MODE_FIXED_SPEED, -4.0, 9.0, 16.0, 0.060, 0.065, 0.080),
    K_TOE_POKE: KickProfile(MODE_FIXED_SPEED, 2.0, 12.0, 16.0, 0.075, 0.060, 0.090),
    K_CLEARANCE: KickProfile(MODE_FIXED_SPEED, 32.0, 20.0, 29.0, 0.140, 0.100, 0.090),
    K_THROW_IN: KickProfile(MODE_FIXED_ELEV, 22.0, 4.0, 17.0, 0.040, 0.040, 0.060),
    K_GK_THROW: KickProfile(MODE_FIXED_ELEV, 10.0, 6.0, 24.0, 0.030, 0.030, 0.050),
    K_PUNT: KickProfile(MODE_FIXED_ELEV, 40.0, 10.0, 34.0, 0.050, 0.040, 0.060),
    K_DROP_KICK: KickProfile(MODE_FIXED_ELEV, 24.0, 10.0, 34.0, 0.045, 0.035, 0.055),
    K_HEADER_PASS: KickProfile(MODE_FIXED_ELEV, 12.0, 4.0, 15.0, 0.070, 0.060, 0.080),
    K_CROSS_LOW: KickProfile(MODE_FIXED_ELEV, 6.0, 8.0, 28.0, 0.034, 0.025, 0.050, spin=14.0),
}

_PROFILE_ARR: Final = {
    name: np.array([getattr(PROFILES[k], name) for k in range(N_KICKS)], dtype=np.float64)
    for name in ("mode", "elev_deg", "speed_lo", "speed_hi", "sig_az", "sig_el", "sig_v", "spin", "backspin")
}


def profile_array(name: str, kick: IA) -> F:
    return np.asarray(_PROFILE_ARR[name][kick])


# --- Flight tables ------------------------------------------------------------------------------------------
TABLE_SPEEDS: Final = np.arange(3.0, 42.01, 0.5)
TABLE_ELEVS: Final = np.radians(np.arange(-10.0, 71.0, 1.0))
TABLE_DISTS: Final = np.arange(0.0, 100.01, 0.5)


@dataclass(frozen=True)
class FlightTables:
    z: F
    """Height relative to the launch point when the ball has travelled d horizontally: [speed, elev, dist]."""
    t: F
    """Flight time to d."""
    speed: F
    """Ball speed at d."""


@functools.cache
def flight_tables(drag_k: float = DRAG_K) -> FlightTables:
    """Drag-only free flight (no ground, no spin) for every (speed, elevation) on the grid."""
    sv, se = np.meshgrid(TABLE_SPEEDS, TABLE_ELEVS, indexing="ij")
    v = np.stack([sv.ravel() * np.cos(se.ravel()), np.zeros(sv.size), sv.ravel() * np.sin(se.ravel())], axis=1)
    x = np.zeros_like(v)
    h = 0.01
    n_steps = 700
    xs = np.empty((n_steps + 1, len(v)))
    zs = np.empty((n_steps + 1, len(v)))
    ss = np.empty((n_steps + 1, len(v)))
    xs[0], zs[0], ss[0] = 0.0, 0.0, np.linalg.norm(v, axis=1)
    for k in range(1, n_steps + 1):
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        a = -drag_k * sp * v
        a[:, 2] -= GRAVITY
        vm = v + 0.5 * h * a
        x = x + h * vm
        spm = np.linalg.norm(vm, axis=1, keepdims=True)
        am = -drag_k * spm * vm
        am[:, 2] -= GRAVITY
        v = v + h * am
        xs[k], zs[k], ss[k] = x[:, 0], x[:, 2], np.linalg.norm(v, axis=1)
    times = np.arange(n_steps + 1) * h
    nd = len(TABLE_DISTS)
    z_tab = np.full((len(v), nd), np.nan)
    t_tab = np.full((len(v), nd), np.nan)
    s_tab = np.full((len(v), nd), np.nan)
    for j in range(len(v)):
        xj = xs[:, j]
        valid = zs[:, j] > -6.0
        last = int(np.nonzero(valid)[0][-1])
        reach = xj[last] >= TABLE_DISTS
        if not reach.any():
            continue
        d = TABLE_DISTS[reach]
        z_tab[j, reach] = np.interp(d, xj[: last + 1], zs[: last + 1, j])
        t_tab[j, reach] = np.interp(d, xj[: last + 1], times[: last + 1])
        s_tab[j, reach] = np.interp(d, xj[: last + 1], ss[: last + 1, j])
    shape = (len(TABLE_SPEEDS), len(TABLE_ELEVS), nd)
    return FlightTables(z=z_tab.reshape(shape), t=t_tab.reshape(shape), speed=s_tab.reshape(shape))


def _frac_index(grid: F, values: F) -> tuple[IA, F]:
    pos = np.interp(values, grid, np.arange(len(grid), dtype=np.float64))
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, len(grid) - 2)
    return i0, pos - i0


def _table_at_speed_dist(tab: F, speed: F, dist: F) -> F:
    """Bilinear interpolation in (speed, dist): returns (N, n_elev)."""
    i, fi = _frac_index(TABLE_SPEEDS, speed)
    j, fj = _frac_index(TABLE_DISTS, dist)
    a = tab[i, :, j] * (1 - fj)[:, None] + tab[i, :, j + 1] * fj[:, None]
    b = tab[i + 1, :, j] * (1 - fj)[:, None] + tab[i + 1, :, j + 1] * fj[:, None]
    out: F = a * (1 - fi)[:, None] + b * fi[:, None]
    return out


def _table_at_elev_dist(tab: F, elev: F, dist: F) -> F:
    """Bilinear interpolation in (elev, dist): returns (N, n_speed)."""
    i, fi = _frac_index(TABLE_ELEVS, elev)
    j, fj = _frac_index(TABLE_DISTS, dist)
    a = tab[:, i, j] * (1 - fj)[None, :] + tab[:, i, j + 1] * fj[None, :]
    b = tab[:, i + 1, j] * (1 - fj)[None, :] + tab[:, i + 1, j + 1] * fj[None, :]
    out: F = (a * (1 - fi)[None, :] + b * fi[None, :]).T
    return out


def solve_elevation(speed: F, dist: F, dz: F, high: bool = False) -> F:
    """Elevation (rad) so a ball launched at ``speed`` is ``dz`` above the launch point after ``dist`` metres.

    The low (direct) solution by default, the high (lob) one with ``high=True``. NaN where unreachable.
    """
    tabs = flight_tables()
    z = _table_at_speed_dist(tabs.z, speed, dist)  # (N, E)
    above = np.nan_to_num(z - dz[:, None], nan=-1e9) >= 0
    n_e = len(TABLE_ELEVS)
    out = np.full(len(speed), np.nan)
    any_above = above.any(axis=1)
    if not any_above.any():
        return out
    if high:
        k = n_e - 1 - np.argmax(above[:, ::-1], axis=1)  # last index above
        k2 = np.minimum(k + 1, n_e - 1)
    else:
        k = np.argmax(above, axis=1)  # first index above
        k2 = np.maximum(k - 1, 0)
    rows = np.arange(len(speed))
    z1, z2 = z[rows, k], z[rows, k2]
    e1, e2 = TABLE_ELEVS[k], TABLE_ELEVS[k2]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.isfinite(z2) & (z1 != z2), (z1 - dz) / (z1 - z2), 0.0)
    frac = np.clip(np.nan_to_num(frac), 0.0, 1.0)
    out[any_above] = (e1 + frac * (e2 - e1))[any_above]
    return out


def solve_speed(elev: F, dist: F, dz: F) -> F:
    """Launch speed so a ball at elevation ``elev`` is ``dz`` above launch height after ``dist`` metres."""
    tabs = flight_tables()
    z = _table_at_elev_dist(tabs.z, elev, dist)  # (N, S), increasing in speed
    above = np.nan_to_num(z - dz[:, None], nan=-1e9) >= 0
    k = np.argmax(above, axis=1)
    rows = np.arange(len(elev))
    ok = above.any(axis=1)
    k0 = np.maximum(k - 1, 0)
    z1, z0 = z[rows, k], z[rows, k0]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.isfinite(z0) & (z1 != z0), (dz - z0) / (z1 - z0), 1.0)
    frac = np.clip(np.nan_to_num(frac, nan=1.0), 0.0, 1.0)
    v = TABLE_SPEEDS[k0] + frac * (TABLE_SPEEDS[k] - TABLE_SPEEDS[k0])
    return np.asarray(np.where(ok, v, TABLE_SPEEDS[-1]), dtype=np.float64)


def flight_time(speed: F, elev: F, dist: F) -> F:
    """Time for an airborne ball to cover ``dist`` horizontally (table lookup)."""
    tabs = flight_tables()
    t_e = _table_at_speed_dist(tabs.t, speed, dist)
    i, fi = _frac_index(TABLE_ELEVS, elev)
    rows = np.arange(len(speed))
    out: F = t_e[rows, i] * (1 - fi) + t_e[rows, i + 1] * fi
    return out


# --- Ground balls (closed form for rolling resistance + quadratic drag) ------------------------------------


def ground_speed_for(dist: F, v_arrive: F, roll_decel: float = ROLL_DECEL, k: float = DRAG_K) -> F:
    """Initial speed of a rolling ball that still has ``v_arrive`` after ``dist`` metres."""
    v0_sq = ((roll_decel + k * v_arrive**2) * np.exp(2 * k * dist) - roll_decel) / k
    return np.asarray(np.sqrt(np.maximum(v0_sq, 0.0)))


def ground_range(v0: F, v_end: F | float = 0.0, roll_decel: float = ROLL_DECEL, k: float = DRAG_K) -> F:
    """Distance a rolling ball covers while slowing from ``v0`` to ``v_end``."""
    v_end_arr = np.broadcast_to(np.asarray(v_end, dtype=np.float64), np.shape(v0))
    return np.asarray(np.log((roll_decel + k * v0**2) / (roll_decel + k * v_end_arr**2)) / (2 * k))


def ground_time(v0: F, dist: F, roll_decel: float = ROLL_DECEL, k: float = DRAG_K) -> F:
    """Time for a rolling ball launched at ``v0`` to cover ``dist`` (inf if it stops short)."""
    reach = ground_range(v0)
    v_sq = ((roll_decel + k * v0**2) * np.exp(-2 * k * np.minimum(dist, reach)) - roll_decel) / k
    v1 = np.sqrt(np.maximum(v_sq, 0.0))
    # dv/dt = -(a + k v^2)  =>  t = (atan(v0 sqrt(k/a)) - atan(v1 sqrt(k/a))) / sqrt(a k)
    c = np.sqrt(k / roll_decel)
    t = (np.arctan(v0 * c) - np.arctan(v1 * c)) / np.sqrt(roll_decel * k)
    return np.where(dist <= reach + 1e-9, t, np.inf)


# --- Launch vectors, spin and noise ---------------------------------------------------------------------------


def launch_velocity(speed: F, elev: F, azimuth: F) -> F:
    c = np.cos(elev)
    return np.stack([speed * c * np.cos(azimuth), speed * c * np.sin(azimuth), speed * np.sin(elev)], axis=1)


def launch_spin(azimuth: F, side: F, back: F) -> F:
    """Spin vector from side-spin (rad/s about +z, positive curls left) and backspin magnitude."""
    out = np.zeros((len(azimuth), 3))
    # backspin axis = v_hat x z = (sin a, -cos a, 0)
    out[:, 0] = back * np.sin(azimuth)
    out[:, 1] = -back * np.cos(azimuth)
    out[:, 2] = side
    return out


def noise_multiplier(quality: F, pressure: F, composure: F, weak_foot_level: F) -> F:
    """Scale applied to a technique's base noise.

    quality: relevant skill 0..1 (0.5 = average). pressure: 0..1. composure: 0..1.
    weak_foot_level: 0 when kicking with the strong foot, else (5 - weak_foot) / 4 in 0..1.
    """
    skill = 1.55 - 1.1 * np.clip(quality, 0.0, 1.0)
    press = 1.0 + 0.7 * np.clip(pressure, 0.0, 1.0) * (1.25 - 0.75 * np.clip(composure, 0.0, 1.0))
    weak = 1.0 + 0.55 * np.clip(weak_foot_level, 0.0, 1.0)
    return np.asarray(skill * press * weak)


@dataclass
class Launch:
    speed: F
    elev: F
    azimuth: F
    spin: F
    """(N, 3)."""

    def velocity(self) -> F:
        return launch_velocity(self.speed, self.elev, self.azimuth)

    def take(self, idx: IA | npt.NDArray[np.bool_]) -> "Launch":
        return Launch(self.speed[idx], self.elev[idx], self.azimuth[idx], self.spin[idx])


def perturb(launch: Launch, kick: IA, mult: F, rng: np.random.Generator) -> Launch:
    """Execution noise: Gaussian on azimuth, elevation and speed, scaled per technique and by ``mult``."""
    n = len(kick)
    s_az = profile_array("sig_az", kick) * mult
    s_el = profile_array("sig_el", kick) * mult
    s_v = profile_array("sig_v", kick) * mult
    z = rng.standard_normal((n, 4))
    speed = np.clip(launch.speed * (1.0 + s_v * z[:, 0]), 1.0, 45.0)
    elev = np.clip(launch.elev + s_el * z[:, 1], -0.5, 1.45)
    az = launch.azimuth + s_az * z[:, 2]
    spin = launch.spin.copy()
    spin[:, 2] *= 1.0 + 0.25 * mult * z[:, 3]
    return Launch(speed, elev, az, spin)


@dataclass
class KickPlan:
    """Planned (noise-free) kick, one row per kick."""

    origin: F
    """(N, 3) ball position at contact."""
    target: F
    """(N, 3) intended end point (receiver / landing spot / goal-mouth point)."""
    kick: IA
    launch: Launch


def plan_kicks(origin: F, target: F, kick: IA, power: F, curve: F, arrive_speed: F | None = None) -> KickPlan:
    """Noise-free launch for each kick (table inversion; refine() corrects for spin).

    ``target[:, 2]`` is the desired ball height at the target (0 = on the grass, ~1.7 for a cross to a head,
    goal-mouth height for shots). ``curve`` in -1..1 is the bend (+ = right from the kicker's view).
    """
    n = len(kick)
    d_vec = target[:, :2] - origin[:, :2]
    dist = np.maximum(np.linalg.norm(d_vec, axis=1), 0.3)
    az = np.arctan2(d_vec[:, 1], d_vec[:, 0])
    mode = profile_array("mode", kick).astype(np.int64)
    lo, hi = profile_array("speed_lo", kick), profile_array("speed_hi", kick)
    elev = np.radians(profile_array("elev_deg", kick))
    dz = target[:, 2] - origin[:, 2]
    speed = np.zeros(n)
    ground = mode == MODE_GROUND
    if ground.any():
        va = 2.5 + 4.5 * power[ground] if arrive_speed is None else arrive_speed[ground]
        g_speed = ground_speed_for(dist[ground], va)
        lofted_ground = elev[ground] > 0
        # low curled / outside balls skid first: a touch faster than a pure roll
        g_speed = np.where(lofted_ground, g_speed * 1.08, g_speed)
        speed[ground] = np.minimum(g_speed, hi[ground])
    fixed_e = mode == MODE_FIXED_ELEV
    if fixed_e.any():
        e = elev[fixed_e].copy()
        need = solve_speed(e, dist[fixed_e], dz[fixed_e])
        for _ in range(4):  # too far for this elevation within the speed cap: loft it more
            over = need > hi[fixed_e]
            if not over.any():
                break
            e[over] += np.radians(6.0)
            need[over] = solve_speed(e[over], dist[fixed_e][over], dz[fixed_e][over])
        elev[fixed_e] = e
        speed[fixed_e] = np.minimum(need, hi[fixed_e])
    fixed_s = mode == MODE_FIXED_SPEED
    if fixed_s.any():
        sp = lo[fixed_s] + (hi[fixed_s] - lo[fixed_s]) * power[fixed_s]
        e = solve_elevation(sp, dist[fixed_s], dz[fixed_s])
        fallback = np.radians(profile_array("elev_deg", kick[fixed_s]))
        speed[fixed_s] = sp
        elev[fixed_s] = np.where(np.isfinite(e), e, fallback)
    side = -np.clip(curve, -1.0, 1.0) * profile_array("spin", kick)
    back = profile_array("backspin", kick)
    spin = launch_spin(az, side, back)
    # aim off to compensate the bend: lateral offset ~ 0.5 * k * omega * d * t
    t_est = dist / np.maximum(speed * np.cos(elev) * 0.8, 1.0)
    offset = 0.5 * DEFAULT.magnus_k * side * dist * t_est
    az = az - np.arctan2(offset, dist)
    spin = launch_spin(az, side, back)
    return KickPlan(origin=origin, target=target, kick=kick, launch=Launch(speed, elev, az, spin))


def refine(plan: KickPlan, horizon_s: float = 3.5, p: BallParams = DEFAULT) -> KickPlan:
    """One correction against exact physics: rescale speed for range and rotate azimuth for lateral error."""
    n = len(plan.kick)
    if n == 0:
        return plan
    dist0 = np.linalg.norm(plan.target[:, :2] - plan.origin[:, :2], axis=1)
    h_speed = np.maximum(plan.launch.speed * np.cos(plan.launch.elev), 3.0)
    need = float(np.max(dist0 / (0.55 * h_speed))) + 0.3  # long enough to get past the target
    steps = int(min(horizon_s, need) * 25) + 2
    pos0 = plan.origin.copy()
    xs, _ = predict(pos0, plan.launch.velocity(), plan.launch.spin, steps, p=p)
    d_vec = plan.target[:, :2] - plan.origin[:, :2]
    dist = np.maximum(np.linalg.norm(d_vec, axis=1), 0.3)
    u = d_vec / dist[:, None]
    rel = xs[:, :, :2] - plan.origin[None, :, :2]
    along = rel[..., 0] * u[None, :, 0] + rel[..., 1] * u[None, :, 1]  # (T, N)
    lateral = -rel[..., 0] * u[None, :, 1] + rel[..., 1] * u[None, :, 0]
    passed = along >= dist[None, :]
    reached = passed.any(axis=0)
    k = np.argmax(passed, axis=0)
    cols = np.arange(n)
    lat_err = np.where(reached, lateral[k, cols], lateral[-1, cols])
    az = plan.launch.azimuth - np.arctan2(lat_err, dist)
    speed = plan.launch.speed.copy()
    mode = profile_array("mode", plan.kick).astype(np.int64)
    # range correction for lofted balls: compare where it comes down to the target height
    lofted = (mode == MODE_FIXED_ELEV) & reached
    if lofted.any():
        z_at = xs[k, cols, 2] - plan.target[:, 2]
        fix = lofted & (np.abs(z_at) > 0.25)
        speed[fix] *= np.clip(1.0 - 0.25 * z_at[fix] / np.maximum(dist[fix], 1.0), 0.85, 1.15)
    short = (mode == MODE_FIXED_ELEV) & ~reached
    speed[short] *= np.sqrt(np.clip(dist[short] / np.maximum(along[-1, short], 1.0), 1.0, 1.3))
    speed = np.minimum(speed, profile_array("speed_hi", plan.kick) * 1.05)
    side = plan.launch.spin[:, 2]
    back = profile_array("backspin", plan.kick)
    spin = launch_spin(az, side, back)
    return KickPlan(plan.origin, plan.target, plan.kick, Launch(speed, plan.launch.elev.copy(), az, spin))


def ball_start(origin_xy: F, kick: IA, player_z: F | None = None) -> F:
    """Contact height by technique (headers from the head, throws from the hands, volleys off the ground)."""
    n = len(kick)
    z = np.full(n, BALL_RADIUS)
    z[np.isin(kick, [K_HEADER, K_HEADER_PASS])] = 1.85
    z[np.isin(kick, [K_THROW_IN, K_GK_THROW])] = 2.0
    z[np.isin(kick, [K_PUNT])] = 0.7
    z[np.isin(kick, [K_VOLLEY])] = 0.55
    if player_z is not None:
        z += np.where(np.isin(kick, [K_HEADER, K_HEADER_PASS]), player_z, 0.0)
    return np.column_stack([origin_xy, z])
