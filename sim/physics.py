"""3D ball physics, batched over N balls (PLAN.md section 3: gravity, quadratic drag, bounce, rolling, Magnus).

All functions take arrays with a leading batch dimension (N, 3) and update them in place, so the same code steps
one match ball or 1,000 rollout balls. Integration is RK2 (midpoint) in the air, which is exact for a drag-free
projectile, plus an exact sub-step at the moment of a bounce.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sim.constants import (
    BALL_RADIUS,
    BOUNCE_FRICTION,
    DRAG_K,
    DT,
    GOAL_DEPTH,
    GOAL_HEIGHT,
    GRAVITY,
    HALF_GOAL,
    HALF_LENGTH,
    MAGNUS_K,
    NET_DAMPING,
    POST_RADIUS,
    REST_SPEED,
    RESTITUTION,
    ROLL_DECEL,
    SPIN_DECAY_AIR,
    SPIN_DECAY_GROUND,
    WOODWORK_RESTITUTION,
)

F = npt.NDArray[np.float64]
B = npt.NDArray[np.bool_]

ROLL_VZ = 0.45
BOUNCE_MU = 0.55
"""Grass friction coefficient for the Coulomb cap on bounce friction."""
ROLL_MAGNUS = 1.0
"""Side-spin turning rate of a rolling ball, as a fraction of the airborne Magnus effect."""
SPIN_DECAY_GROUND_SIDE = 0.5
"""Bounces slower than this (m/s, vertical) turn into rolling."""


@dataclass(frozen=True)
class BallParams:
    drag_k: float = DRAG_K
    magnus_k: float = MAGNUS_K
    restitution: float = RESTITUTION
    bounce_friction: float = BOUNCE_FRICTION
    roll_decel: float = ROLL_DECEL
    gravity: float = GRAVITY


DEFAULT = BallParams()
NO_DRAG = BallParams(drag_k=0.0, magnus_k=0.0)


@dataclass
class StepFlags:
    bounced: B
    woodwork: B
    """Ball hit a post or the crossbar during this step."""


def _air_accel(vel: F, spin: F, p: BallParams) -> F:
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    # omega x v written out (np.cross is slow on small arrays)
    acc = -p.drag_k * speed * vel
    acc[:, 0] += p.magnus_k * (spin[:, 1] * vel[:, 2] - spin[:, 2] * vel[:, 1])
    acc[:, 1] += p.magnus_k * (spin[:, 2] * vel[:, 0] - spin[:, 0] * vel[:, 2])
    acc[:, 2] += p.magnus_k * (spin[:, 0] * vel[:, 1] - spin[:, 1] * vel[:, 0]) - p.gravity
    return acc


def roll(speed: F, dt: float, p: BallParams = DEFAULT) -> tuple[F, F]:
    """Exact rolling update for dv/dt = -(a + k v^2): new speed and distance travelled in ``dt``."""
    a, k = p.roll_decel, p.drag_k
    if k <= 0:
        t_stop = speed / a
        tt = np.minimum(dt, t_stop)
        return np.maximum(speed - a * dt, 0.0), speed * tt - 0.5 * a * tt * tt
    c = np.sqrt(k / a)
    w = np.sqrt(a * k)
    phi = np.arctan(speed * c)
    phi1 = np.maximum(phi - w * dt, 0.0)
    return np.tan(phi1) / c, np.log(np.cos(phi1) / np.cos(phi)) / k


def on_ground(pos: F, vel: F) -> B:
    return np.asarray((pos[:, 2] <= BALL_RADIUS + 1e-6) & (np.abs(vel[:, 2]) < 1e-6))


def step_ball(pos: F, vel: F, spin: F, dt: float = DT, p: BallParams = DEFAULT, frame: bool = True) -> StepFlags:
    """Advance every ball by ``dt`` in place. Returns which balls bounced or hit the woodwork."""
    n = len(pos)
    bounced = np.zeros(n, dtype=bool)
    prev = pos.copy() if frame else pos
    ground = on_ground(pos, vel)
    air = ~ground
    if air.any():
        idx = np.nonzero(air)[0]
        x0, v0, w = pos[idx], vel[idx], spin[idx]
        a0 = _air_accel(v0, w, p)
        v_mid = v0 + 0.5 * dt * a0
        x1 = x0 + dt * v_mid
        v1 = v0 + dt * _air_accel(v_mid, w, p)
        hit = x1[:, 2] < BALL_RADIUS
        if hit.any():
            h = np.nonzero(hit)[0]
            x1[h], v1[h] = _bounce(x0[h], v0[h], v1[h], w[h], dt, p)
            w[h, :2] *= 0.55  # back / top spin is mostly spent on the bounce; side-spin survives it
            w[h, 2] *= 0.9
            bounced[idx[h]] = True
        pos[idx], vel[idx] = x1, v1
        spin[idx] = w * np.exp(-SPIN_DECAY_AIR * dt)
    if ground.any():
        g = np.nonzero(ground)[0]
        vh = vel[g, :2]
        speed = np.linalg.norm(vh, axis=1)
        new_speed, travel = roll(speed, dt, p)
        new_speed[new_speed < REST_SPEED] = 0.0
        # side-spin on a rolling ball turns its direction (weaker than in the air)
        turn = ROLL_MAGNUS * p.magnus_k * spin[g, 2] * dt
        heading = np.arctan2(vh[:, 1], vh[:, 0])
        mid = heading + 0.5 * turn
        moving = speed > 0
        pos[g, 0] += np.where(moving, travel * np.cos(mid), 0.0)
        pos[g, 1] += np.where(moving, travel * np.sin(mid), 0.0)
        pos[g, 2] = BALL_RADIUS
        vel[g, 0] = new_speed * np.cos(heading + turn)
        vel[g, 1] = new_speed * np.sin(heading + turn)
        vel[g, 2] = 0.0
        spin[g, :2] *= np.exp(-SPIN_DECAY_GROUND * dt)
        spin[g, 2] *= np.exp(-SPIN_DECAY_GROUND_SIDE * dt)
    woodwork = collide_goal_frame(prev, pos, vel, dt) if frame else np.zeros(n, dtype=bool)
    return StepFlags(bounced=bounced, woodwork=woodwork)


def _bounce(x0: F, v0: F, v1: F, w: F, dt: float, p: BallParams) -> tuple[F, F]:
    """Exact (gravity-only) impact time inside the step, reflect, then finish the step ballistically."""
    z0 = x0[:, 2] - BALL_RADIUS
    vz0 = v0[:, 2]
    # z0 + vz0 t - g t^2 / 2 = 0, positive root
    disc = np.maximum(vz0**2 + 2 * p.gravity * z0, 0.0)
    tau = np.clip((vz0 + np.sqrt(disc)) / p.gravity, 0.0, dt)
    frac = (tau / dt)[:, None]
    vi = v0 + frac * (v1 - v0)  # velocity at impact (drag folded in linearly)
    xi = x0 + v0 * tau[:, None]
    xi[:, 2] = BALL_RADIUS
    vz_out = -p.restitution * np.minimum(vi[:, 2], 0.0)
    # contact-point slip: v_h + omega x (0, 0, -r)
    slip_x = vi[:, 0] - BALL_RADIUS * w[:, 1]
    slip_y = vi[:, 1] + BALL_RADIUS * w[:, 0]
    vo = vi.copy()
    # Coulomb limit: a grazing bounce (small normal impulse) cannot remove much horizontal speed
    slip = np.hypot(slip_x, slip_y)
    cap = BOUNCE_MU * (1.0 + p.restitution) * np.abs(np.minimum(vi[:, 2], 0.0))
    scale = np.minimum(
        p.bounce_friction, np.divide(cap, slip, out=np.full_like(slip, p.bounce_friction), where=slip > 1e-9)
    )
    vo[:, 0] -= scale * slip_x
    vo[:, 1] -= scale * slip_y
    vo[:, 2] = vz_out
    rest = dt - tau
    rolling = vz_out < ROLL_VZ
    vo[rolling, 2] = 0.0
    xo = xi + vo * rest[:, None]
    flying = ~rolling
    xo[flying, 2] = BALL_RADIUS + vz_out[flying] * rest[flying] - 0.5 * p.gravity * rest[flying] ** 2
    vo[flying, 2] = vz_out[flying] - p.gravity * rest[flying]
    xo[:, 2] = np.maximum(xo[:, 2], BALL_RADIUS)
    vo[rolling, 2] = 0.0
    return xo, vo


def _swept_contact(a0: F, a1: F, b0: F, b1: F, ca: F, cb: F, reach: float) -> tuple[B, F, F, F]:
    """First contact of segments (a0,b0)->(a1,b1) with a disc of radius ``reach`` around (ca, cb) in 2D.

    Returns (contact, s, na, nb): whether the segment enters the disc, the segment parameter of first contact,
    and the unit normal from the disc centre to the ball at contact.
    """
    da, db = a1 - a0, b1 - b0
    ra, rb = a0 - ca, b0 - cb
    qa = da * da + db * db
    qb = 2 * (ra * da + rb * db)
    qc = ra * ra + rb * rb - reach * reach
    disc = qb * qb - 4 * qa * qc
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(qa > 1e-12, (-qb - np.sqrt(np.maximum(disc, 0.0))) / (2 * qa), 0.0)
    inside0 = qc <= 0
    s = np.where(inside0, 0.0, s)
    contact = inside0 | ((disc >= 0) & (s >= 0) & (s <= 1) & (qa > 1e-12))
    pa, pb = ra + s * da, rb + s * db
    d = np.maximum(np.hypot(pa, pb), 1e-9)
    return contact, np.clip(s, 0.0, 1.0), pa / d, pb / d


def collide_goal_frame(prev: F, pos: F, vel: F, dt: float = DT) -> B:
    """Posts, crossbar and net at both ends (swept, so fast shots cannot tunnel). Returns woodwork hits."""
    n = len(pos)
    hit = np.zeros(n, dtype=bool)
    near = (np.abs(pos[:, 0]) > HALF_LENGTH - 2.0) | (np.abs(prev[:, 0]) > HALF_LENGTH - 2.0)
    if not near.any():
        return hit
    idx = np.nonzero(near)[0]
    x0, x, v = prev[idx], pos[idx], vel[idx]
    side = np.where(x[:, 0] + x0[:, 0] >= 0, 1.0, -1.0)
    gx = side * HALF_LENGTH
    reach = BALL_RADIUS + POST_RADIUS
    local_hit = np.zeros(len(idx), dtype=bool)
    done = np.zeros(len(idx), dtype=bool)
    for py in (HALF_GOAL, -HALF_GOAL):
        c, s, nx, ny = _swept_contact(x0[:, 0], x[:, 0], x0[:, 1], x[:, 1], gx, np.full(len(idx), py), reach)
        z_at = x0[:, 2] + s * (x[:, 2] - x0[:, 2])
        vn = v[:, 0] * nx + v[:, 1] * ny
        m = c & (z_at < GOAL_HEIGHT + POST_RADIUS) & (vn < 0) & ~done
        if m.any():
            k = (1 + WOODWORK_RESTITUTION) * vn[m]
            v[m, 0] -= k * nx[m]
            v[m, 1] -= k * ny[m]
            x[m, 0] = gx[m] + nx[m] * reach
            x[m, 1] = py + ny[m] * reach
            x[m, 2] = z_at[m]
            local_hit |= m
            done |= m
    c, s, nx, nz = _swept_contact(x0[:, 0], x[:, 0], x0[:, 2], x[:, 2], gx, np.full(len(idx), GOAL_HEIGHT), reach)
    y_at = x0[:, 1] + s * (x[:, 1] - x0[:, 1])
    vn = v[:, 0] * nx + v[:, 2] * nz
    m = c & (np.abs(y_at) <= HALF_GOAL) & (vn < 0) & ~done
    if m.any():
        k = (1 + WOODWORK_RESTITUTION) * vn[m]
        v[m, 0] -= k * nx[m]
        v[m, 2] -= k * nz[m]
        x[m, 0] = gx[m] + nx[m] * reach
        x[m, 2] = GOAL_HEIGHT + nz[m] * reach
        x[m, 1] = y_at[m]
        local_hit |= m
    # net: inside the goal volume the ball is caught by the netting
    inside = (np.abs(x[:, 0]) > HALF_LENGTH) & (np.abs(x[:, 1]) < HALF_GOAL) & (x[:, 2] < GOAL_HEIGHT)
    inside &= np.abs(x0[:, 1]) < HALF_GOAL + 0.5
    inside &= np.abs(x0[:, 0]) <= HALF_LENGTH + GOAL_DEPTH  # it came in through the mouth, not round the back
    if inside.any():
        v[inside] *= np.exp(-NET_DAMPING * dt)
        back = side[inside] * (HALF_LENGTH + GOAL_DEPTH - BALL_RADIUS)
        xi, vi = x[inside], v[inside]
        over = np.abs(xi[:, 0]) > np.abs(back)
        xi[over, 0] = back[over]
        vi[over, 0] = 0.0
        x[inside], v[inside] = xi, vi
    pos[idx], vel[idx] = x, v
    hit[idx] = local_hit
    return hit


def predict(pos: F, vel: F, spin: F, n_steps: int, dt: float = DT, p: BallParams = DEFAULT) -> tuple[F, F]:
    """Noise-free trajectories: positions and velocities of shape (n_steps + 1, N, 3), index 0 = now."""
    x, v, w = pos.astype(np.float64).copy(), vel.astype(np.float64).copy(), spin.astype(np.float64).copy()
    xs = np.empty((n_steps + 1, len(x), 3))
    vs = np.empty((n_steps + 1, len(x), 3))
    xs[0], vs[0] = x, v
    for k in range(1, n_steps + 1):
        step_ball(x, v, w, dt, p)
        xs[k], vs[k] = x, v
    return xs, vs


def topspin_axis(azimuth: F) -> F:
    """Unit spin axis that gives topspin for a ball travelling along ``azimuth`` (z x v_hat)."""
    out = np.zeros((len(azimuth), 3))
    out[:, 0] = -np.sin(azimuth)
    out[:, 1] = np.cos(azimuth)
    return out
