"""Player steering on the pitch plane plus a vertical jump component (PLAN.md section 3).

Batched over any leading shape (e.g. (B, P)). Players steer toward a target point with a speed cap: forward
acceleration falls off with speed, braking uses the (larger) deceleration limit, and the total acceleration is
clipped to the player's physical limits, which gives a natural turning radius at speed.
"""

import numpy as np
import numpy.typing as npt

from sim.constants import DT, GRAVITY, MAX_TURN_RATE

F = npt.NDArray[np.float64]


def steer(
    pos: F,
    vel: F,
    target: F,
    speed_cap: F,
    max_speed: F,
    max_accel: F,
    max_decel: F,
    dt: float = DT,
    arrive_gain: float = 0.75,
    tau: F | float = 0.4,
) -> tuple[F, F]:
    """One kinematic step toward ``target``. Shapes: pos/vel/target (..., 2), the rest (...). Returns new pos, vel.

    The velocity relaxes toward the desired velocity with time constant ``tau`` and is clipped to the limits.
    """
    to = target - pos
    dist = np.linalg.norm(to, axis=-1)
    direction = to / np.maximum(dist, 1e-9)[..., None]
    cap = np.minimum(speed_cap, max_speed)
    # brake so we can stop at the target
    arrive = np.sqrt(2.0 * arrive_gain * max_decel * dist)
    desired_speed = np.minimum(cap, arrive)
    desired_speed = np.where(dist < 0.05, 0.0, desired_speed)
    v_des = direction * desired_speed[..., None]
    # first-order response (time constant tau): small corrections are gentle, big ones saturate the limits
    dv = (v_des - vel) * np.minimum(1.0, dt / np.asarray(tau))[..., None]
    speed = np.linalg.norm(vel, axis=-1)
    u = vel / np.maximum(speed, 1e-9)[..., None]
    par = np.sum(dv * u, axis=-1)
    perp = dv - par[..., None] * u
    a_fwd = max_accel * np.clip(1.0 - 0.55 * speed / np.maximum(max_speed, 1e-9), 0.3, 1.0)
    lim_par = np.where(par >= 0, a_fwd, max_decel) * dt
    par_c = np.clip(par, -lim_par, lim_par)
    lim_perp = np.maximum(max_accel, max_decel) * 0.85 * dt
    perp_n = np.linalg.norm(perp, axis=-1)
    perp_c = perp * np.minimum(1.0, lim_perp / np.maximum(perp_n, 1e-9))[..., None]
    dv_c = par_c[..., None] * u + perp_c
    # never exceed the overall acceleration limit
    lim = np.maximum(max_accel, max_decel) * dt
    n = np.linalg.norm(dv_c, axis=-1)
    dv_c = dv_c * np.minimum(1.0, lim / np.maximum(n, 1e-9))[..., None]
    v_new = vel + dv_c
    s_new = np.linalg.norm(v_new, axis=-1)
    v_new = v_new * np.minimum(1.0, max_speed / np.maximum(s_new, 1e-9))[..., None]
    pos_new = pos + 0.5 * (vel + v_new) * dt
    return pos_new, v_new


def turn_toward(facing: F, desired: F, speed: F, dt: float = DT) -> F:
    """Rotate body orientation toward ``desired`` with a speed-dependent turn-rate limit."""
    rate = MAX_TURN_RATE / (1.0 + 0.35 * speed)
    diff = np.arctan2(np.sin(desired - facing), np.cos(desired - facing))
    out = facing + np.clip(diff, -rate * dt, rate * dt)
    return np.asarray(np.arctan2(np.sin(out), np.cos(out)))


def step_jump(z: F, vz: F, dt: float = DT) -> tuple[F, F]:
    """Ballistic body height for jumps and dives; lands at z = 0."""
    airborne = (z > 0) | (vz > 0)
    z_new = np.where(airborne, z + vz * dt - 0.5 * GRAVITY * dt * dt, 0.0)
    vz_new = np.where(airborne, vz - GRAVITY * dt, 0.0)
    landed = z_new <= 0
    return np.where(landed, 0.0, z_new), np.where(landed, 0.0, vz_new)


def jump_speed(jumping: F) -> F:
    """Take-off speed from the Jumping attribute (1..99): 2.6..3.6 m/s, i.e. 0.34..0.66 m of lift."""
    return np.asarray(2.6 + 1.0 * np.clip(jumping, 1, 99) / 99.0)


def drain_stamina(stamina: F, speed: F, stamina_attr: F, dt: float = DT) -> F:
    """Fatigue: high-intensity running drains, walking recovers. Tuned so an average player ends a match ~0.6."""
    endurance = 0.55 + 0.9 * np.clip(stamina_attr, 1, 99) / 99.0
    per_s = np.where(speed > 5.5, 9e-4, np.where(speed > 3.5, 1.2e-4, -3e-5))
    drain = np.where(per_s > 0, per_s / endurance, per_s * endurance)
    return np.asarray(np.clip(stamina - drain * dt, 0.0, 1.0))


def effective_max_speed(max_speed: F, stamina: F) -> F:
    return np.asarray(max_speed * (0.9 + 0.1 * stamina))
