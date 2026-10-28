"""Technique test (PLAN.md Step 3): each technique gives a distinguishable flight matching the taxonomy's
kinematic definitions (docs/event_taxonomy.md 3.1.2 and 3.5.1), and the resolver's choices hold in play."""

import numpy as np

from shared.schemas.actions import ActionIntent
from sim import kick as K
from sim.constants import BALL_RADIUS, HALF_LENGTH
from sim.physics import predict
from sim.tests.scenes import base_match, engine_for, home_ids, place


def _flight(kick: int, dist: float, curve: float = 0.0, power: float = 0.5, z_target: float = 0.0) -> dict[str, float]:
    origin = np.array([[0.0, 0.0, BALL_RADIUS]])
    target = np.array([[dist, 0.0, z_target]])
    plan = K.plan_kicks(origin, target, np.array([kick]), np.array([power]), np.array([curve]))
    if kick != K.K_GROUND or curve:
        plan = K.refine(plan)
    v = plan.launch.velocity()
    if kick == K.K_GROUND and not curve:
        v[:, 2] = 0.0
    xs, vs = predict(origin, v, plan.launch.spin, 100)
    path = xs[:, 0]
    along = path[:, 0]
    k = int(np.argmax(along >= dist)) if (along >= dist).any() else len(path) - 1
    seg = path[: k + 1]
    return {
        "peak": float(seg[:, 2].max()),
        "speed": float(np.linalg.norm(vs[0, 0])),
        "elev_deg": float(np.degrees(plan.launch.elev[0])),
        "curl": float(np.abs(seg[:, 1]).max()),
        "time": k * 0.04,
    }


def test_pass_techniques_have_distinct_flights() -> None:
    std = _flight(K.K_GROUND, 20.0)
    drv = _flight(K.K_DRIVEN, 30.0, power=0.7)
    lob = _flight(K.K_LOFTED, 40.0)
    chip = _flight(K.K_CHIPPED, 18.0)
    curl = _flight(K.K_CURLED, 25.0, curve=0.8)
    cross = _flight(K.K_CROSS, 30.0, z_target=1.7)
    assert std["peak"] <= 0.3  # ground (taxonomy 3.1.1)
    assert drv["elev_deg"] <= 12 and drv["speed"] >= 18  # driven
    assert lob["elev_deg"] >= 20 and lob["peak"] > 1.5  # lofted
    assert chip["elev_deg"] >= 35 and chip["peak"] > lob["peak"] * 0.5  # chipped: steep, short
    assert curl["curl"] >= 1.5  # curled: >= 1.5 m off the chord
    assert cross["peak"] > 1.5
    assert drv["speed"] > std["speed"] and drv["time"] < lob["time"]


def test_shot_techniques_have_distinct_flights() -> None:
    drv = _flight(K.K_SHOT_DRIVEN, 20.0, power=0.8, z_target=0.6)
    placed = _flight(K.K_SHOT_PLACED, 20.0, z_target=0.6)
    finesse = _flight(K.K_SHOT_FINESSE, 20.0, curve=-0.7, z_target=1.2)
    chip = _flight(K.K_SHOT_CHIP, 20.0, z_target=1.9)
    assert drv["speed"] >= 25.0 > placed["speed"]  # taxonomy: driven >= 25 m/s, placed < 25
    assert finesse["curl"] >= 1.0 and finesse["speed"] < 25.0
    assert chip["elev_deg"] >= 30 and chip["peak"] > 2.4


def test_skill_changes_noise_not_the_plan() -> None:
    rng = np.random.default_rng(0)
    origin = np.tile([0.0, 0.0, BALL_RADIUS], (400, 1))
    target = np.tile([25.0, 0.0, 0.0], (400, 1))
    kicks = np.full(400, K.K_GROUND)
    plan = K.plan_kicks(origin, target, kicks, np.full(400, 0.5), np.zeros(400))
    spread = {}
    for q in (0.2, 0.9):
        mult = K.noise_multiplier(np.full(400, q), np.zeros(400), np.full(400, 0.5), np.zeros(400))
        noisy = K.perturb(plan.launch, kicks, mult, rng)
        spread[q] = float(np.std(noisy.azimuth))
    assert spread[0.2] > 2.0 * spread[0.9]


def test_striker_and_fullback_strike_differently() -> None:
    """PLAN 2.6: a striker's driven finish and a fullback's speculative effort look and behave differently."""
    m0, _ = base_match()
    h = home_ids(m0)
    st, fb = h[9], h[2]
    strike = {"finishing": 92, "shot_power": 92, "long_shots": 88, "composure": 90}
    weak = {"finishing": 35, "shot_power": 55, "long_shots": 35, "composure": 45}
    m, _ = base_match(attr_overrides={st: strike, fb: weak})
    out = {}
    for pid in (st, fb):
        place(m, pid, (30.0, 4.0))
        eng, ref = engine_for(m)
        r = eng.preview(
            ref, ActionIntent(kind="shot", player=pid, target=(HALF_LENGTH, -2.5, 0.5), power=0.9, source="fan")
        )
        out[pid] = r
    assert out[st].shot_technique == "driven"
    la_st, la_fb = out[st].launch, out[fb].launch
    el_st, el_fb = out[st].ellipse, out[fb].ellipse
    assert la_st is not None and la_fb is not None and el_st is not None and el_fb is not None
    assert la_st.speed_mps > la_fb.speed_mps
    assert el_fb.semi_major_m > 1.3 * el_st.semi_major_m
