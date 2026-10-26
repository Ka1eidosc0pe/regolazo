import numpy as np

from shared.model_api import (
    SHOT_BODY_PART,
    SHOT_SITUATION,
    AnalyticsModels,
    PassFeatures,
    PostShotFeatures,
    ShotFeatures,
    StubModels,
    decision_value,
    load_models,
    shot_geometry,
)


def _shots(x: list[float], y: list[float], **over: object) -> ShotFeatures:
    n = len(x)
    base: dict[str, object] = {
        "x": np.array(x),
        "y": np.array(y),
        "body_part": np.zeros(n, dtype=np.int64),
        "technique": np.zeros(n, dtype=np.int64),
        "situation": np.zeros(n, dtype=np.int64),
        "gk_x": np.full(n, 51.5),
        "gk_y": np.zeros(n),
        "n_defenders_in_cone": np.zeros(n, dtype=np.int64),
        "nearest_defender_m": np.full(n, 3.0),
        "under_pressure": np.zeros(n, dtype=bool),
        "first_time": np.zeros(n, dtype=bool),
    }
    base.update(over)
    return ShotFeatures(**base)  # type: ignore[arg-type]


def test_stub_satisfies_protocol_and_loader_falls_back(tmp_path: object) -> None:
    assert isinstance(StubModels(), AnalyticsModels)
    assert isinstance(load_models(), AnalyticsModels)


def test_shot_geometry() -> None:
    dist, angle = shot_geometry(np.array([41.5]), np.array([0.0]))
    assert abs(dist[0] - 11.0) < 1e-9
    assert abs(angle[0] - 2 * np.arctan(3.66 / 11.0)) < 1e-9


def test_stub_xg_shape_and_monotonicity() -> None:
    m = StubModels()
    xg = m.xg(_shots([46.0, 36.0, 20.0, 36.0], [0.0, 0.0, 0.0, 25.0]))
    assert xg.shape == (4,) and np.all((xg > 0) & (xg < 1))
    assert xg[0] > xg[1] > xg[2] and xg[1] > xg[3]
    header = m.xg(_shots([46.0], [0.0], body_part=np.array([SHOT_BODY_PART["head"]])))
    assert header[0] < xg[0]
    pen = m.xg(_shots([41.5], [0.0], situation=np.array([SHOT_SITUATION["penalty"]])))
    assert abs(pen[0] - 0.76) < 1e-9
    # 6-yard box central shot should be a big chance, 30 m shot a long shot
    assert xg[0] > 0.3 and xg[2] < 0.05


def test_stub_psxg_corner_beats_centre() -> None:
    m = StubModels()
    f = PostShotFeatures(
        end_y=np.array([0.0, 3.3]),
        end_z=np.array([0.5, 2.2]),
        speed_mps=np.array([25.0, 25.0]),
        distance_m=np.array([16.0, 16.0]),
        gk_x=np.array([51.5, 51.5]),
        gk_y=np.array([0.0, 0.0]),
        gk_dive=np.array([50.0, 50.0]),
    )
    p = m.psxg(f)
    assert p[1] > p[0]


def test_stub_xpass_and_xt_and_pitch_control() -> None:
    m = StubModels()
    f = PassFeatures(
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 0.0]),
        end_x=np.array([8.0, 40.0]),
        end_y=np.array([0.0, 0.0]),
        height=np.zeros(2, dtype=np.int64),
        kind=np.zeros(2, dtype=np.int64),
        nearest_defender_to_receiver_m=np.array([5.0, 2.0]),
        defenders_in_lane=np.array([0, 2]),
        under_pressure=np.zeros(2, dtype=bool),
    )
    p = m.xpass(f)
    assert p[0] > 0.85 and p[1] < p[0]
    xt = m.xt(np.array([-40.0, 0.0, 45.0]), np.array([0.0, 0.0, 0.0]))
    assert xt[0] < xt[1] < xt[2]
    att = np.array([[10.0, 0.0]])
    dfn = np.array([[-10.0, 0.0]])
    zero = np.zeros((1, 2))
    pc = m.pitch_control(att, zero, dfn, zero, np.zeros(2), np.array([[10.0, 0.0], [-10.0, 0.0], [0.0, 0.0]]))
    assert pc[0] > 0.9 and pc[1] < 0.1 and abs(pc[2] - 0.5) < 1e-9


def test_decision_value() -> None:
    v = decision_value(np.array([0.5]), np.array([0.2]), np.array([-0.1]))
    assert abs(v[0] - 0.05) < 1e-12
