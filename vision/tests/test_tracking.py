import numpy as np

from vision.tracking import Detection, FrameResult, assign_teams, gate_ball, kit_hist


def test_assign_teams_by_kit_hue() -> None:
    home = kit_hist(("#c8102e", "#1e88e5", "halves"))  # red / blue halves
    away = kit_hist(("#1e88e5", "#7b1e3a", "plain"))  # blue
    rng = np.random.default_rng(0)
    cols = {}
    for k in range(10):
        base = home if k < 5 else away
        noisy = np.clip(base + rng.normal(0, 0.03, base.shape), 0, None)
        cols[k] = noisy / noisy.sum()
    teams = assign_teams(cols, {"home": home, "away": away})
    assert [teams[k] for k in range(10)] == ["home"] * 5 + ["away"] * 5


def ball(t: float, x: float, conf: float = 0.9) -> FrameResult:
    d = Detection(-1, 4, conf, (0, 0, 1, 1), (x, 0.0))
    return FrameResult(t, [], d, [d])


def test_gate_ball_rejects_implausible_jumps() -> None:
    res = [ball(0.0, 0.0), ball(0.04, 0.5), ball(0.08, 30.0), ball(0.12, 1.5), ball(0.16, 2.0, conf=0.2)]
    out = gate_ball(res)
    assert out[:, 0].tolist()[:2] == [0.0, 0.5]
    assert np.isnan(out[2, 0])  # 30 m in 0.04 s
    assert out[3, 0] == 1.5 and out[4, 0] == 2.0
