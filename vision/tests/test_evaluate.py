from vision.evaluate import Report, Score, combine, match


def test_match_is_one_to_one_within_tolerance_and_team() -> None:
    det = [(10.0, "home"), (10.3, "home"), (20.0, "away"), (30.0, "home")]
    gt = [(10.1, "home"), (20.5, "home"), (31.5, "home")]
    s = match(det, gt, tol=1.0)  # type: ignore[arg-type]
    assert (s.tp, s.fp, s.fn) == (1, 3, 2)
    assert s.pairs == [(0, 0)]
    assert abs(s.precision - 0.25) < 1e-9 and abs(s.recall - 1 / 3) < 1e-9


def test_combine_pools_counts_and_weights_mae() -> None:
    a = Report(0, 1, 1.0, {"pass": Score(2, 1, 1, [])}, {"speed": (1.0, 10, "km/h")}, (1.0, 10))
    b = Report(2, 3, 1.0, {"pass": Score(1, 0, 3, [])}, {"speed": (4.0, 30, "km/h")}, (0.5, 30))
    c = combine([a, b])
    assert (c.scores["pass"].tp, c.scores["pass"].fp, c.scores["pass"].fn) == (3, 1, 4)
    assert c.mae["speed"] == (3.25, 40, "km/h")
    assert c.team_accuracy == (0.625, 40)
