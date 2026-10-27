from pathlib import Path

import numpy as np
import pytest

from analytics.data import metrica

HEADER = [
    ",,,Home,,Home,,,",
    ",,,1,,2,,,",
    "Period,Frame,Time [s],Player1,,Player2,,Ball,",
]


def _write(tmp: Path, rows: list[str]) -> None:
    d = tmp / "Sample_Game_9"
    d.mkdir()
    (d / "Sample_Game_9_RawTrackingData_Home_Team.csv").write_text("\n".join(HEADER + rows) + "\n")
    (d / "Sample_Game_9_RawEventsData.csv").write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
        "From,To,Start X,Start Y,End X,End Y\n"
        "Home,PASS,,1,1,0.04,3,0.12,Player1,Player2,0.5,0.5,0.75,0.25\n"
    )


def test_load_tracking_converts_to_metres(tmp_path: Path) -> None:
    rows = [f"1,{i + 1},{(i + 1) * 0.04:.2f},0.0,0.5,{0.5 + i * 0.001},0.5,NaN,NaN" for i in range(20)]
    _write(tmp_path, rows)
    tr = metrica.load_tracking(9, "Home", root=tmp_path)
    assert tr.players == ["Player1", "Player2"]
    assert tr.pos.shape == (20, 2, 2)
    assert tr.pos[0, 0].tolist() == [-52.5, 0.0]
    assert np.isnan(tr.ball).all()
    assert metrica.goalkeeper_index(tr) == 0
    vel = metrica.velocities(tr.pos, tr.period)
    # Player2 moves 0.001 * 105 m per frame = 2.625 m/s along +x.
    assert np.nanmedian(vel[:, 1, 0]) == pytest.approx(2.625, rel=1e-6)
    assert np.nanmax(np.abs(vel[:, 0])) == pytest.approx(0.0)


def test_load_events(tmp_path: Path) -> None:
    _write(tmp_path, ["1,1,0.04,0.0,0.5,0.5,0.5,0.5,0.5"])
    (ev,) = metrica.load_events(9, root=tmp_path)
    assert ev.type == "PASS" and ev.to_player == "Player2"
    assert ev.start == (0.0, 0.0)
    assert ev.end == pytest.approx((26.25, 17.0))


def test_smooth_is_nan_aware() -> None:
    a = np.arange(20, dtype=float)
    a[10] = np.nan
    s = metrica.smooth(a)
    assert np.isnan(s[:3]).all() and np.isnan(s[7:14]).all()
    assert s[5] == pytest.approx(5.0)
