"""N4 画像单测：六维计算 + 短板排序 + 雷达归一化。"""

from __future__ import annotations

import polars as pl

from csc import profile


def _mk_frame():
    return pl.DataFrame({
        "first_fire": ["me", "me", "him", "him", "him"],
        "fault": ["none", "aim", "position", "info", "none"],
        "first_seen_gap_ms": [-100, 50, 0, None, 200],
        "my_aim_off_at_see_deg": [2.0, 20.0, 5.0, 8.0, 30.0],
        "pro_cumulative_percentile": [0.5, 0.1, 0.3, 0.2, 0.6],
        "escape_time": [1.0, 2.0, 3.0, 0.5, 4.0],
    })


def test_compute_dims():
    dims = profile.compute_dims(_mk_frame())
    assert "duel_win" in dims and "aim_off" in dims
    # 对枪转化：first_fire=me 共 2 个，其中 1 个 fault=none → 50%
    assert dims["duel_win"]["value"] == 50.0
    assert dims["duel_win"]["n"] == 2
    # 预瞄偏差中位 = 8.0
    assert dims["aim_off"]["value"] == 8.0


def test_short_board_min_n_guard():
    dims = {"duel_win": {"value": 0.0, "n": 3}, "aim_off": {"value": 20.0, "n": 13}}
    sb = profile.short_board(dims, min_n=10)
    # n=3 标样本不足
    assert any("样本不够" in s for s in sb)
    # n=13 参与排序
    assert any("准星偏差" in s for s in sb)


def test_radar_points_range():
    dims = {
        "duel_win": {"value": 80.0, "n": 10},
        "gap_ms": {"value": -50.0, "n": 10},
        "aim_off": {"value": 5.0, "n": 10},
        "zone_norm": {"value": 0.7, "n": 10},
        "escape": {"value": 1.0, "n": 10},
        "info": {"value": 10.0, "n": 10},
    }
    pts = profile.radar_points(dims)
    assert len(pts) == 6
    assert all(0 <= p <= 1 for p in pts)
