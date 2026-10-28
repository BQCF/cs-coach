"""报告配图单测：report_charts 生成 4 张图。"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from csc import report_charts


def test_generate_charts(tmp_path):
    frame = pl.DataFrame({
        "fault": ["aim", "aim", "position", "preaim", "none"],
        "my_aim_off_at_see_deg": [2.0, 20.0, 5.0, 44.0, 1.0],
        "killer_aim_off_deg": [1.0, 2.0, 1.0, 3.0, 0.5],
        "my_dmg": [0, 50, 30, 20, 10],
        "his_dmg": [42, 61, 32, 56, 88],
        "death_tick": [1, 2, 3, 4, 5],
        "my_x": [0.0, 10.0, 20.0, 30.0, 40.0],
        "my_y": [0.0, 10.0, 20.0, 30.0, 40.0],
    })
    made = report_charts.generate(frame, tmp_path)
    # 至少 fault/duel/scatter 三张
    assert "fault_distribution" in made
    assert "duel_damage" in made
    assert "death_scatter" in made
    for name in made.values():
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 0
