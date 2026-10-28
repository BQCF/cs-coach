"""N5 单测：verify_pose 处方 + build_findings 聚合。"""

from __future__ import annotations

import polars as pl

from csc import findings
from csc.zones import Zone


class FakeGeo:
    """简化 geo：exposure 距死点 (100,100) 越近越高，挪远暴露面下降，可测 verify_pose。"""

    def __init__(self):
        self.hot = (100.0, 100.0, 0.0)

    def exposure(self, pos, **kw):
        d = ((pos[0] - self.hot[0]) ** 2 + (pos[1] - self.hot[1]) ** 2) ** 0.5
        return (max(0, 3 - int(d // 80)), ["e1"])

    def crossfire(self, pos, **kw):
        return 0

    def escape_time(self, pos, **kw):
        return 2.0


def test_verify_pose_finds_better():
    geo = FakeGeo()
    z = Zone("A主道", (0.0, 0.0), 400.0, site="A")
    vp = findings.verify_pose(geo, z, (100.0, 100.0, 0.0), step=40.0, k=3)
    assert "base" in vp and "candidates" in vp
    assert vp["base"]["exposure"] is not None
    # 候选必须暴露面更低
    for c in vp["candidates"]:
        assert c["exposure"] < vp["base"]["exposure"]


def test_build_findings_position_advice():
    """同 zone >=2 position 死亡 → finding 带 verify 处方。"""
    geo = FakeGeo()
    zmap = {"A主道": Zone("A主道", (0.0, 0.0), 500.0, site="A")}
    frame = pl.DataFrame({
        "zone": ["A主道", "A主道", "A主道"],
        "fault": ["position", "position", "aim"],
        "death_tick": [1, 2, 3],
        "my_x": [100.0, 120.0, 0.0], "my_y": [100.0, 110.0, 0.0], "my_z": [0.0, 0.0, 0.0],
        "exposure": [3, 3, 1], "crossfire": [2, 2, 0], "escape_time": [1.9, 1.9, 0.5],
        "pro_cumulative_percentile": [0.04, 0.04, 0.5],
    })
    fl = findings.build_findings(frame, geo, zmap, min_deaths=2)
    assert len(fl) == 1
    f = fl[0]
    assert f["dominant_fault"] == "position"
    assert f.get("advice") is not None  # 有处方
    assert "verify" in f["advice"]
    assert f["advice"]["verify"]["exposure"][1] < f["advice"]["verify"]["exposure"][0]


def test_build_findings_min_deaths():
    frame = pl.DataFrame({
        "zone": ["A", "B"], "fault": ["position", "position"],
        "death_tick": [1, 2], "my_x": [0.0, 0.0], "my_y": [0.0, 0.0], "my_z": [0.0, 0.0],
    })
    fl = findings.build_findings(frame, None, {}, min_deaths=2)
    assert fl == []  # 每 zone 只 1 个死亡，不生成
