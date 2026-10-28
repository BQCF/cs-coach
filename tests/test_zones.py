"""N2 zone 单测：zone_of 语义 + suggest 覆盖。"""

from __future__ import annotations

from csc import zones
from csc.zones import Zone


def test_zone_of_inside():
    zs = [Zone("B点-门", (1000.0, 200.0), 240.0, site="B"),
          Zone("A主道", (-500.0, -600.0), 300.0, site="A")]
    name, d = zones.zone_of(zs, (1000.0, 250.0, 0.0))
    assert name == "B点-门"
    assert d < 240


def test_zone_of_outside_gets_tilde():
    zs = [Zone("B点-门", (1000.0, 200.0), 240.0, site="B")]
    name, d = zones.zone_of(zs, (2000.0, 200.0, 0.0))
    assert name == "~B点-门"   # 最近但在半径外 → 加 ~ 前缀
    assert d > 240


def test_zone_of_empty():
    name, d = zones.zone_of([], (0.0, 0.0, 0.0))
    assert name == "unknown"


def test_suggest_zones_produces_named():
    pts = [(-1000.0, 0.0, 0.0)] * 60 + [(500.0, 300.0, 0.0)] * 60 + [(500.0, 300.0, 0.0)] * 40
    zs = zones.suggest_zones(pts, n=10, eps=250.0, min_samples=20)
    assert len(zs) >= 1
    assert all(z.name.startswith("zone_") for z in zs)
    assert all(z.radius > 0 for z in zs)


def test_zone_of_attrib_all_points():
    """任何点都能归属（unknown=0）——§4.5 验收。"""
    zs = [Zone("a", (0.0, 0.0), 300.0), Zone("b", (1000.0, 0.0), 300.0)]
    for pos in [(0.0, 0.0, 0.0), (1000.0, 0.0, 0.0), (500.0, 999.0, 0.0), (-3000.0, 0.0, 0.0)]:
        name, _ = zones.zone_of(zs, pos)
        assert name != "unknown"
