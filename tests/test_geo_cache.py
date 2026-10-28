"""GeoContext.visible 记忆化：exposure/crossfire 与 killer_emerge_tick 的重复查询不打 BVH。"""

from __future__ import annotations

from csc.config import EntryPoint, GeomCfg, MapSpec
from csc.geometry import GeoContext


class CountingVC:
    def __init__(self) -> None:
        self.calls = 0

    def is_visible(self, a, b) -> bool:
        self.calls += 1
        return True


def _geo() -> tuple[GeoContext, CountingVC]:
    vc = CountingVC()
    spec = MapSpec(
        name="de_test",
        regions={"A": {"center": (0.0, 0.0, 0.0), "radius": 5000.0}},
        entry_points={
            "A": (
                EntryPoint("e1", (100.0, 0.0, 0.0)),
                EntryPoint("e2", (0.0, 100.0, 0.0)),
                EntryPoint("e3", (-100.0, 0.0, 0.0)),
            )
        },
    )
    return GeoContext(map_name="de_test", cfg=GeomCfg(), spec=spec, vc=vc), vc


def test_visible_same_args_hits_cache():
    geo, vc = _geo()
    a, b = (0.0, 0.0, 0.0), (100.0, 0.0, 0.0)
    assert geo.visible(a, b, tick=10) is True
    assert geo.visible(a, b, tick=10) is True
    assert vc.calls == 1  # 第二次命中缓存
    assert geo.visible(a, b, tick=11) is True
    assert vc.calls == 2  # 换 tick 是不同查询


def test_crossfire_reuses_exposure_visibility():
    geo, vc = _geo()
    exp, names = geo.exposure((0.0, 0.0, 0.0), tick=20)
    cross = geo.crossfire((0.0, 0.0, 0.0), tick=20)
    assert exp == 3 and len(names) == 3
    assert cross == 3  # 三个暴露点两两夹角都 > 60°
    assert vc.calls == 3  # crossfire 的三次可见性全部命中 exposure 的缓存
