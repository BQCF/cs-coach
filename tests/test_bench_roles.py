"""bench_roles / benchmark 的 M4 数学与角色单测（此前该模块零覆盖）。"""

from __future__ import annotations

import polars as pl

from csc import bench_roles, benchmark
from csc.config import BenchCfg, MapSpec

CFG = BenchCfg()
SPEC = MapSpec(
    name="de_test",
    regions={
        "A": {"center": (0.0, 0.0, 0.0), "radius": 1100.0},
        "B": {"center": (5000.0, 0.0, 0.0), "radius": 1100.0},
    },
)


def _track(points):
    return pl.DataFrame(
        [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in points],
        schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64},
    )


# --------------------------------------------------------------------------- 分位


def test_neighborhood_frequency():
    pts = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (5000.0, 0.0, 0.0)]
    assert abs(bench_roles.neighborhood_frequency((0.0, 0.0, 0.0), pts, 150.0) - 2 / 3) < 1e-9
    assert bench_roles.neighborhood_frequency((0.0, 0.0, 0.0), [], 150.0) == 0.0


def test_cumulative_percentile_hot_vs_cold():
    # 网格 96：(0,0) 是 5 点的热门格，(96,0) 是 1 点的冷门格
    points = [(0.0, 0.0, 0.0)] * 5 + [(200.0, 0.0, 0.0)]
    grid = benchmark.grid_frame(points, 96.0)
    hot = bench_roles.cumulative_percentile((5.0, 5.0, 0.0), grid, 96.0)
    cold = bench_roles.cumulative_percentile((200.0, 0.0, 0.0), grid, 96.0)
    assert hot is not None and cold is not None
    assert hot > cold  # 越热门累积分位越高


def test_cumulative_percentile_empty_grid():
    assert bench_roles.cumulative_percentile((0.0, 0.0, 0.0), pl.DataFrame(), 96.0) is None


def test_percentile_for_hit_and_miss():
    grid = benchmark.grid_frame([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], 96.0)
    assert benchmark.percentile_for((1.0, 1.0, 0.0), grid, 96.0) == 1.0
    assert benchmark.percentile_for((500.0, 500.0, 0.0), grid, 96.0) == 0.0
    assert benchmark.percentile_for((0.0, 0.0, 0.0), pl.DataFrame(), 96.0) is None


# --------------------------------------------------------------------------- 同经济


def test_same_economy_and_bucket():
    assert bench_roles.same_economy(1000.0, 1400.0)
    assert not bench_roles.same_economy(1000.0, 2000.0)
    assert bench_roles.economy_bucket(1000.0, 500.0) == 2


# --------------------------------------------------------------------------- 角色


def test_ct_role_anchor_a():
    pts = _track([(0.0, 0.0, 0.0)] * 10 + [(50.0, 0.0, 0.0)] * 5)
    assert bench_roles.classify_round_role(pts, SPEC, CFG, side="counter-terrorist") == "ct_A_main"


def test_ct_role_anchor_b():
    pts = _track([(5000.0, 0.0, 0.0)] * 10)
    assert bench_roles.classify_round_role(pts, SPEC, CFG, side="counter-terrorist") == "ct_B_main"


def test_t_role_by_weapon_and_bomb():
    pts = _track([(2500.0, 0.0, 0.0)] * 5)
    assert bench_roles.classify_round_role(pts, SPEC, CFG, side="terrorist", primary_weapon="awp") == "t_awp"
    assert bench_roles.classify_round_role(pts, SPEC, CFG, side="terrorist", has_bomb=True) == "t_entry"


def test_t_role_lurker_and_trader():
    # 大范围游走 = 自由人；小范围抱团不进包点 = 补枪手
    spread = _track([(2500.0, 0.0, 0.0), (3500.0, 0.0, 0.0), (1500.0, 0.0, 0.0), (2500.0, 800.0, 0.0)])
    assert bench_roles.classify_round_role(spread, SPEC, CFG, side="terrorist") == "t_lurker"
    tight = _track([(2500.0, 0.0, 0.0)] * 6)
    assert bench_roles.classify_round_role(tight, SPEC, CFG, side="terrorist") == "t_trader"


def test_role_empty_track_is_unknown():
    assert bench_roles.classify_round_role(pl.DataFrame(), SPEC, CFG, side="terrorist") == "unknown"
