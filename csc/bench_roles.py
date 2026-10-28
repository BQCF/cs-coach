"""M4 规格实现（用户拍板 2026-10-23）：累积分位 + 邻域频率 + 8 角色 + 同经济匹配。

分位语义：
* ``neighborhood_frequency`` —— 某位置 ``neighborhood_radius`` 内的职业落位占比（0~1）。
  越高 = 职业越常站这附近 = 站位越常规。
* ``cumulative_percentile`` —— 累积分位（0~1）：比该网格更冷门的网格总占比。越高越热门。

经济匹配：equipment_value 差 <= ``economy_tolerance``（默认 500）算同经济；能对上同枪械更好。

角色定位（8 个，按回合判定）：
* 防守 CT：``ct_A_main`` / ``ct_A_support`` / ``ct_B_main`` / ``ct_B_support``
* 进攻 T：``t_entry``(突破手) / ``t_lurker``(自由人) / ``t_trader``(补枪手) / ``t_awp``(AWP)

普通玩家没有职业那种固定角色，分析用户自己的 demo 时以「回合」为单位判定角色
（每回合角色可能不同），不看整场固定角色。
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

from .config import BenchCfg, MapSpec, Vec3, dist2

# 防守 / 进攻 的 8 个角色名
CT_ROLES = ("ct_A_main", "ct_A_support", "ct_B_main", "ct_B_support")
T_ROLES = ("t_entry", "t_lurker", "t_trader", "t_awp")
ALL_ROLES = CT_ROLES + T_ROLES

AWP_WEAPONS = {"awp", "ssg08", "scar20", "g3sg1"}  # 狙击枪族


# --------------------------------------------------------------------------- 分位


def neighborhood_frequency(pos: Vec3, points: Sequence[Vec3], radius: float) -> float:
    """pos 周围 radius 内的落位点占总点的比例（邻域频率，0~1）。"""
    if not points:
        return 0.0
    inside = sum(1 for p in points if dist2(pos, p) <= radius)
    return inside / len(points)


def cumulative_percentile(pos: Vec3, grid: pl.DataFrame, grid_size: float) -> float | None:
    """累积分位：比 pos 所在网格更冷门的网格总占比（0~1）。0=最冷门，1=最热门。"""
    if grid is None or grid.is_empty():
        return None
    cx = int(pos[0] // grid_size)
    cy = int(pos[1] // grid_size)
    hit = grid.filter((pl.col("cx") == cx) & (pl.col("cy") == cy))
    my_count = int(hit["count"][0]) if not hit.is_empty() else 0
    total = int(grid["count"].sum())
    if total == 0:
        return None
    # 比我更冷门的格子的点数总和 / 总点数
    colder = grid.filter(pl.col("count") < my_count)
    colder_pts = int(colder["count"].sum())
    return colder_pts / total


# --------------------------------------------------------------------------- 同经济


def same_economy(value_a: float, value_b: float, tol: float = 500.0) -> bool:
    """装备价值差 <= tol = 同经济。"""
    return abs(float(value_a) - float(value_b)) <= tol


def economy_bucket(value: float, tol: float = 500.0) -> int:
    """把装备价值量化成桶（便于 group_by 比较同经济）。"""
    return int(round(float(value) / tol))


# --------------------------------------------------------------------------- 角色


def _zone_ratio(points: Sequence[Vec3], center: Vec3, radius: float) -> float:
    if not points:
        return 0.0
    return sum(1 for p in points if dist2(p, center) <= radius) / len(points)


def classify_round_role(
    track: pl.DataFrame,
    spec: MapSpec,
    cfg: BenchCfg,
    *,
    side: str,
    has_bomb: bool = False,
    primary_weapon: str | None = None,
) -> str:
    """以「单回合轨迹」判定角色。普通玩家每回合角色可不同。

    判定优先级：
    * T 且用狙 → ``t_awp``
    * T 带包 → ``t_entry``（突破手/带包推进）
    * CT：按包点停留 + 到达时间判 main/support
    """
    pts = _track_points(track)
    if not pts:
        return "unknown"
    weapon = (primary_weapon or "").lower()

    if side == "terrorist":
        if weapon in AWP_WEAPONS:
            return "t_awp"
        if has_bomb:
            return "t_entry"
        # 突破手：早期快速推进到包点附近；自由人：分散游走；补枪手：跟在主攻后
        a_ratio = _zone_ratio(pts, _center(spec, "A"), cfg.anchor_radius)
        b_ratio = _zone_ratio(pts, _center(spec, "B"), cfg.anchor_radius)
        if max(a_ratio, b_ratio) >= cfg.anchor_hold_ratio:
            return "t_entry"
        if _dispersion(pts) > cfg.anchor_radius * 2:
            return "t_lurker"
        return "t_trader"

    # CT：A/B 区主防 / 协防 —— 按"离哪个包点更近"归属，再按停留占比分主/协
    a_center, b_center = _center(spec, "A"), _center(spec, "B")
    # 质心到 A/B 的距离决定主要防守区
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    centroid = (cx, cy, 0.0)
    near = "A" if dist2(centroid, a_center) <= dist2(centroid, b_center) else "B"
    ratio = _zone_ratio(pts, a_center if near == "A" else b_center, cfg.anchor_radius * 1.5)
    if near == "A":
        return "ct_A_main" if ratio >= cfg.anchor_hold_ratio else "ct_A_support"
    return "ct_B_main" if ratio >= cfg.anchor_hold_ratio else "ct_B_support"


def _center(spec: MapSpec, region: str) -> Vec3:
    r = spec.regions.get(region) or {}
    c = r.get("center")
    return tuple(c) if c else (0.0, 0.0, 0.0)


def _track_points(track: pl.DataFrame | None) -> list[Vec3]:
    if track is None or track.is_empty() or "x" not in track.columns:
        return []
    return [(float(r["x"]), float(r["y"]), float(r.get("z") or 0.0)) for r in track.to_dicts()]


def _dispersion(points: Sequence[Vec3]) -> float:
    """点集的平均到质心距离（衡量游走 vs 固守）。"""
    if len(points) < 2:
        return 0.0
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sum(dist2(p, (cx, cy, 0.0)) for p in points) / len(points)
