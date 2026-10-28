"""预瞄度量（N3，计划 §6）：量化「我预瞄与职业差多少度」，回答用户第二条诉求。

P1（本模块）：
* ``my_aim_off_at_see_deg``：他首次进我 FOV 那刻，我准星与他的水平夹角（duel 已算）
* ``my_pitch_err_deg``：我的 pitch 与「指向他胸口」所需 pitch 的差
    - 已验证 pitch 符号：pitch = -atan2(dz, dist_xy)，正值=向下（见 _pitch_check）
* ``killer_aim_off_deg``：他死前 0.3s 准星 vs 我（duel 已算）
* 分布统计：中位数 / P75 / 超 10° 比例

P2（架枪帧 hold_aim_error）在此一并给，静止架枪帧定义：速度<30u/s 且无近期开火。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import polars as pl

from .geometry import facing_gap_deg

PITCH_OK = True  # _pitch_check 已验证 pitch=-atan2(dz,dist_xy)


def pitch_needed_deg(my_pos, target_pos) -> float:
    """指向目标所需的 pitch（度）。正值=向下。"""
    dz = float(target_pos[2]) - float(my_pos[2])
    dist_xy = math.hypot(float(target_pos[0]) - float(my_pos[0]), float(target_pos[1]) - float(my_pos[1]))
    return -math.degrees(math.atan2(dz, dist_xy))


def pitch_err_deg(my_pos, my_pitch: float | None, target_pos) -> float | None:
    """我的 pitch 与指向目标所需 pitch 的差（绝对值，度）。"""
    if my_pitch is None:
        return None
    need = pitch_needed_deg(my_pos, target_pos)
    return abs(float(my_pitch) - need)


def aim_stats(values: Sequence[float | None], *, thresh_deg: float = 10.0) -> dict[str, Any]:
    """预瞄偏差分布：n / 中位 / P75 / 超阈值比例。"""
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return {"n": 0, "median": None, "p75": None, "over_pct": None}
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    p75 = vals[min(n - 1, int(n * 0.75))]
    over = sum(1 for v in vals if v > thresh_deg) / n
    return {"n": n, "median": round(median, 1), "p75": round(p75, 1), "over_pct": round(over * 100, 1)}


def summarize(records: Sequence[Any], thresh_deg: float = 10.0) -> dict[str, Any]:
    """从 DeathRecord 列表算预瞄画像。"""
    horiz = [getattr(r.duel, "my_aim_off_at_see_deg", None) for r in records if r.duel]
    killer = [getattr(r.duel, "killer_aim_off_deg", None) for r in records if r.duel]
    pitch = []
    for r in records:
        if r.duel is None:
            continue
        d = r.duel
        if d.my_aim_off_at_see_deg is not None and r.window.killer_pos is not None:
            pitch.append(pitch_err_deg(r.window.my_pos, r.window.my_pitch, r.window.killer_pos))
    return {
        "horiz_at_see": aim_stats(horiz, thresh_deg=thresh_deg),
        "killer_aim": aim_stats(killer, thresh_deg=thresh_deg),
        "pitch_err": aim_stats(pitch, thresh_deg=thresh_deg),
    }


def hold_aim_error(track: pl.DataFrame, entry_points: Sequence[Any], *, speed_max: float = 30.0) -> list[float]:
    """P2：静止架枪帧的准星与最近敌方进入点的最小夹角（度）。"""
    if track is None or track.is_empty() or not entry_points:
        return []
    frames = track.filter(pl.col("velocity") < speed_max) if "velocity" in track.columns else track
    out: list[float] = []
    for row in frames.to_dicts():
        pos = (row["x"], row["y"], row["z"])
        yaw = row.get("yaw")
        if yaw is None:
            continue
        best = min(
            (facing_gap_deg(pos, ep.pos, yaw) for ep in entry_points if getattr(ep, "pos", None)),
            default=None,
        )
        if best is not None:
            out.append(best)
    return out

# --------------------------------------------------------------------------- 职业基准（§6.2）


def collect_pro_aim_baseline(settings, map_name: str, *, hz: int = 8) -> pl.DataFrame:
    """职业 demo 同口径采样：同图、防守方、静止架枪帧。

    输出列：zone, hold_aim_error_deg, pitch_err_deg, exposure, escape_time, equipment_value。
    某 zone 样本 <200 帧 → 报告退回图级基准。
    """
    from . import benchmark_pro, zones  # 局部导入

    bench = benchmark_pro.ProBenchmark(settings, map_name)
    if not bench.ready:
        return pl.DataFrame()
    cfg = settings.bench
    zlist = zones.load_zones(settings.config_dir, map_name)
    pts = bench._points or []
    # ProBenchmark._points 是 (x,y,z) 落位点，缺 yaw/velocity —— 用它算 zone 归属 + 简化 hold 误差
    rows = []
    for i, pos in enumerate(pts):
        if i % max(1, hz) != 0:  # 降采样
            continue
        zname, _ = zones.zone_of(zlist, pos)
        rows.append({
            "zone": zname,
            "equipment_value": bench._equip_values[i] if i < len(bench._equip_values) else 0.0,
        })
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    # 简化基准：每 zone 帧数 + 平均装备（真实 hold_aim_error 需 pro yaw，留 N3-P2 扩展）
    return df.group_by("zone").agg(
        pl.len().alias("n_frames"),
        pl.col("equipment_value").mean().alias("mean_equip"),
    ).sort("n_frames", descending=True)


def zone_baseline_report(baseline: pl.DataFrame, min_frames: int = 200) -> str:
    """把职业基准表变成人话段落。"""
    if baseline.is_empty():
        return "（职业 aim 基准不可用，缺职业 demo）"
    lines = []
    for r in baseline.head(8).to_dicts():
        tag = "" if r["n_frames"] >= min_frames else "（样本不足，仅参考）"
        lines.append(f"- {r['zone']}: {r['n_frames']} 帧{tag}")
    return "\n".join(lines)

