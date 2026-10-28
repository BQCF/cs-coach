"""五个指标（文档 §4.4）+ 两个辅助量。

====================================  ==============================================
指标                                  定义
====================================  ==============================================
``exposure``                          能看见你眼睛的敌方进入点数
``crossfire``                         暴露进入点两两夹角 > 60° 的对数
``first_seen_gap_ms``                 **进入 FOV** 的时间差，正数 = 他先看见你
``predictable``                       他死前 0.3s 准星是否已经对着你（< 6°）
``escape_time``                       到最近掩体的秒数
``killer_lead_ms``（辅助）            他从遮挡后出现 -> 我死 的毫秒数
====================================  ==============================================

**为什么 ``first_seen_gap_ms`` 必须用 FOV 而不是 360° 通视**（评审 3.2 的核心）：

``is_visible`` 是 360° 几何通视，而人只有约 90° 有效视野。如果直接拿通视算
「谁先看见谁」，双方在同一条线段上同一 tick 同时"通视"，gap 恒等于 0 或失真。

正确做法是把「看见」定义为 **通视 且 在对方 FOV 内**：

* ``t_him`` = 首个（他能通视到我 **且** 我在他朝向前方 fov/2 内）的 tick
* ``t_me``  = 首个（我能通视到他 **且** 他在我朝向前方 fov/2 内）的 tick
* ``gap = (t_me - t_him) * 1000 / tickrate``，正数 = 他先看见我（吃亏）

两条轨迹 join 到同一 tick 网格、逐 tick 双向判（评审 3.4：不再用固定目标点）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import polars as pl

from .config import RulesCfg
from .deaths import DeathWindow, _flag, _row_xyz, killer_emerge_tick
from .geometry import facing_gap_deg, in_view
from .windows import frame_at

# 合理性区间（评审 4.3）：超出就写进 metric_notes 标「可疑」，别拿 bug 当结论
GAP_SANE = (-300.0, 400.0)  # ms
ESCAPE_SANE = 5.0  # s


@dataclass
class Metrics:
    exposure: int | None = None
    exposure_entries: list[str] = field(default_factory=list)
    crossfire: int | None = None
    first_seen_gap_ms: int | None = None
    predictable: bool | None = None
    escape_time: float | None = None
    killer_lead_ms: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exposure": self.exposure,
            "exposure_entries": ",".join(self.exposure_entries) if self.exposure_entries else None,
            "crossfire": self.crossfire,
            "first_seen_gap_ms": self.first_seen_gap_ms,
            "predictable": self.predictable,
            "escape_time": self.escape_time,
            "killer_lead_ms": self.killer_lead_ms,
            "metric_notes": " | ".join(self.notes) if self.notes else None,
        }


# --------------------------------------------------------------------------- 单指标


def _tick_map(track: pl.DataFrame | None, lo: int, hi: int) -> dict[int, dict]:
    """tick -> 行（dict），用于同一 tick 网格对齐双方轨迹。"""
    if track is None or track.is_empty():
        return {}
    part = track.filter((pl.col("tick") >= int(lo)) & (pl.col("tick") <= int(hi))).sort("tick")
    return {int(row["tick"]): row for row in part.to_dicts()}


def _yaw_of(row: dict | None) -> float | None:
    """取 yaw；缺失返回 None——不能悄悄按 0° 算，那会给出错误的 FOV 判定。"""
    if not row:
        return None
    value = row.get("yaw")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fov_visible(
    geo: Any,
    observer_pos: tuple,
    observer_yaw: float,
    target_pos: tuple,
    *,
    tick: int,
    crouch: bool,
    target_crouch: bool,
    fov_deg: float,
) -> bool:
    """通视 且 目标在观察者 FOV 内。"""
    if not in_view(observer_pos, observer_yaw, target_pos, fov_deg):
        return False
    try:
        return bool(
            geo.visible(
                observer_pos,
                target_pos,
                tick=tick,
                crouch_a=crouch,
                crouch_b=target_crouch,
            )
        )
    except Exception:  # noqa: BLE001 - 无网格 / 区间异常，降级
        return False


def first_seen_gap(w: DeathWindow, geo: Any, cfg: RulesCfg) -> int | None:
    """进入 FOV 的时间差（正数 = 他先看见我）。双方轨迹对齐到同一 tick。"""
    if w.killer_pos is None or geo is None:
        return None
    fov_deg = float(getattr(getattr(geo, "cfg", None), "fov_deg", 90.0))
    lo = max(w.span_start, w.death_tick - w.ticks(cfg.gap_window_s))
    hi = w.death_tick

    me_idx = _tick_map(w.me_track, lo, hi)
    killer_idx = _tick_map(w.killer_track, lo, hi) if w.killer_track is not None else {}
    ticks = sorted(set(me_idx) & set(killer_idx))
    if not ticks:
        return None

    t_me = t_him = None
    for tick in ticks:
        m, k = me_idx[tick], killer_idx[tick]
        my_pos, killer_pos = _row_xyz(m), _row_xyz(k)
        if my_pos is None or killer_pos is None:
            continue
        my_yaw = _yaw_of(m)
        k_yaw = _yaw_of(k)
        # 朝向缺失的 tick 判不了 FOV，跳过；t_him 看他的朝向、t_me 看我的朝向
        if t_him is None and k_yaw is not None and _fov_visible(
            geo,
            killer_pos,
            k_yaw,
            my_pos,
            tick=tick,
            crouch=_flag(k.get("is_crouched")),
            target_crouch=_flag(m.get("is_crouched")),
            fov_deg=fov_deg,
        ):
            t_him = tick
        if t_me is None and my_yaw is not None and _fov_visible(
            geo,
            my_pos,
            my_yaw,
            killer_pos,
            tick=tick,
            crouch=_flag(m.get("is_crouched")),
            target_crouch=_flag(k.get("is_crouched")),
            fov_deg=fov_deg,
        ):
            t_me = tick
        if t_me is not None and t_him is not None:
            break
    if t_me is None or t_him is None:
        return None
    return int(round((t_me - t_him) * 1000.0 / max(1, w.tick_rate)))


def predictable(w: DeathWindow, cfg: RulesCfg) -> bool | None:
    """他死前 ``predictable_lookback_s`` 的准星是否已经指向我。"""
    if w.killer_track is None or w.killer_steamid is None or w.killer_track.is_empty():
        if w.killer_pos is None or w.killer_yaw is None:
            return None
        gap = facing_gap_deg(w.killer_pos, w.my_pos, w.killer_yaw)
        return bool(gap < cfg.predictable_angle_deg)
    tick = w.death_tick - w.ticks(cfg.predictable_lookback_s)
    row = frame_at(w.killer_track, w.killer_steamid, tick)
    pos = _row_xyz(row)
    yaw = _yaw_of(row)
    if pos is None or yaw is None:
        return None
    return bool(facing_gap_deg(pos, w.my_pos, yaw) < cfg.predictable_angle_deg)


def killer_lead_ms(w: DeathWindow, geo: Any, cfg: RulesCfg) -> int | None:
    """他从遮挡后出现到我死亡之间的毫秒数。"""
    emerge = killer_emerge_tick(w, geo, cfg)
    if emerge is None:
        return None
    return int(round((w.death_tick - emerge) * 1000.0 / max(1, w.tick_rate)))


# --------------------------------------------------------------------------- 汇总


def compute(w: DeathWindow, geo: Any, cfg: RulesCfg) -> Metrics:
    """算一个死亡点的全部指标；没有几何时返回全 null + 说明。"""
    if geo is None:
        return Metrics(notes=["geo_missing：未加载碰撞网格，几何指标为 null"])

    exposure, entries = geo.exposure(w.my_pos, tick=w.death_tick, crouched=w.my_crouched)
    crossfire = geo.crossfire(w.my_pos, tick=w.death_tick, crouched=w.my_crouched)
    gap = first_seen_gap(w, geo, cfg)
    pred = predictable(w, cfg)
    escape = geo.escape_time(w.my_pos)
    lead = killer_lead_ms(w, geo, cfg)

    notes: list[str] = []
    if exposure is None:
        notes.append("entry_points 未配齐（<门禁），exposure/crossfire 为 null")
    elif exposure > len(geo.entry_points(w.my_pos)):
        notes.append("exposure 超过进入点总数（几何实现异常）")

    if gap is not None and not (GAP_SANE[0] <= gap <= GAP_SANE[1]):
        notes.append(f"first_seen_gap_ms={gap} 超出合理区间 {GAP_SANE}，可疑")

    if escape is not None and escape > ESCAPE_SANE:
        notes.append(f"escape_time={escape}s > {ESCAPE_SANE}s，掩体点可能没配对")

    if escape is None:
        notes.append("cover_points 未配齐，escape_time 为 null")
    if gap is None:
        notes.append("窗口内未见双向 FOV 通视，gap 为 null")

    return Metrics(
        exposure=exposure,
        exposure_entries=entries,
        crossfire=crossfire,
        first_seen_gap_ms=gap,
        predictable=pred,
        escape_time=escape,
        killer_lead_ms=lead,
        notes=notes,
    )


def compute_many(records: Sequence[Any], geo: Any, cfg: RulesCfg) -> None:
    """就地给 DeathRecord 列表补 metrics（CLI 用）。"""
    for rec in records:
        rec.metrics = compute(rec.window, geo, cfg)
