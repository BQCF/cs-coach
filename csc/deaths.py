"""死亡点提取与归因。

归因规则表（文档 §4.2，按顺序匹配，命中即停）：

===========  ==================  ==========================================
优先级        标签                判定
===========  ==================  ==========================================
1            被闪                死亡前 2s 内被闪且 duration >= 1.2s
2            背身/侧身           击杀者方向与死者朝向夹角 > 90°
3            打空/换弹           死亡前 0.6s 开空弹匣，或最近一次开火 < 0.15s
4            移动中对枪          死亡前 0.5s 平均速度 > 60 u/s
5            被拉死              击杀者从遮挡后出现，且死者几乎静止
6            架死                双方提前 0.5s 就互相可见
7            信息缺失硬拉        主动拉 + 无队友视野证据（**本版未实现，降级**）
8            其他/对枪           兜底
===========  ==================  ==========================================

规则 1–4 只需要 demo 表；5–6 需要几何（``GeoContext``）；7 是 TODO。
**没有几何时不会报错，只会退化成 1–4/8 的标签**——这是 M1 能在没下地图资源时先跑通的关键。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import polars as pl

from .config import RulesCfg, Vec3
from .geometry import facing_gap_deg
from .windows import TRACK_COLUMNS, frame_at, slice_track

if TYPE_CHECKING:  # pragma: no cover
    from .demo_io import DemoContext
    from .geometry import GeoContext
    from .metrics import Metrics
    from .windows import Span

LABELS: tuple[str, ...] = (
    "被闪",
    "背身/侧身",
    "打空/换弹",
    "正面遭遇",
    "移动中对枪",
    "被拉死",
    "架死",
    "信息缺失硬拉",
    "其他/对枪",
)

INFO_MISSING_IMPLEMENTED = True  # 规则 7（信息缺失硬拉）已实现


# --------------------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class DeathWindow:
    """一个死亡点的全部输入。纯数据，手搓即可单测，不依赖 awpy。"""

    death_tick: int
    round_num: int | None
    tick_rate: int
    my_steamid: int
    my_name: str
    killer_steamid: int | None
    killer_name: str | None
    weapon: str
    headshot: bool
    my_pos: Vec3
    killer_pos: Vec3 | None
    my_yaw: float | None
    killer_yaw: float | None
    my_pitch: float | None
    killer_pitch: float | None
    my_speed: float
    killer_speed: float
    my_crouched: bool
    killer_crouched: bool
    me_track: pl.DataFrame = field(default_factory=pl.DataFrame)
    killer_track: pl.DataFrame | None = None
    blinds: pl.DataFrame = field(default_factory=pl.DataFrame)
    shots: pl.DataFrame = field(default_factory=pl.DataFrame)
    smokes: pl.DataFrame = field(default_factory=pl.DataFrame)
    damages: pl.DataFrame = field(default_factory=pl.DataFrame)
    span_start: int = 0
    span_end: int = 0
    geo_missing: bool = False

    # -- 时间换算 -----------------------------------------------------------
    def ticks(self, seconds: float) -> int:
        return int(round(seconds * self.tick_rate))

    @property
    def death_s(self) -> float:
        return self.death_tick / float(self.tick_rate)


@dataclass
class Verdict:
    label: str
    confidence: float
    rule: str
    note: str = ""


@dataclass
class DeathRecord:
    window: DeathWindow
    verdict: Verdict
    metrics: "Metrics | None" = None
    duel: Any = None      # DuelFacts（duel.build_duel）
    fault: Any = None     # FaultVerdict（fault.classify_fault）
    zone: str | None = None
    zone_dist: float | None = None


# --------------------------------------------------------------------------- 小工具


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _opt_num(value: Any) -> float | None:
    """数值化但保留 None：朝向/俯仰缺失时应降级 null，而不是悄悄按 0° 算。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> bool:
    return bool(value) if value is not None else False


def _row_xyz(row: dict | None, prefix: str = "") -> Vec3 | None:
    if not row:
        return None
    try:
        return (float(row[f"{prefix}x"]), float(row[f"{prefix}y"]), float(row[f"{prefix}z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _ranged(df: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    if not isinstance(df, pl.DataFrame) or df.is_empty() or "tick" not in df.columns:
        return pl.DataFrame()
    return df.filter((pl.col("tick") >= int(start)) & (pl.col("tick") <= int(end)))


def _avg_speed(track: pl.DataFrame, start_tick: int, end_tick: int, tick_rate: int = 64) -> float:
    """区间内平均速度；优先用 velocity 列，缺失时用位置差兜底（u/s）。"""
    if track is None or track.is_empty():
        return 0.0
    part = _ranged(track, start_tick, end_tick)
    if part.is_empty():
        return 0.0
    if "velocity" in part.columns:
        values = [v for v in part["velocity"].to_list() if v is not None]
        if values:
            return float(sum(values)) / len(values)
    rows = part.sort("tick").to_dicts()
    if len(rows) < 2:
        return 0.0
    first, last = rows[0], rows[-1]
    dt_s = (float(last["tick"]) - float(first["tick"])) / max(1, int(tick_rate))
    if dt_s <= 0:
        return 0.0
    dx = _num(last.get("x")) - _num(first.get("x"))
    dy = _num(last.get("y")) - _num(first.get("y"))
    return math.hypot(dx, dy) / dt_s


def _round_for_tick(rounds: pl.DataFrame, tick: int) -> int | None:
    if not isinstance(rounds, pl.DataFrame) or rounds.is_empty() or "round_num" not in rounds.columns:
        return None
    fallback: tuple[int, int] | None = None  # 回合间隙（end 与下一 start 之间）时回退到最近开始的回合
    for row in rounds.to_dicts():
        start = row.get("start_tick")
        end = row.get("end_tick") or row.get("official_end_tick")
        if start is None:
            continue
        if int(start) <= int(tick) and (end is None or int(tick) <= int(end)):
            return int(row["round_num"])
        if int(start) <= int(tick) and (fallback is None or int(start) > fallback[0]):
            fallback = (int(start), int(row["round_num"]))
    return fallback[1] if fallback else None


# --------------------------------------------------------------------------- 提取


def extract_deaths(kills: pl.DataFrame, my_steamid: int) -> pl.DataFrame:
    """筛出「我」的死亡事件。"""
    if not isinstance(kills, pl.DataFrame) or kills.is_empty() or "victim_steamid" not in kills.columns:
        return pl.DataFrame()
    return kills.filter(pl.col("victim_steamid") == int(my_steamid)).sort("tick")


def build_death_windows(
    ctx: "DemoContext",
    my_steamid: int,
    spans: Sequence["Span"],
    snap: pl.DataFrame,
    *,
    my_name: str | None = None,
) -> list[DeathWindow]:
    """把 kills 行 + 轨迹切片组装成 DeathWindow 列表。"""
    kills = ctx.table("kills")
    deaths = extract_deaths(kills, my_steamid)
    if deaths.is_empty():
        return []

    blinds_all = ctx.table("blinds")
    shots_all = ctx.table("shots")
    smokes_all = ctx.table("smokes")
    damages_all = ctx.table("damages")
    rounds_all = ctx.table("rounds")
    frames = ctx.death_frames

    span_list = list(spans)
    out: list[DeathWindow] = []
    for row in deaths.to_dicts():
        death_tick = int(row["tick"])
        span = next((s for s in span_list if s.start_tick <= death_tick <= s.end_tick), None)
        start = span.start_tick if span else max(0, death_tick - 3 * ctx.tick_rate)
        end = span.end_tick if span else death_tick + ctx.tick_rate

        killer_steamid = row.get("attacker_steamid")
        killer_steamid = int(killer_steamid) if killer_steamid not in (None, 0) else None

        me_frame = _frame_from(frames, my_steamid, death_tick)
        me_pos = _row_xyz(me_frame) or _row_xyz(snap_row_at(snap, my_steamid, death_tick))
        if me_pos is None:
            continue  # 位置都取不到，跳过（会体现在 skipped 计数里）

        killer_pos = None
        killer_frame = None
        if killer_steamid is not None:
            killer_frame = _frame_from(frames, killer_steamid, death_tick)
            killer_pos = _row_xyz(killer_frame) or _row_xyz(snap_row_at(snap, killer_steamid, death_tick))
            if killer_pos is None:
                killer_pos = _row_xyz(row, "attacker_")

        me_track = slice_track(snap, my_steamid, start, end)
        killer_track = slice_track(snap, killer_steamid, start, end) if killer_steamid else None

        # 死亡瞬间的姿态：优先用 death_frames（精确到死亡 tick），退化到窗口轨迹最后一帧
        me_src = me_frame if me_frame is not None else _last_row(me_track)
        killer_src = killer_frame if killer_frame is not None else _last_row(killer_track)
        my_yaw = _opt_num(me_src.get("yaw") if me_src else None)
        killer_yaw = _opt_num(killer_src.get("yaw") if killer_src else None)

        out.append(
            DeathWindow(
                death_tick=death_tick,
                round_num=_round_for_tick(rounds_all, death_tick),
                tick_rate=ctx.tick_rate,
                my_steamid=int(my_steamid),
                my_name=str(row.get("victim_name") or my_name or ""),
                killer_steamid=killer_steamid,
                killer_name=(str(row.get("attacker_name")) if row.get("attacker_name") else None),
                weapon=str(row.get("weapon") or "unknown"),
                headshot=_flag(row.get("headshot")),
                my_pos=me_pos,
                killer_pos=killer_pos,
                my_yaw=my_yaw,
                killer_yaw=killer_yaw,
                my_pitch=_opt_num(me_src.get("pitch") if me_src else None),
                killer_pitch=_opt_num(killer_src.get("pitch") if killer_src else None),
                my_speed=_avg_speed(me_track, death_tick - ctx.tick_rate, death_tick, ctx.tick_rate),
                killer_speed=(
                    _avg_speed(killer_track, death_tick - ctx.tick_rate, death_tick, ctx.tick_rate)
                    if killer_track is not None
                    else 0.0
                ),
                my_crouched=_flag(me_src.get("is_crouched") if me_src else None),
                killer_crouched=_flag(killer_src.get("is_crouched") if killer_src else None),
                me_track=me_track,
                killer_track=killer_track,
                blinds=_victim_events(blinds_all, my_steamid, start, end),
                shots=_ranged(shots_all, start, end),
                smokes=smokes_all,
                damages=_ranged(damages_all, start, end),
                span_start=start,
                span_end=end,
            )
        )
    return out


def _victim_events(df: pl.DataFrame, steamid: int, start: int, end: int) -> pl.DataFrame:
    part = _ranged(df, start, end)
    if part.is_empty() or "victim_steamid" not in part.columns:
        return pl.DataFrame()
    return part.filter(pl.col("victim_steamid") == int(steamid))


def _mine_shots(w: DeathWindow, start: int, end: int) -> pl.DataFrame:
    """窗口内「我」的开火事件。"""
    part = _ranged(w.shots, start, end)
    if part.is_empty() or "steamid" not in part.columns:
        return pl.DataFrame()
    return part.filter(pl.col("steamid") == int(w.my_steamid))


def _frame_from(df: pl.DataFrame, steamid: int, tick: int) -> dict | None:
    if not isinstance(df, pl.DataFrame) or df.is_empty() or "steamid" not in df.columns:
        return None
    part = df.filter(pl.col("steamid") == int(steamid))
    if "tick" in part.columns:
        part = part.filter(pl.col("tick") <= int(tick))
    if part.is_empty():
        return None
    if "tick" in part.columns:
        part = part.sort("tick")
    return part.row(part.height - 1, named=True)


def snap_row_at(snap: pl.DataFrame, steamid: int, tick: int) -> dict | None:
    return frame_at(snap, steamid, tick)


def _last_row(track: pl.DataFrame | None) -> dict | None:
    if track is None or track.is_empty():
        return None
    ordered = track.sort("tick")
    return ordered.row(ordered.height - 1, named=True)


def _pos_at(track: pl.DataFrame | None, steamid: int | None, tick: int) -> Vec3 | None:
    """某玩家在某 tick（或之前最后一帧）的位置。"""
    if track is None or steamid is None:
        return None
    return _row_xyz(frame_at(track, steamid, tick))


# --------------------------------------------------------------------------- 归因规则


RuleFn = Callable[[DeathWindow, RulesCfg, "GeoContext | None"], "Verdict | None"]


def rule_blind(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    lookback = w.death_tick - w.ticks(cfg.blind_lookback_s)
    rows = _ranged(w.blinds, lookback, w.death_tick)
    if rows.is_empty() or "duration" not in rows.columns:
        return None
    durations = [_num(v) for v in rows["duration"].to_list() if v is not None]
    longest = max(durations) if durations else 0.0
    if longest >= cfg.blind_min_duration_s:
        return Verdict("被闪", min(1.0, 0.5 + longest / 4.0), "blind", f"致盲 {longest:.2f}s")
    return None


def rule_behind(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    if w.killer_pos is None or w.my_yaw is None:
        return None
    gap = facing_gap_deg(w.my_pos, w.killer_pos, w.my_yaw)
    if gap > cfg.behind_angle_deg:
        return Verdict("背身/侧身", 0.7, "behind", f"击杀者在朝向 {gap:.0f}° 方向")
    return None


def rule_empty_clip(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    window = w.death_tick - w.ticks(cfg.empty_clip_window_s)
    mine = _mine_shots(w, window, w.death_tick)
    if mine.is_empty():
        return None
    if "num_bullets_remaining" in mine.columns:
        empties = [v for v in mine["num_bullets_remaining"].to_list() if v is not None]
        if empties and min(empties) <= 0:
            return Verdict("打空/换弹", 0.65, "empty_clip", "弹匣打空")
    last_tick = int(mine["tick"].max()) if "tick" in mine.columns else 0
    if last_tick >= w.death_tick - w.ticks(cfg.near_miss_window_s):
        return Verdict("打空/换弹", 0.5, "empty_clip", "死前 0.15s 内开过火")
    return None



def rule_encounter(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    """正面遭遇：双方 FOV 内、先手差极小（<60ms）的**站住**正面对枪。

    移动门控同 rule_holding：跑动中的对枪归「移动中对枪」（encounter 在 RULES
    里排在 moving 前面，不加门控会把移动中对枪全部吞掉，eval recall=0）。
    """
    if geo is None or w.killer_pos is None:
        return None
    if _moving_at_death(w, cfg):
        return None  # 跑动中死亡归「移动中对枪」，不算正面遭遇
    # 需要 first_seen_gap：双方几乎同时看到对方
    try:
        from .metrics import first_seen_gap  # 局部导入避免循环

        gap = first_seen_gap(w, geo, cfg)
    except Exception:  # noqa: BLE001
        return None
    if gap is not None and abs(gap) < 60:
        return Verdict("正面遭遇", 0.55, "encounter", f"正面遭遇，先手差 {gap}ms")
    return None


def rule_moving(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    if _moving_at_death(w, cfg):
        speed = _avg_speed(w.me_track, w.death_tick - w.ticks(cfg.moving_lookback_s), w.death_tick, w.tick_rate)
        return Verdict("移动中对枪", 0.6, "moving", f"死前 0.5s 均速 {speed:.0f} u/s")
    return None


def rule_peeked(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    """被拉死：他从遮挡后出现的那一刻，我几乎静止。"""
    if geo is None or w.killer_pos is None:
        return None
    emerge = killer_emerge_tick(w, geo, cfg)
    if emerge is None:
        return None
    speed = _avg_speed(w.me_track, emerge - w.ticks(cfg.peek_speed_lookback_s), emerge, w.tick_rate)
    if speed > cfg.peek_speed_max:
        return None
    return Verdict("被拉死", 0.6, "peeked", f"他于 {emerge} tick 从遮挡后出现，我当时 {speed:.0f} u/s")


def _moving_at_death(w: DeathWindow, cfg: RulesCfg) -> bool:
    """死亡前 ``moving_lookback_s`` 均速是否超过「移动中对枪」阈值。"""
    window = w.death_tick - w.ticks(cfg.moving_lookback_s)
    return _avg_speed(w.me_track, window, w.death_tick, w.tick_rate) > cfg.moving_speed


def rule_holding(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    """架死：双方在死亡前相当久就已经互相可见，且我是站住架枪的。

    评审 3.3：不能用死亡时的位置去判 0.5s 前的通视（击杀者这 0.5s 可能移动过）。
    改成从轨迹里取 probe 时刻双方的实际位置，取不到就返回 None。

    移动门控：跑动中的死亡不该算「架死」——否则会把「移动中对枪」吞掉
    （eval 实测：5 个人工标注的移动中对枪被 holding/encounter 抢走）。
    """
    if geo is None or w.killer_pos is None:
        return None
    if _moving_at_death(w, cfg):
        return None  # 跑动中死亡归「移动中对枪」，不是架死
    probe = w.death_tick - w.ticks(cfg.contact_time_s)
    # 评审 3.3：双方位置都必须取 probe 时刻的轨迹值；取不到就返回 None，
    # 绝不回退死亡时位置（那是拿未来的位置判过去的通视）。
    if w.killer_track is None:
        return None
    me_at = _pos_at(w.me_track, w.my_steamid, probe)
    k_at = _pos_at(w.killer_track, w.killer_steamid, probe)
    if me_at is None or k_at is None:
        return None
    me_row = frame_at(w.me_track, w.my_steamid, probe)
    k_row = frame_at(w.killer_track, w.killer_steamid, probe) if w.killer_track is not None else None
    try:
        a = geo.visible(
            me_at,
            k_at,
            tick=probe,
            crouch_a=_flag(me_row.get("is_crouched") if me_row else False),
            crouch_b=_flag(k_row.get("is_crouched") if k_row else False),
        )
        b = geo.visible(
            w.my_pos,
            w.killer_pos,
            tick=w.death_tick - 1,
            crouch_a=w.my_crouched,
            crouch_b=w.killer_crouched,
        )
    except Exception:  # noqa: BLE001 - 降级
        return None
    if a and b:
        return Verdict("架死", 0.55, "holding", f"死亡前 {cfg.contact_time_s}s 已双向通视")
    return None


def rule_info_missing(w: DeathWindow, cfg: RulesCfg, geo: "GeoContext | None") -> Verdict | None:
    """信息缺失硬拉（规则7）：我主动 peek/推进，但击杀者在我视野外/未暴露，
    没掌握他位置信息就硬拉被杀，且无队友开火掩护。"""
    if w.killer_pos is None:
        return None
    he_visible = False
    if geo is not None:
        try:
            he_visible = _killer_in_my_fov(w, geo)
        except Exception:  # noqa: BLE001
            he_visible = False
    if he_visible:
        return None  # 我看得到他，不算信息缺失

    my_speed = _avg_speed(w.me_track, w.death_tick - w.ticks(cfg.moving_lookback_s), w.death_tick, w.tick_rate)
    if my_speed <= cfg.peek_speed_max * 2:
        return None  # 没在推进

    if _has_teammate_cover(w):
        return None  # 有队友掩护

    return Verdict("信息缺失硬拉", 0.5, "info_missing",
                   f"推进中被杀且看不到击杀者（{my_speed:.0f} u/s，无队友掩护）")


def _killer_in_my_fov(w: DeathWindow, geo: "GeoContext") -> bool:
    """死亡时击杀者是否在我 FOV 内且通视（我"看得到"他）。"""
    from .geometry import in_view  # 局部导入避免循环

    fov = float(getattr(getattr(geo, "cfg", None), "fov_deg", 90.0))
    if not in_view(w.my_pos, w.my_yaw or 0.0, w.killer_pos, fov):
        return False
    return bool(geo.visible(w.my_pos, w.killer_pos, tick=w.death_tick,
                            crouch_a=w.my_crouched, crouch_b=w.killer_crouched))


def _has_teammate_cover(w: DeathWindow) -> bool:
    """该时段内是否有队友开火/伤害事件（掩护证据）。"""
    d = w.damages
    if d is None or d.is_empty() or "attacker_steamid" not in d.columns:
        return False
    lo = w.death_tick - w.ticks(1.0)
    try:
        recent = d.filter((pl.col("tick") >= lo) & (pl.col("tick") <= w.death_tick))
    except Exception:  # noqa: BLE001
        return False
    if recent.is_empty():
        return False
    teammates = recent.filter(
        (pl.col("attacker_steamid") != w.my_steamid)
        & (pl.col("attacker_steamid") != w.killer_steamid)
    )
    return teammates.height > 0


RULES: tuple[tuple[str, RuleFn, bool], ...] = (
    ("blind", rule_blind, False),
    ("behind", rule_behind, False),
    ("info_missing", rule_info_missing, True),
    ("peeked", rule_peeked, True),
    ("holding", rule_holding, True),
    ("encounter", rule_encounter, True),
    ("moving", rule_moving, False),
)


def attribute(
    w: DeathWindow,
    cfg: RulesCfg,
    geo: "GeoContext | None" = None,
) -> Verdict:
    """按优先级匹配，命中即停。没有 geo 时跳过需要几何的规则。"""
    for name, fn, needs_geo in RULES:
        if needs_geo and geo is None:
            continue
        verdict = fn(w, cfg, geo)
        if verdict is not None:
            return verdict
    note = "无几何，仅规则 1-4" if (geo is None and w.killer_pos is not None) else "兜底"
    return Verdict("其他/对枪", 0.2, "fallback", note)


def killer_emerge_tick(w: DeathWindow, geo: "GeoContext", cfg: RulesCfg) -> int | None:
    """击杀者「从遮挡后出现」的 tick：!visible(t-1) 且 visible(t)。

    直接顺序扫轨迹（每 tick 一帧），O(n)。别用 frame_at 逐 tick 回查——那是 O(n²)，
    而且本函数在 rule_peeked 与 killer_lead_ms 各调一次。
    """
    if w.killer_track is None or w.killer_track.is_empty():
        return None
    start = max(w.span_start, w.death_tick - w.ticks(cfg.peek_search_s))
    rows = w.killer_track.filter(pl.col("tick") >= int(start)).sort("tick").to_dicts()
    if not rows:
        return None
    me_eye_pos = w.my_pos
    prev_visible: bool | None = None
    for row in rows:
        pos = _row_xyz(row)
        if pos is None:
            continue
        try:
            now = geo.visible(
                pos,
                me_eye_pos,
                tick=int(row["tick"]),
                crouch_a=_flag(row.get("is_crouched")),
                crouch_b=w.my_crouched,
            )
        except Exception:  # noqa: BLE001
            return None
        if prev_visible is False and now:
            return int(row["tick"])
        prev_visible = now
    return None


# --------------------------------------------------------------------------- 输出表


FRAME_COLUMNS: tuple[str, ...] = (
    "death_tick",
    "death_time_s",
    "round_num",
    "victim_name",
    "killer_name",
    "killer_steamid",
    "weapon",
    "headshot",
    "my_x",
    "my_y",
    "my_z",
    "my_yaw",
    "my_speed",
    "killer_x",
    "killer_y",
    "killer_z",
    "label",
    "label_conf",
    "rule",
    "note",
)


def records_to_frame(records: Sequence[DeathRecord], *, demo: str | None = None) -> pl.DataFrame:
    """DeathRecord 列表 -> deaths.csv 用的扁平表（指标列由 metrics 追加）。

    ``demo``：demo 文件名，写进首列——eval 按 (demo, death_tick) 对齐标注，
    没有它跨 demo 同 tick 会静默错配。
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        w, v = rec.window, rec.verdict
        row: dict[str, Any] = {
            "death_tick": w.death_tick,
            "death_time_s": round(w.death_s, 2),
            "round_num": w.round_num,
            "victim_name": w.my_name,
            "killer_name": w.killer_name,
            "killer_steamid": w.killer_steamid,
            "weapon": w.weapon,
            "headshot": w.headshot,
            "my_x": round(w.my_pos[0], 1),
            "my_y": round(w.my_pos[1], 1),
            "my_z": round(w.my_pos[2], 1),
            "my_yaw": round(w.my_yaw, 1) if w.my_yaw is not None else None,
            "my_speed": round(w.my_speed, 1),
            "killer_x": round(w.killer_pos[0], 1) if w.killer_pos else None,
            "killer_y": round(w.killer_pos[1], 1) if w.killer_pos else None,
            "killer_z": round(w.killer_pos[2], 1) if w.killer_pos else None,
            "label": v.label,
            "mechanism": v.label,
            "label_conf": round(v.confidence, 2),
            "rule": v.rule,
            "note": v.note,
        }
        if rec.metrics is not None:
            row.update(rec.metrics.to_dict())
        if rec.duel is not None:
            row.update(rec.duel.to_dict())
        if rec.fault is not None:
            row.update(rec.fault.to_dict())
        if rec.zone is not None:
            row["zone"] = rec.zone
            row["zone_dist"] = round(rec.zone_dist, 1) if rec.zone_dist is not None else None
        if demo is not None:
            row["demo"] = demo
        rows.append(row)

    if not rows:
        schema = {col: pl.Utf8 for col in FRAME_COLUMNS}
        if demo is not None:
            schema = {"demo": pl.Utf8, **schema}
        return pl.DataFrame(schema=schema)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    if demo is not None:
        frame = frame.select(["demo"] + [c for c in frame.columns if c != "demo"])
    return frame
