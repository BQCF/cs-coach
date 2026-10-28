"""对枪重建（N1）：从 shots / damages 重建每一次交火的「证据链」。

回答用户第三条诉求：「哪几次死亡是我站位没问题、先看到敌人，但枪法输了」。

三条关键口径（算错就全错，见计划 §5.1/§9）：
1. ``my_shots`` 用**交火窗口** ``[t_see_him, death]``，不是「死前 2s」——
   否则压枪 / 打别人的子弹都算进来。
2. ``my_hits`` 直接数 ``damages`` 行数，**不要**拿 shots 去匹配 damages
   （一枪多记录、穿透多目标）。
3. ``health_pre`` 是谁的血取决于 attacker：``attacker=我`` 的行是**他的**血，
   ``attacker=他`` 的行是**我的**血。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import polars as pl

from .config import RulesCfg
from .geometry import in_view

# fault_sub 取值
FAULT_SUBS = ("whiff", "low_dmg", "late", "wide", "exposed", "no_escape", "")


@dataclass
class DuelFacts:
    """一次交火的全部可核对事实。None = 数据缺失，不猜。"""

    my_aim_off_deg: float | None = None
    my_aim_off_at_see_deg: float | None = None
    killer_aim_off_deg: float | None = None
    my_pitch_err_deg: float | None = None
    first_fire: str = "none"            # me / him / tie / none
    fire_gap_ms: int | None = None      # 正 = 他先
    my_react_ms: int | None = None      # 他进 FOV -> 我首枪
    my_ttfd_ms: int | None = None       # 他进 FOV -> 我首次命中
    my_shots: int = 0
    my_hits: int = 0
    my_dmg: int = 0
    my_head_hits: int = 0
    his_shots: int = 0
    his_hits: int = 0
    his_dmg: int = 0
    my_hp_at_engage: int | None = None
    his_hp_at_engage: int | None = None
    clip_state: str = "ok"              # empty / reloading / ok
    t_see_him: int | None = None        # 他首次进我 FOV 的绝对 tick

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- 小工具


def _shots(df: pl.DataFrame, steamid: int | None, lo: int, hi: int) -> pl.DataFrame:
    if steamid is None or df is None or df.is_empty():
        return pl.DataFrame()
    if "steamid" not in df.columns or "tick" not in df.columns:
        return pl.DataFrame()
    return df.filter(
        (pl.col("steamid") == int(steamid))
        & (pl.col("tick") >= int(lo))
        & (pl.col("tick") <= int(hi))
    ).sort("tick")


def _damage(df: pl.DataFrame, attacker: int | None, victim: int | None, lo: int, hi: int) -> pl.DataFrame:
    if attacker is None or victim is None or df is None or df.is_empty():
        return pl.DataFrame()
    need = {"attacker_steamid", "victim_steamid", "tick"}
    if not need <= set(df.columns):
        return pl.DataFrame()
    return df.filter(
        (pl.col("attacker_steamid") == int(attacker))
        & (pl.col("victim_steamid") == int(victim))
        & (pl.col("tick") >= int(lo))
        & (pl.col("tick") <= int(hi))
    ).sort("tick")


def _track_at(track: pl.DataFrame | None, steamid: int | None, tick: int | None) -> dict | None:
    """track 里 <= tick 的最后一帧。"""
    if track is None or track.is_empty() or tick is None or steamid is None:
        return None
    part = track.filter((pl.col("steamid") == int(steamid)) & (pl.col("tick") <= int(tick))).sort("tick")
    if part.is_empty():
        return None
    return part.row(part.height - 1, named=True)


def _row_pos(row: dict | None) -> tuple[float, float, float] | None:
    if not row:
        return None
    try:
        return (float(row["x"]), float(row["y"]), float(row["z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _aim_off(my_pos, my_yaw: float | None, target_pos, target_yaw: float | None) -> float | None:
    """我的准星方向与「指向目标」的水平夹角（度）。"""
    if my_yaw is None or my_pos is None or target_pos is None:
        return None
    from .geometry import facing_gap_deg

    return facing_gap_deg(my_pos, target_pos, my_yaw)


def first_fov_tick(w: Any, geo: Any, cfg: RulesCfg) -> int | None:
    """他首次进入我 FOV（通视 且 在我视野角内）的 tick。逐 tick 扫交火窗口。"""
    if geo is None or w.killer_pos is None or w.killer_track is None or w.killer_track.is_empty():
        return None
    if w.me_track is None or w.me_track.is_empty():
        return None
    fov = float(getattr(getattr(geo, "cfg", None), "fov_deg", 90.0))
    lo = max(w.span_start, w.death_tick - w.ticks(cfg.gap_window_s))
    hi = w.death_tick
    me_idx = {
        int(r["tick"]): r
        for r in w.me_track.filter((pl.col("tick") >= lo) & (pl.col("tick") <= hi)).to_dicts()
    }
    k_idx = {
        int(r["tick"]): r
        for r in w.killer_track.filter((pl.col("tick") >= lo) & (pl.col("tick") <= hi)).to_dicts()
    }
    for tick in sorted(set(me_idx) & set(k_idx)):
        m, k = me_idx[tick], k_idx[tick]
        mp, kp = _row_pos(m), _row_pos(k)
        if mp is None or kp is None:
            continue
        my_yaw = float(m.get("yaw") or 0.0)
        if not in_view(mp, my_yaw, kp, fov):
            continue
        try:
            if geo.visible(mp, kp, tick=tick,
                           crouch_a=bool(m.get("is_crouched")), crouch_b=bool(k.get("is_crouched"))):
                return tick
        except Exception:  # noqa: BLE001
            return None
    return None


# --------------------------------------------------------------------------- 主入口


def build_duel(w: Any, geo: Any, cfg: RulesCfg) -> DuelFacts:
    """重建一个死亡点的对枪事实。"""
    d = DuelFacts()
    dt = int(w.death_tick)

    # 1) 交火起点：他首次进我 FOV，否则回退死前 duel_fallback_s
    t_see = first_fov_tick(w, geo, cfg)
    d.t_see_him = t_see
    lo = t_see if t_see is not None else dt - w.ticks(getattr(cfg, "duel_fallback_s", 2.0))

    # 2) 开火事实
    mine = _shots(w.shots, w.my_steamid, lo, dt)
    his = _shots(w.shots, w.killer_steamid, lo, dt)
    d.my_shots = mine.height
    d.his_shots = his.height

    if mine.height and his.height:
        mt, ht = int(mine["tick"].min()), int(his["tick"].min())
        d.first_fire = "me" if mt < ht else ("him" if ht < mt else "tie")
        d.fire_gap_ms = int(round((ht - mt) * 1000 / max(1, w.tick_rate)))
    elif mine.height:
        d.first_fire = "me"
    elif his.height:
        d.first_fire = "him"

    # 3) 命中事实（直接数 damages 行）
    to_him = _damage(w.damages, w.my_steamid, w.killer_steamid, lo, dt)
    to_me = _damage(w.damages, w.killer_steamid, w.my_steamid, lo, dt)
    d.my_hits = to_him.height
    d.his_hits = to_me.height
    d.my_dmg = int(to_him["dmg_health"].sum()) if to_him.height and "dmg_health" in to_him.columns else 0
    d.his_dmg = int(to_me["dmg_health"].sum()) if to_me.height and "dmg_health" in to_me.columns else 0
    if to_him.height and "hitgroup_name" in to_him.columns:
        d.my_head_hits = to_him.filter(pl.col("hitgroup_name") == "head").height

    # 4) 血量：health_pre 由 attacker 决定归属（我打他=他的血；他打我=我的血）
    if to_him.height and "health_pre" in to_him.columns:
        d.his_hp_at_engage = int(to_him["health_pre"][0])   # 第一次命中前他的血
    if to_me.height and "health_pre" in to_me.columns:
        d.my_hp_at_engage = int(to_me["health_pre"][0])     # 第一次挨打前我的血

    # 5) 反应 / 首次命中时间
    if mine.height and t_see is not None:
        d.my_react_ms = int(round((int(mine["tick"].min()) - t_see) * 1000 / max(1, w.tick_rate)))
    if to_him.height and t_see is not None:
        d.my_ttfd_ms = int(round((int(to_him["tick"].min()) - t_see) * 1000 / max(1, w.tick_rate)))

    # 6) 准星：死亡瞬间 + 他进 FOV 那刻（这才是「预瞄」该量的）
    d.my_aim_off_deg = _aim_off(w.my_pos, w.my_yaw, w.killer_pos, w.killer_yaw)
    if t_see is not None:
        me_row = _track_at(w.me_track, w.my_steamid, t_see)
        k_row = _track_at(w.killer_track, w.killer_steamid, t_see)
        mp = _row_pos(me_row) or w.my_pos
        kp = _row_pos(k_row) or w.killer_pos
        d.my_aim_off_at_see_deg = _aim_off(mp, (me_row or {}).get("yaw", w.my_yaw), kp,
                                          (k_row or {}).get("yaw", w.killer_yaw))
    # 他死前 0.3s 准星 vs 我
    k_pre = _track_at(w.killer_track, w.killer_steamid, dt - w.ticks(cfg.predictable_lookback_s))
    if k_pre is not None:
        kp = _row_pos(k_pre)
        d.killer_aim_off_deg = _aim_off(kp, k_pre.get("yaw"), w.my_pos, w.my_yaw)

    # 6.5) pitch 偏差（垂直预瞄，符号已验证）
    if w.killer_pos is not None:
        from .preaim import pitch_err_deg

        d.my_pitch_err_deg = pitch_err_deg(w.my_pos, w.my_pitch, w.killer_pos)

    # 7) 弹匣状态（独立列，不再当死因）
    d.clip_state = _clip_state(w, mine)
    return d


def _clip_state(w: Any, mine: pl.DataFrame) -> str:
    if mine.is_empty():
        return "ok"
    if "num_bullets_remaining" in mine.columns:
        vals = [v for v in mine["num_bullets_remaining"].to_list() if v is not None]
        if vals and min(vals) <= 0:
            return "empty"
    if "is_reloading" in mine.columns:
        if mine.filter(pl.col("is_reloading") == True).height:  # noqa: E712
            return "reloading"
    return "ok"
