"""职业基准（M4）：落位采集 + 落位网格 + 分位。

* 采集：``collect_pro_positions`` 从职业 demo 读落位点，混杂变量（回合阶段 /
  经济档 / 人数状态）必须传齐（评审 4.4），否则拿 5v5 的站位对比职业 1v3 的
  站位，分位没有意义；
* 角色判定：按「回合」判 8 角色，见 :mod:`csc.bench_roles`（整场固定角色的
  旧简化版已删——普通玩家每回合角色可不同）。
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

from .config import BenchCfg, Vec3


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame()


# --------------------------------------------------------------------------- 网格


def world_to_grid(pos: Vec3, grid: float) -> tuple[int, int]:
    """世界坐标 -> 网格单元 (cx, cy)。"""
    return (int(pos[0] // grid), int(pos[1] // grid))


def grid_frame(points: Sequence[Vec3], grid: float) -> pl.DataFrame:
    """落位点 -> 网格频次表。"""
    rows = [{"cx": int(p[0] // grid), "cy": int(p[1] // grid)} for p in points]
    if not rows:
        return pl.DataFrame(schema={"cx": pl.Int64, "cy": pl.Int64, "count": pl.Int64, "pct": pl.Float64})
    df = pl.DataFrame(rows).group_by(["cx", "cy"]).agg(pl.len().alias("count"))
    return df.with_columns((pl.col("count") / df["count"].sum()).alias("pct"))


def percentile_for(pos: Vec3, grid: pl.DataFrame, grid_size: float) -> float | None:
    """死亡点所在网格的落位频次分位（0~1）；不在表里 = 0。"""
    if grid.is_empty():
        return None
    cx, cy = world_to_grid(pos, grid_size)
    hit = grid.filter((pl.col("cx") == cx) & (pl.col("cy") == cy))
    if hit.is_empty():
        return 0.0
    return float(hit["pct"][0])


# --------------------------------------------------------------------------- 采集


def collect_pro_positions(
    ctx: Any,
    *,
    round_phase: str = "early",  # early / mid / late
    buy_type: str | None = None,  # pistol / eco / force / full
    n_alive: int | None = None,  # 5 = 5v5
    side: str = "counter-terrorist",  # 默认取防守方落位
    holding_only: bool = True,  # 只统计静止架枪点（默认）
    holding_speed: float = 80.0,  # 架枪速度阈值 u/s（走路≈250）
    cfg: BenchCfg | None = None,
) -> pl.DataFrame:
    """从职业 demo 读落位点（供网格化热图 + 分位）。混杂变量必须传齐。

    只取 ``round_phase`` 对应时段的存活玩家位置：

    * ``early`` = freeze_end 后前 60s（常规站位/默认展开）
    * ``mid``   = 60~120s，``late`` = 120s 后（残局）
    """
    from . import demo_io  # 局部导入，避免循环

    cfg = cfg or BenchCfg()
    phases = {"early": (0, cfg.role_window_s), "mid": (cfg.role_window_s, 2 * cfg.role_window_s),
              "late": (2 * cfg.role_window_s, 6 * cfg.role_window_s)}
    if round_phase not in phases:
        raise ValueError(f"round_phase 必须是 {list(phases)}，实际 {round_phase!r}")
    phase_start_s, phase_end_s = phases[round_phase]

    demo = demo_io.open_demo(ctx)
    demo.load("rounds", "round_economy")
    rounds = demo.rounds.to_dicts()
    econ = demo.round_economy.to_dicts()

    econ_by_round = {(e["round_num"], e["side"]): e for e in econ}
    valid_rounds = []
    for rd in rounds:
        e = econ_by_round.get((rd["round_num"], side))
        if e is None:
            continue
        if buy_type and e["buy_type"] != buy_type:
            continue
        if n_alive is not None and e["n_players"] != n_alive:
            continue
        valid_rounds.append({**rd, "buy_type": e["buy_type"], "n_players": e["n_players"]})
    if not valid_rounds:
        return _empty_frame()

    tr = int(getattr(demo, "tick_rate", 64) or 64)
    from .config import load_settings  # 局部导入
    from .bench_roles import classify_round_role  # 局部导入避免循环

    settings = load_settings()
    map_name = getattr(ctx, "map_name", "") or ""
    spec = settings.map_spec(map_name)

    all_rows: list[dict] = []
    # 逐回合取快照 + 逐回合判角色（普通玩家每回合角色可不同）
    for rd in valid_rounds:
        base = int(rd["freeze_end_tick"] or rd["start_tick"])
        span = (base + int(phase_start_s * tr), base + int(phase_end_s * tr))
        snap = demo_io.snapshot_windows(demo, [span])
        if snap.is_empty():
            continue
        snap = snap.filter((pl.col("side") == side) & (pl.col("health") > 0))
        if holding_only and "velocity" in snap.columns:
            snap = snap.filter(pl.col("velocity") < holding_speed)
        if snap.is_empty():
            continue

        _extra = [c for c in ("velocity", "equipment_value", "primary_weapon", "active_weapon", "has_bomb") if c in snap.columns]
        _cols = ["tick", "steamid", "name", "x", "y", "z"] + _extra
        frame = snap.select(_cols)

        # 本回合每个玩家的角色
        role_map: dict[Any, str] = {}
        equip_map: dict[Any, float] = {}
        pw_map: dict[Any, str | None] = {}
        bomb_map: dict[Any, bool] = {}
        for sid in frame["steamid"].unique().to_list():
            ptrack = snap.filter(pl.col("steamid") == sid).select(["tick", "x", "y", "z"])
            sub = frame.filter(pl.col("steamid") == sid)
            has_bomb = bool(sub["has_bomb"][0]) if "has_bomb" in sub.columns and sub.height else False
            pw = sub["primary_weapon"][0] if "primary_weapon" in sub.columns and sub.height else None
            role_map[sid] = classify_round_role(ptrack, spec, cfg, side=side, has_bomb=has_bomb, primary_weapon=pw)
            equip_map[sid] = float(sub["equipment_value"][0]) if "equipment_value" in sub.columns and sub.height else 0.0
            pw_map[sid] = pw
            bomb_map[sid] = has_bomb

        for r in frame.iter_rows(named=True):
            pos = (float(r["x"]), float(r["y"]), float(r["z"]))
            sid = r["steamid"]
            all_rows.append({
                "tick": int(r["tick"]),
                "map_name": map_name,
                "round_num": rd["round_num"],
                "steamid": sid,
                "name": r["name"],
                "x": pos[0], "y": pos[1], "z": pos[2],
                "velocity": float(r.get("velocity") or 0.0),
                "equipment_value": equip_map.get(sid, 0.0),
                "primary_weapon": (pw_map.get(sid) or ""),
                "has_bomb": bool(bomb_map.get(sid, False)),
                "role": role_map.get(sid, "unknown"),
                "region": spec.region_of(pos),
                "round_phase": round_phase,
                "buy_type": buy_type or "any",
            })
    if not all_rows:
        return _empty_frame()
    _schema = {
        "tick": pl.Int64, "map_name": pl.Utf8, "round_num": pl.Int64, "steamid": pl.UInt64, "name": pl.Utf8,
        "x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "velocity": pl.Float64,
        "equipment_value": pl.Float64, "primary_weapon": pl.Utf8, "has_bomb": pl.Boolean,
        "role": pl.Utf8, "region": pl.Utf8, "round_phase": pl.Utf8, "buy_type": pl.Utf8,
    }
    return pl.DataFrame(all_rows, schema=_schema)
