"""报告配图（可视化）：把 deaths.csv 的结论画成一组图。

产出（供 report.md 引用）：
* fault_distribution.png  —— 责任分布（position/preaim/aim/info/none）
* preaim_distribution.png —— 预瞄偏差直方图（你 vs 对手）
* duel_damage.png         —— aim 死亡的我 vs 他伤害对比
* death_scatter.png       —— 死亡位置散点（按 fault 着色）
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import plot


def generate(frame: pl.DataFrame, out_dir: Path, *, prefix: str = "") -> dict[str, str]:
    """生成一组配图，返回 {图名: 相对路径}。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = prefix
    made: dict[str, str] = {}

    # 1) fault 责任分布
    if "fault" in frame.columns and frame.height:
        fc = frame["fault"].value_counts().to_dicts()
        labels = [r["fault"] for r in fc]
        counts = [r["count"] for r in fc]
        f = plot.fault_distribution_chart(labels, counts, out_dir / f"{p}fault_distribution.png",
                                          title="责任分布（谁的锅）")
        made["fault_distribution"] = f.name

    # 2) 预瞄偏差分布
    if "my_aim_off_at_see_deg" in frame.columns:
        mine = [v for v in frame["my_aim_off_at_see_deg"].to_list() if v is not None]
        theirs = [v for v in (frame["killer_aim_off_deg"].to_list()
                              if "killer_aim_off_deg" in frame.columns else []) if v is not None]
        if mine:
            f = plot.preaim_distribution_chart(mine, theirs, out_dir / f"{p}preaim_distribution.png",
                                               title="预瞄偏差分布（你 vs 对手）")
            made["preaim_distribution"] = f.name

    # 3) 对枪伤害对比（aim 类死亡）
    if "fault" in frame.columns and "my_dmg" in frame.columns:
        aim = frame.filter(pl.col("fault") == "aim")
        if aim.height:
            ticks = aim["death_tick"].to_list()
            my_d = aim["my_dmg"].to_list()
            his_d = aim["his_dmg"].to_list() if "his_dmg" in frame.columns else [0] * len(ticks)
            f = plot.duel_damage_chart(ticks, my_d, his_d, out_dir / f"{p}duel_damage.png",
                                       title="枪法问题：每次对枪的伤害对比")
            made["duel_damage"] = f.name

    # 4) 死亡位置散点
    if {"my_x", "my_y", "fault"} <= set(frame.columns) and frame.height:
        f = plot.death_scatter_chart(frame["my_x"].to_list(), frame["my_y"].to_list(),
                                     frame["fault"].to_list(), out_dir / f"{p}death_scatter.png",
                                     title="死亡位置分布（按责任着色）")
        made["death_scatter"] = f.name

    return made
