"""findings + verify_pose（N5/N6，计划 §8）：按 zone 讲完整故事（问题→量化→证据→处方→职业对比）。

用户核心诉求（原话）：「我的问题是出现在那个位置，而不是只是告诉我预瞄差了多少」——
报告按空间位置组织，每个位置讲一个完整故事，指标只是佐证。

* ``verify_pose(geo, zone, my_pos)``：zone 内网格搜更优站位，返回前后指标对比。
  **没有 verify 结果的 position finding 不写绝对建议。**
* ``build_findings(frame, geo, zmap)``：按 zone 聚合，输出完整故事结构。
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

from .config import Vec3
from .zones import Zone, zone_of


def verify_pose(geo: Any, zone: Zone, my_pos: Vec3, *, step: float = 40.0, k: int = 3) -> dict[str, Any]:
    """在 zone 内网格搜更优站位，返回 {base, candidates}。"""
    base = {
        "exposure": geo.exposure(my_pos)[0],
        "crossfire": geo.crossfire(my_pos),
        "escape_time": geo.escape_time(my_pos),
    }
    out = []
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            p = (my_pos[0] + dx * step, my_pos[1] + dy * step, my_pos[2])
            if ((p[0] - zone.center[0]) ** 2 + (p[1] - zone.center[1]) ** 2) ** 0.5 > zone.radius:
                continue
            exp, entries = geo.exposure(p)
            if exp is None or base["exposure"] is None or exp >= base["exposure"]:
                continue
            out.append({
                "pos": [round(p[0], 1), round(p[1], 1)],
                "exposure": exp,
                "crossfire": geo.crossfire(p),
                "escape_time": geo.escape_time(p),
                "entries": entries,
            })
    out.sort(key=lambda r: (r["exposure"], r["escape_time"] if r["escape_time"] is not None else 9))
    return {"base": base, "candidates": out[:k]}


def build_findings(frame: pl.DataFrame, geo: Any, zmap: dict[str, Zone],
                   *, min_deaths: int = 1) -> list[dict[str, Any]]:
    """按 zone 聚合死亡点，输出完整故事（N6 §T2）。

    每个 finding：
    * zone / n_deaths / dominant_fault
    * evidence：该 zone 的量化指标均值（exposure/escape/gap/aim_off/职业分位）
    * typical_deaths：3 个典型 tick（含机制+证据串）
    * pro_compare：职业落位分位 + 冷门排名
    * verify_pose：position 时必有
    * advice：action + verify 前后数字 + note
    """
    if frame.is_empty() or "zone" not in frame.columns or "fault" not in frame.columns:
        return []

    findings = []
    for zname in frame["zone"].drop_nulls().unique().to_list():
        sub = frame.filter(pl.col("zone") == zname)
        if sub.height < min_deaths:
            continue

        faults = sub["fault"].value_counts().to_dicts()
        dom = max(faults, key=lambda r: r["count"]) if faults else {"fault": "?", "count": 0}
        dominant = dom["fault"]

        # 量化证据（均值）
        ev = _evidence(sub)

        # 典型死亡（取前 3 个 tick）
        typical = []
        for r in sub.head(3).iter_rows(named=True):
            typical.append({
                "tick": r.get("death_tick"),
                "mechanism": r.get("mechanism"),
                "fault": r.get("fault"),
                "evidence": r.get("fault_evidence") or r.get("note") or "",
            })

        # 职业对比
        pro_pct = ev.get("pro_zone_pct")
        cold_rank = _cold_rank(frame, zname)

        finding: dict[str, Any] = {
            "id": f"F-{dominant[:3]}-{zname}",
            "zone": zname,
            "n_deaths": sub.height,
            "dominant_fault": dominant,
            "fault_dist": {r["fault"]: r["count"] for r in faults},
            "evidence": ev,
            "typical_deaths": typical,
            "pro_compare": {"zone_pct": pro_pct, "cold_rank": cold_rank},
            "verify_pose": None,
            "advice": None,
            "confidence": "high" if sub.height >= 3 else ("mid" if sub.height == 2 else "low"),
        }

        # position → verify_pose + advice（§8.2：没有 verify 不写绝对建议）
        if dominant == "position":
            zkey = zname.lstrip("~")
            zone = zmap.get(zkey)
            if zone and geo is not None and sub.height:
                pos_row = sub.row(0, named=True)
                pos = (pos_row.get("my_x", 0.0), pos_row.get("my_y", 0.0), pos_row.get("my_z", 0.0))
                vp = verify_pose(geo, zone, pos)
                finding["verify_pose"] = vp
                if vp["candidates"]:
                    best = vp["candidates"][0]
                    finding["advice"] = {
                        "action": f"挪到 [{best['pos'][0]}, {best['pos'][1]}]",
                        "verify": {
                            "exposure": [vp["base"]["exposure"], best["exposure"]],
                            "crossfire": [vp["base"]["crossfire"], best["crossfire"]],
                            "escape_time": [vp["base"]["escape_time"], best["escape_time"]],
                        },
                        "note": "几何验算结果，非拍脑袋",
                    }
                else:
                    # 无可优化空间 → 说清楚是什么问题
                    esc = ev.get("escape_time")
                    finding["advice"] = {
                        "action": None,
                        "verify": None,
                        "note": f"该点位暴露面低但退路 {esc}s，是掩体配置问题" if (esc or 0) > 2
                                else "该点位无明显几何优化空间，对照热图自查站位习惯",
                    }
        elif dominant == "aim":
            # aim 类：给枪法建议（非几何，verify 不适用）
            finding["advice"] = {
                "action": "练枪法（该位置先手/命中不足）",
                "verify": None,
                "note": "见 duel_damage 图，非站位问题",
            }
        elif dominant == "preaim":
            finding["advice"] = {
                "action": "预瞄该位置常见出口方向",
                "verify": None,
                "note": f"准星偏差均值 {ev.get('my_aim_off_at_see_deg', '?')}°",
            }

        findings.append(finding)

    findings.sort(key=lambda f: -f["n_deaths"])
    return findings


def _evidence(sub: pl.DataFrame) -> dict[str, Any]:
    ev: dict[str, Any] = {}
    for col in ("exposure", "crossfire", "escape_time", "pro_cumulative_percentile",
                "first_seen_gap_ms", "my_aim_off_at_see_deg"):
        if col in sub.columns:
            vals = [v for v in sub[col].to_list() if v is not None]
            if vals:
                key = {"pro_cumulative_percentile": "pro_zone_pct",
                       "first_seen_gap_ms": "gap_mean",
                       "my_aim_off_at_see_deg": "aim_off_mean",
                       "escape_time": "escape_mean"}.get(col, col + "_mean")
                ev[key] = round(sum(vals) / len(vals), 2)
    return ev


def _cold_rank(frame: pl.DataFrame, zname: str) -> str:
    """该 zone 的职业分位在全死亡点里的冷门排名。"""
    if "pro_cumulative_percentile" not in frame.columns:
        return ""
    all_vals = sorted([v for v in frame["pro_cumulative_percentile"].to_list() if v is not None])
    sub_vals = [v for v in frame.filter(pl.col("zone") == zname)["pro_cumulative_percentile"].to_list() if v is not None]
    if not all_vals or not sub_vals:
        return ""
    avg = sum(sub_vals) / len(sub_vals)
    rank = sum(1 for v in all_vals if v < avg) + 1
    return f"{rank}/{len(all_vals)}"
