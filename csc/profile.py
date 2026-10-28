"""画像（N5/N6）：「你是什么水平、短板排序」——短板诊断卡（大白话版）。

设计原则（用户要求）：**读者是普通 CS 玩家 / 比赛粉丝，不用专业英文术语**。
所以输出都是「准星偏了多少度」「撤回掩体要几秒」这种大白话，不用 FOV/percentile 这类词。

六维能力（按「影响死亡数 × 与职业差」排序）：
* 对枪赢面   —— 你先开枪的对枪里，赢了几次
* 谁先看见谁 —— 你和他谁先看见对方，差多少毫秒
* 准星偏差   —— 敌人冒出来时，你准星偏了几度
* 站位常规度 —— 你站的位置职业哥常不常站
* 撤回掩体   —— 撤回掩体要几秒
* 情报失误   —— 没看见人就被杀的比例
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

DIMS = [
    ("duel_win", "对枪赢面", "high", "你先开枪的对枪里赢了几次"),
    ("gap_ms", "谁先看见谁", "small", "你和他谁先看见对方，差多少毫秒"),
    ("aim_off", "准星偏差", "small", "敌人冒出来时你准星偏了几度"),
    ("zone_norm", "站位常规度", "high", "你站的位置职业哥常不常站"),
    ("escape", "撤回掩体", "small", "撤回掩体要几秒"),
    ("info", "情报失误", "small", "没看见人就被打死的比例"),
]

ADVICE = {
    "duel_win": "对枪前先站稳、把准星压在爆头线上再开枪",
    "gap_ms": "别一直晃，卡好角度等人来，你先看见他就赢一半",
    "aim_off": "多练预瞄——进每个路口前先把准星贴着墙角，等人冒出来",
    "zone_norm": "看看职业热图，别站那些没人站的冷门位",
    "escape": "选站位时先想好「打不过往哪退」，别站没掩体的地方",
    "info": "多看小地图、听脚步，别闷头往前冲",
}


def _median(vals: list[float]) -> float | None:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def compute_dims(frame: pl.DataFrame) -> dict[str, Any]:
    """从 deaths 汇总六维。返回 {dim_key: {value, n}}。"""
    if frame.is_empty():
        return {}
    out: dict[str, Any] = {}
    if "first_fire" in frame.columns and "fault" in frame.columns:
        me_first = frame.filter(pl.col("first_fire") == "me")
        n = me_first.height
        wins = me_first.filter(pl.col("fault") != "aim").height
        out["duel_win"] = {"value": round(wins / n * 100, 1) if n else None, "n": n}
    if "first_seen_gap_ms" in frame.columns:
        vals = [v for v in frame["first_seen_gap_ms"].to_list() if v is not None]
        out["gap_ms"] = {"value": _median(vals), "n": len(vals)}
    if "my_aim_off_at_see_deg" in frame.columns:
        vals = [v for v in frame["my_aim_off_at_see_deg"].to_list() if v is not None]
        out["aim_off"] = {"value": _median(vals), "n": len(vals)}
    if "pro_cumulative_percentile" in frame.columns:
        vals = [v for v in frame["pro_cumulative_percentile"].to_list() if v is not None]
        out["zone_norm"] = {"value": round(_median(vals), 3) if vals else None, "n": len(vals)}
    if "escape_time" in frame.columns:
        vals = [v for v in frame["escape_time"].to_list() if v is not None]
        out["escape"] = {"value": _median(vals), "n": len(vals)}
    if "fault" in frame.columns:
        n = frame.height
        info = frame.filter(pl.col("fault") == "info").height
        out["info"] = {"value": round(info / n * 100, 1) if n else None, "n": n}
    return out


def short_board(dims: dict[str, Any], *, min_n: int = 10) -> list[str]:
    """短板排序（大白话）。n<min_n 的标「样本不够」。"""
    def score(key, v):
        if v.get("value") is None:
            return 0
        good = dict((d[0], d[2]) for d in DIMS)[key]
        return -v["value"] if good == "high" else v["value"]

    names = {d[0]: d[1] for d in DIMS}
    ok = [(k, v) for k, v in dims.items() if v.get("value") is not None and v.get("n", 0) >= min_n]
    weak = [(k, v) for k, v in dims.items() if v.get("value") is not None and v.get("n", 0) < min_n]
    ranked = sorted(ok, key=lambda kv: score(kv[0], kv[1]), reverse=True)
    out = [f"{names[k]}（{_fmt(v['value'])}，样本 {v['n']}）" for k, v in ranked]
    out += [f"{names[k]}（{_fmt(v['value'])}，样本 {v['n']}）—— 样本不够，先不下结论" for k, v in weak]
    return out


def _fmt(v):
    return round(v, 2) if isinstance(v, float) else v


def short_board_cards(dims: dict[str, Any], frame: pl.DataFrame,
                      pro_baseline: dict[str, float] | None = None, *, min_n: int = 10) -> list[dict[str, Any]]:
    """短板诊断卡：按「影响死亡数 × 与职业差」排序，每卡含发生在哪 + 量化影响 + 改善方向。"""
    def impact(key, v):
        if v.get("value") is None:
            return 0
        good = dict((d[0], d[2]) for d in DIMS)[key]
        sev = -v["value"] if good == "high" else v["value"]
        return sev * v.get("n", 1)

    ok = [(k, v) for k, v in dims.items() if v.get("value") is not None and v.get("n", 0) >= min_n]
    ranked = sorted(ok, key=lambda kv: impact(kv[0], kv[1]), reverse=True)
    names = {d[0]: d[1] for d in DIMS}

    cards = []
    for i, (k, v) in enumerate(ranked, 1):
        pro_val = (pro_baseline or {}).get(k)
        diff = None
        if pro_val is not None and v["value"]:
            good = dict((d[0], d[2]) for d in DIMS)[k]
            diff = ((v["value"] - pro_val) / pro_val * 100) if good == "high" else ((pro_val - v["value"]) / pro_val * 100)
        top_zones = _top_zones(frame, k)
        cards.append({
            "rank": i, "name": names[k], "value": v["value"], "n": v["n"],
            "pro": pro_val, "diff": round(diff, 0) if diff is not None else None,
            "top_zones": top_zones, "advice": ADVICE.get(k, ""),
        })
    return cards


def _top_zones(frame: pl.DataFrame, dim_key: str, top: int = 3) -> list[str]:
    """该维度最差的 zone（按值最差排序）。"""
    col = {"gap_ms": "first_seen_gap_ms", "aim_off": "my_aim_off_at_see_deg",
           "escape": "escape_time", "zone_norm": "pro_cumulative_percentile",
           "duel_win": None, "info": None}.get(dim_key)
    if col is None or col not in frame.columns or "zone" not in frame.columns:
        return []
    good = dict((d[0], d[2]) for d in DIMS)[dim_key]
    zstats = []
    for z in frame["zone"].drop_nulls().unique().to_list():
        vals = [v for v in frame.filter(pl.col("zone") == z)[col].to_list() if v is not None]
        if vals:
            zstats.append((z, _median(vals)))
    if good == "high":
        zstats.sort(key=lambda x: x[1])
    else:
        zstats.sort(key=lambda x: -x[1])
    return [z for z, _ in zstats[:top]]


def radar_points(dims: dict[str, Any]) -> list[float]:
    """六维归一化 0-1（越靠外越好）供雷达图。"""
    pts = []
    for key, _name, good, _f in DIMS:
        v = dims.get(key, {}).get("value")
        if v is None:
            pts.append(0.0)
            continue
        if key == "duel_win":
            pts.append(v / 100)
        elif key == "zone_norm":
            pts.append(v)
        elif key == "info":
            pts.append(1 - v / 100)
        elif key == "gap_ms":
            pts.append(max(0.0, 1 - max(0, v) / 500))
        elif key == "aim_off":
            pts.append(max(0.0, 1 - v / 45))
        elif key == "escape":
            pts.append(max(0.0, 1 - v / 5))
        else:
            pts.append(0.0)
    return [round(p, 3) for p in pts]
