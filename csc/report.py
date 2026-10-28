"""报告：deaths.csv + report.md。

报告只做**描述性统计 + 人话结论**，不输出「你应该站这里」的绝对判断（除非有
暴露面/退路数据支撑）。样本不足 ``report.min_samples`` 时不下结论，直接写「样本不足」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from .config import ReportCfg


def write_deaths_csv(frame: pl.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def summarize(frame: pl.DataFrame) -> dict[str, Any]:
    """按标签分组统计：条数 + 各指标均值（可空的列自动跳过）。"""
    if frame.is_empty() or "label" not in frame.columns:
        return {"total": 0, "by_label": []}

    metric_cols = [c for c in ("first_seen_gap_ms", "exposure", "crossfire", "escape_time") if c in frame.columns]
    label_group = frame.group_by("label").agg(
        [
            pl.len().alias("n"),
            *[
                pl.col(c).drop_nulls().mean().alias(f"{c}_mean")
                for c in metric_cols
            ],
        ]
    ).sort("n", descending=True)
    return {
        "total": frame.height,
        "by_label": label_group.to_dicts(),
        "metric_cols": metric_cols,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)






def _render_zone_stories(fl: list, frame: pl.DataFrame) -> list[str]:
    """按位置复盘：每个 zone（>=2 次死亡）讲一个完整故事。"""
    stories = [f for f in (fl or []) if f.get("n_deaths", 0) >= 2]
    if not stories:
        return ["## 按位置复盘", "", "（暂无单点位 ≥2 次死亡，攒多几局后更有针对性）"]
    lines = ["## 按位置复盘（每个 zone 至少 2 次死亡才列）", ""]
    for f in stories:
        z = f.get("zone", "?")
        dom = f.get("dominant_fault", "?")
        n = f.get("n_deaths", 0)
        ev = f.get("evidence", {})
        pro = f.get("pro_compare", {})
        advice = f.get("advice") or {}

        dom_cn = {"aim":"枪法","position":"站位","position_config":"地图设置","preaim":"预瞄","info":"信息","none":"被阴"}.get(dom, dom)
        lines.append(f"### {z}：{n} 次死亡，主要是{dom_cn}问题")
        # 这里的问题
        probs = []
        if ev.get("exposure_mean") is not None:
            probs.append(f"暴露给 {ev['exposure_mean']} 个敌人")
        if ev.get("escape_mean") is not None:
            probs.append(f"撤回掩体要 {ev['escape_mean']} 秒")
        if ev.get("gap_mean") is not None:
            probs.append(f"谁先看见谁差 {ev['gap_mean']:+.0f} 毫秒")
        if ev.get("aim_off_mean") is not None:
            probs.append(f"准星偏了 {ev['aim_off_mean']}°")
        if probs:
            lines.append(f"- **这里的问题**：{'、'.join(probs)}")
        # 发生了什么（典型死亡）
        typical = f.get("typical_deaths", [])
        if typical:
            ev_texts = [f"tick {t.get('tick')}（{t.get('evidence','')[:40]}）" for t in typical]
            lines.append(f"- **发生了什么**：{'；'.join(ev_texts)}")
        # 职业对比
        if pro.get("zone_pct") is not None:
            rank = f"（冷门程度排第 {pro['cold_rank']}）" if pro.get("cold_rank") else ""
            lines.append(f"- **职业对比**：职业常站度 {pro['zone_pct']}{rank}（1=常站，0=冷门）")
        # 处方
        if advice.get("action"):
            v = advice.get("verify") or {}
            parts = [f"{k} {pair[0]}→{pair[1]}" for k, pair in v.items()
                     if pair and pair[0] is not None and pair[1] is not None]
            verify_str = f"（{'；'.join(parts)}）" if parts else ""
            lines.append(f"- **处方**：{advice['action']} {verify_str}")
        elif advice.get("note"):
            lines.append(f"- **处方**：{advice['note']}")
        # 典型死亡 tick（供截图）
        ticks = [str(t.get("tick")) for t in typical if t.get("tick")]
        if ticks:
            lines.append(f"- **典型死亡**：{'、'.join('tick ' + x for x in ticks)}")
        lines.append("")
    return lines


def _render_global_profile(frame: pl.DataFrame) -> list[str]:
    """全局画像（下沉到报告末尾，大白话）。"""
    from .preaim import aim_stats

    lines = ["## 整体情况", ""]
    if "my_aim_off_at_see_deg" in frame.columns:
        st = aim_stats(frame["my_aim_off_at_see_deg"].to_list())
        if st["n"]:
            lines.append(f"- 准星偏差中位 **{st['median']}°**（敌人冒出来时你偏这么多），{st['over_pct']}% 的死亡偏超 10°")
            if "zone" in frame.columns:
                worst = None
                for z in frame["zone"].drop_nulls().unique().to_list():
                    sub = frame.filter(pl.col("zone") == z)
                    zs = aim_stats(sub["my_aim_off_at_see_deg"].to_list())
                    if zs["n"] >= 2 and (worst is None or (zs["median"] or 0) > (worst[1] or 0)):
                        worst = (z, zs["median"])
                if worst and st["median"]:
                    ratio = (worst[1] / st["median"]) if st["median"] else 0
                    lines.append(f"- 但 **{worst[0]}** 这个点偏到 {worst[1]:.1f}°，是平均的 {ratio:.1f} 倍（最差的点位）")
    if "pro_cumulative_percentile" in frame.columns:
        vals = [v for v in frame["pro_cumulative_percentile"].to_list() if v is not None]
        if vals:
            avg = sum(vals) / len(vals)
            cold = sum(1 for v in vals if v < 0.2)
            lines.append(f"- 站位常规度 {avg:.2f}（1=职业常站，0=冷门），{cold}/{len(vals)} 次死在冷门位")
    return lines


def write_report_md(
    frame: pl.DataFrame,
    summary: Mapping[str, Any],
    path: Path,
    *,
    title: str = "CS Coach 体检报告",
    player: str | None = None,
    caveats: Sequence[str] | None = None,
    findings_list: Sequence[Any] | None = None,
    charts: Mapping[str, str] | None = None,
    radar_dots_img: str | None = None,
    radar_chart_img: str | None = None,
    fp_shots: list | None = None,
    cfg: ReportCfg | None = None,
) -> Path:
    """报告新结构（用户 spec）：位置分布总览 → 按位置复盘 → 全局画像（下沉）。

    删掉「按归因标签统计」「责任分布」独立段；六维雷达放最后。
    """
    cfg = cfg or ReportCfg()
    lines = [f"# {title}", ""]
    if player:
        lines.append(f"- 玩家：{player}")
    lines.append(f"- 死亡点总数：{summary.get('total', frame.height)}")
    lines.append("")

    # 1. 位置分布总览（雷达打点）
    lines += ["## 位置分布总览", ""]
    if radar_dots_img:
        lines.append(f"![雷达打点]({radar_dots_img})")
    else:
        lines.append("（雷达图生成需要 awpy 资源）")
    lines.append("")

    # 1.5 第一人称视角示意（站位/预瞄问题的直观证据）
    if fp_shots:
        lines += [
            "", "## 你看这一枪是怎么输的（第一人称视角）", "",
            "> 有「游戏内截图」字样的卡片是**真截图 + 标注**；其余是地图模型渲染的示意图——`.dem` 只记录位置/朝向/事件数据，不含游戏画面。",
            "> 想把示意图换成真截图：按 `replay_guide.md` 操作（CS2 控制台 `demo_gototick <tick>` 跳过去截图，放进 `shots/<tick>.png`），重跑报告自动替换为「真截图 + 标注」版。", "",
        ]
        for fp in fp_shots:
            fname = fp.get("file", "")
            caption = fp.get("caption", "")
            lines.append(f"![{caption}]({fname})")
            lines.append("")
            if caption:
                lines.append(f"*{caption}*")
                lines.append("")

    # 2. 按位置复盘（每 zone >=2 死亡）
    lines += _render_zone_stories(list(findings_list or []), frame)

    # 3. 全局画像（下沉）
    lines += _render_global_profile(frame)

    # 4. 六维能力图（放最后）
    if radar_chart_img:
        lines += ["", "## 六维能力图", "", f"![六维能力]({radar_chart_img})"]

    # 5. 预瞄分布 / 对枪图（佐证，放末尾）
    if charts:
        extra = []
        for key, title2 in (("preaim_distribution", "预瞄偏差分布"), ("duel_damage", "对枪伤害对比")):
            if key in charts:
                extra += ["", f"### {title2}（佐证）", f"![{title2}]({charts[key]})"]
        if extra:
            lines += ["", "## 佐证图表", *extra]

    if caveats:
        lines += ["", "## 已知误差 / 降级", ""]
        for c in caveats:
            lines.append(f"- {c}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
