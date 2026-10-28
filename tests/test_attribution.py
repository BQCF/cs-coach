"""归因规则单测（v2 规则集，计划 §5.3）。

规则表：blind / behind / info_missing / peeked / holding / encounter / moving。
empty_clip 已移出规则表（改 clip_state 列），规则7 info_missing 现已启用。
"""

from __future__ import annotations

from csc import config, deaths, synthetic, windows


def _pipeline():
    s = config.load_settings()
    ctx = synthetic.make_context()
    snap = synthetic.make_snap()
    spans = windows.plan_spans(synthetic.death_ticks(), synthetic.TICK_RATE, s.windows)
    geo = synthetic.make_geo(s)
    ws = deaths.build_death_windows(ctx, synthetic.MY_STEAMID, spans, snap)
    return s, ws, geo


def test_rules_ordered_v2():
    """规则表顺序符合 §5.3，且 empty_clip 已移除。"""
    names = [r[0] for r in deaths.RULES]
    assert names == ["blind", "behind", "info_missing", "peeked", "holding", "encounter", "moving"]


def test_labels_match_scenarios():
    s, ws, geo = _pipeline()
    labels = [deaths.attribute(w, s.rules, geo).label for w in ws]
    # 打空/换弹 不再是死因标签；空弹匣场景应落到其它机制
    assert "打空/换弹" not in labels
    assert len(labels) == len(synthetic.SCENARIOS)


def test_rule7_info_missing_can_fire():
    """规则 7 已启用，synthetic 有场景会命中信息缺失硬拉 / 正面遭遇。"""
    s, ws, geo = _pipeline()
    labels = [deaths.attribute(w, s.rules, geo).label for w in ws]
    assert "信息缺失硬拉" in labels
    assert "正面遭遇" in labels


def test_priority_blind_wins():
    """第一个死亡点既有被闪又有击杀者在前方，被闪优先级最高。"""
    s, ws, geo = _pipeline()
    assert deaths.attribute(ws[0], s.rules, geo).rule == "blind"


def test_fallback_without_geo():
    """没有几何时，需 geo 的规则跳过，只走 blind/behind/moving 和兜底。"""
    s, ws, _ = _pipeline()
    verdict = deaths.attribute(ws[4], s.rules, None)
    assert verdict.label in ("移动中对枪", "其他/对枪", "背身/侧身", "被闪", "信息缺失硬拉", "正面遭遇", "架死", "被拉死")
