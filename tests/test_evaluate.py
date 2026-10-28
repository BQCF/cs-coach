"""M3 eval 的对齐与零匹配行为。

两个曾经的坑：
1. 零匹配时输出「准确率 0.0% 全部一致 ✅」——假成功，必须明确报「没有匹配样本」；
2. 只按 death_tick join，跨 demo 同 tick 会静默错配——deaths.csv 带 demo 列时必须按场对齐。
"""

from __future__ import annotations

import polars as pl

from csc import evaluate

LABELS = pl.DataFrame(
    {
        "demo": ["a.dem", "a.dem", "b.dem"],
        "death_tick": [100, 200, 300],
        "label": ["被闪", "被拉死", "被闪"],
        "note": ["", "", ""],
    }
)


def _preds(with_demo: bool) -> pl.DataFrame:
    data = {
        "death_tick": [100, 200, 300],
        "label": ["被闪", "架死", "移动中对枪"],
        "label_conf": [1.0, 0.55, 0.6],
        "rule": ["blind", "holding", "moving"],
    }
    if with_demo:
        data["demo"] = ["a.dem", "a.dem", "b.dem"]
    return pl.DataFrame(data)


def test_demo_aware_join_matches_same_demo_only():
    """带 demo 列时按 (demo, tick) 对齐；b.dem 的 300 与 a.dem 的 300 不会错配。"""
    labels = pl.DataFrame(
        {
            "demo": ["a.dem", "b.dem"],
            "death_tick": [300, 300],
            "label": ["被闪", "架死"],
            "note": ["", ""],
        }
    )
    preds = pl.DataFrame(
        {
            "demo": ["b.dem"],
            "death_tick": [300],
            "label": ["架死"],
            "label_conf": [1.0],
            "rule": ["holding"],
        }
    )
    result = evaluate.evaluate(labels, preds)
    assert result["matched_on"] == "demo+death_tick"
    assert result["n"] == 1  # a.dem@300 不该被 b.dem@300 顶上
    assert result["accuracy"] == 1.0


def test_tick_only_fallback_warns_mode():
    result = evaluate.evaluate(LABELS, _preds(with_demo=False))
    assert result["matched_on"] == "death_tick"
    assert result["n"] == 3
    # a.dem 被闪@100 对、200 被拉死→架死 错、b.dem 300 被闪→移动中对枪 错
    assert result["accuracy"] == round(1 / 3, 4)


def test_zero_match_is_not_success():
    """零匹配：accuracy=None，报告里必须是「没有匹配样本」而不是「全部一致」。"""
    preds = pl.DataFrame(
        {
            "demo": ["c.dem"],
            "death_tick": [999],
            "label": ["被闪"],
            "label_conf": [1.0],
            "rule": ["blind"],
        }
    )
    result = evaluate.evaluate(LABELS, preds)
    assert result["n"] == 0
    assert result["accuracy"] is None
    md = evaluate.render_report(result)
    assert "没有任何标注能对上" in md
    assert "全部一致" not in md
    assert "0.0%" not in md
