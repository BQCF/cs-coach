"""M3 评估：把自动归因（deaths.csv）跟人工标注（labels.csv）对比，算准确率。

labels.csv 格式（golden labels，人工标注 ground truth）::

    demo,death_tick,label,note
    vitality_vs_g2_inferno.dem,5586,移动中对枪,冲出去被打
    vitality_vs_g2_inferno.dem,8934,背身/侧身,被绕后

按 ``demo + death_tick`` 对齐（deaths.csv 缺 demo 列时回退只按 death_tick，
并在报告里标注对齐方式），输出：
* 总体准确率
* 每标签的精确率/召回率/混淆对
* 归因不一致的死亡点明细（便于回头调规则）

一对不上就明确报「没有匹配样本」，绝不假装「全部一致」。

M3 验收：抽 10+ 个死亡点人工标注，准确率 > 60%。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

LABELS_NAME = "labels.csv"


def load_labels(path: Path) -> pl.DataFrame:
    """读人工标注。列：demo, death_tick, label[, note]。"""
    if not Path(path).exists():
        raise FileNotFoundError(f"标注文件不存在：{path}")
    df = pl.read_csv(path)
    required = {"demo", "death_tick", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv 缺列：{missing}（需要 demo, death_tick, label）")
    return df.with_columns(pl.col("death_tick").cast(pl.Int64))


def load_predictions(deaths_csv: Path) -> pl.DataFrame:
    """读自动归因结果。"""
    if not Path(deaths_csv).exists():
        raise FileNotFoundError(f"deaths.csv 不存在：{deaths_csv}")
    df = pl.read_csv(deaths_csv)
    return df.with_columns(pl.col("death_tick").cast(pl.Int64))


def match_mode(preds: pl.DataFrame) -> str:
    """对齐方式：deaths.csv 带 demo 列时可精确到「同一场」，否则只能按 tick。"""
    return "demo+death_tick" if "demo" in preds.columns else "death_tick"


def _join(labels: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    """对齐标注与预测；preds 带 demo 列时按 (demo, death_tick) 精确对齐。"""
    cols = ["death_tick", "label", "label_conf", "rule"]
    on = ["death_tick"]
    if "demo" in preds.columns:
        cols = ["demo"] + cols
        on = ["demo", "death_tick"]
    return labels.join(preds.select(cols), on=on, how="inner", suffix="_pred")


def evaluate(labels: pl.DataFrame, preds: pl.DataFrame) -> dict:
    """算准确率 + 每标签指标 + 不一致明细；零匹配时 accuracy=None（不是 0）。"""
    joined = _join(labels, preds)
    mode = match_mode(preds)
    if joined.is_empty():
        return {"n": 0, "accuracy": None, "per_label": [], "mismatches": [], "matched_on": mode}

    n = joined.height
    correct = joined.filter(pl.col("label") == pl.col("label_pred")).height
    accuracy = correct / n

    per_label = []
    for lab in labels["label"].unique().to_list():
        gt = joined.filter(pl.col("label") == lab)
        tp = gt.filter(pl.col("label") == pl.col("label_pred")).height
        pred_as_lab = joined.filter(pl.col("label_pred") == lab).height
        recall = tp / gt.height if gt.height else 0.0
        precision = tp / pred_as_lab if pred_as_lab else 0.0
        per_label.append({
            "label": lab, "support": gt.height,
            "precision": round(precision, 3), "recall": round(recall, 3),
        })

    mcols = (["demo"] if "demo" in joined.columns else []) + [
        "death_tick", "label", "label_pred", "label_conf", "rule",
    ]
    mismatches = joined.filter(pl.col("label") != pl.col("label_pred")).select(mcols).to_dicts()

    return {
        "n": n, "accuracy": round(accuracy, 4),
        "per_label": per_label, "mismatches": mismatches, "matched_on": mode,
    }


def render_report(result: dict) -> str:
    lines = ["# M3 归因评估报告", ""]
    lines.append(f"标注样本数：{result['n']}（对齐方式：{result.get('matched_on', 'death_tick')}）")
    if not result["n"]:
        lines += [
            "",
            "**没有任何标注能对上自动结果，准确率无意义。**",
            "",
            "常见原因：`--deaths` 不是标注的那场 demo / death_tick 对不上 / labels.csv 写错。",
            "deaths.csv 由新版本 csc 生成时带 demo 列，会自动按「同一场 demo」对齐。",
        ]
        return "\n".join(lines)
    lines.append(f"**总体准确率：{result['accuracy']*100:.1f}%**")
    lines.append("")
    if result["per_label"]:
        lines += ["## 每标签指标", "", "| 标签 | 样本 | 精确率 | 召回率 |", "| --- | --- | --- | --- |"]
        for r in result["per_label"]:
            lines.append(f"| {r['label']} | {r['support']} | {r['precision']} | {r['recall']} |")
        lines.append("")
    if result["mismatches"]:
        lines += [f"## 归因不一致（{len(result['mismatches'])} 个）", ""]
        for m in result["mismatches"]:
            where = f"{m['demo']} " if m.get("demo") else ""
            lines.append(
                f"- {where}tick {m['death_tick']}：标注「{m['label']}」"
                f"→ 自动「{m['label_pred']}」(rule={m['rule']}, conf={m['label_conf']})"
            )
    else:
        lines.append("全部一致 ✅")
    return "\n".join(lines)
