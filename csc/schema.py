"""列契约。

awpy 不同版本字段名会漂移（``victim_name`` / ``attacker_x`` 之类），解析完第一时间
校验并落盘 ``output/columns.md``，比等到指标算出 NaN 再回查便宜得多。

契约只做**warn**，不抛异常：缺列的后果由下游各模块自己降级。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import polars as pl

# --------------------------------------------------------------------------- 契约定义

KILLS = (
    "tick",
    "attacker_steamid",
    "attacker_name",
    "attacker_x",
    "attacker_y",
    "attacker_z",
    "victim_steamid",
    "victim_name",
    "victim_x",
    "victim_y",
    "victim_z",
    "weapon",
    "headshot",
)

SNAP = (
    "tick",
    "steamid",
    "name",
    "side",
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "velocity",
    "health",
    "armor",
    "is_crouched",
)

BLINDS = ("tick", "victim_steamid", "victim_name", "attacker_steamid", "duration")
SHOTS = ("tick", "steamid", "name", "x", "y", "z", "yaw", "pitch", "weapon", "num_bullets_remaining")
SMOKES = ("start_tick", "end_tick", "x", "y", "z")
FIRES = ("start_tick", "end_tick", "x", "y", "z")
ROUNDS = ("round_num", "start_tick", "freeze_end_tick", "end_tick")
PLAYERS = ("steamid", "name", "side")
DAMAGES = ("tick", "attacker_steamid", "victim_steamid")
ECONOMY = ("round_num", "side", "equipment_value", "buy_type")

CONTRACTS: dict[str, tuple[str, ...]] = {
    "kills": KILLS,
    "blinds": BLINDS,
    "shots": SHOTS,
    "smokes": SMOKES,
    "fires": FIRES,
    "rounds": ROUNDS,
    "players": PLAYERS,
    "damages": DAMAGES,
    "snap": SNAP,
    "round_economy": ECONOMY,
}

# 缺了这些表基本没法做死亡分析
CRITICAL = ("kills", "players")
# 缺了这些只是少某个指标
OPTIONAL = ("blinds", "shots", "smokes", "fires", "rounds", "damages", "round_economy", "snap")


# --------------------------------------------------------------------------- 校验


@dataclass(frozen=True)
class ContractIssue:
    table: str
    kind: str  # missing_table / missing_column / empty
    detail: str

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"[{self.kind}] {self.table}: {self.detail}"


def validate(tables: Mapping[str, pl.DataFrame]) -> list[ContractIssue]:
    """按契约检查，返回问题清单（不抛异常）。"""
    issues: list[ContractIssue] = []
    for table, required in CONTRACTS.items():
        df = tables.get(table)
        if df is None:
            severity = "missing_critical_table" if table in CRITICAL else "missing_table"
            issues.append(ContractIssue(table, severity, "表不存在，相关结论会降级"))
            continue
        columns = set(df.columns)
        missing = [col for col in required if col not in columns]
        if missing:
            issues.append(ContractIssue(table, "missing_column", "缺少列：" + ", ".join(missing)))
        if df.height == 0:
            issues.append(ContractIssue(table, "empty", "表为空（该 demo 可能没有这类事件）"))
    return issues


def table_stats(tables: Mapping[str, pl.DataFrame]) -> dict[str, dict[str, int]]:
    return {
        name: {"rows": df.height, "cols": len(df.columns)}
        for name, df in tables.items()
        if isinstance(df, pl.DataFrame)
    }


def render_markdown(tables: Mapping[str, pl.DataFrame], issues: Iterable[ContractIssue]) -> str:
    """生成 output/columns.md 的正文。"""
    issues = list(issues)
    lines = [
        "# 实际列名与契约核对",
        "",
        "由 `python -m csc parse` 自动生成。awpy 版本升级后重点看这里。",
        "",
        "## 概览",
        "",
        "| 表 | 行数 | 列数 | 状态 |",
        "| --- | --- | --- | --- |",
    ]
    issue_tables = {i.table for i in issues}
    for name, stat in table_stats(tables).items():
        status = "⚠️ 有问题" if name in issue_tables else "✅"
        lines.append(f"| `{name}` | {stat['rows']} | {stat['cols']} | {status} |")

    if issues:
        lines += ["", "## 问题", ""]
        for issue in issues:
            lines.append(f"- **{issue.table}** `{issue.kind}`：{issue.detail}")
    else:
        lines += ["", "全部契约列齐备 ✅"]

    lines += ["", "## 各表实际列", ""]
    for name, df in tables.items():
        if not isinstance(df, pl.DataFrame):
            continue
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("```text")
        lines.append(", ".join(df.columns))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
