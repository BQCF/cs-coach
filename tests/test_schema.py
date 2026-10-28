"""列契约单测（评审 3.5：critical 与 optional 缺表要有区分）。"""

from __future__ import annotations

import polars as pl

from csc import schema


def test_missing_critical_table_severity():
    issues = schema.validate({"players": pl.DataFrame({"steamid": [1], "name": ["a"], "side": ["ct"]})})
    kinds = {i.kind for i in issues}
    assert "missing_critical_table" in kinds  # kills 是 critical
    assert "missing_table" in kinds            # blinds 等 optional


def test_missing_column_detected():
    df = pl.DataFrame({"tick": [1]})
    issues = schema.validate({"kills": df})
    assert any(i.kind == "missing_column" and i.table == "kills" for i in issues)


def test_render_markdown_runs():
    tables = {"kills": pl.DataFrame({"tick": [1]})}
    issues = schema.validate(tables)
    md = schema.render_markdown(tables, issues)
    assert "# 实际列名与契约核对" in md
    assert "kills" in md
