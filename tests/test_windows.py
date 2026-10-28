"""窗口规划 + 缓存签名单测（评审 3.1：缓存 key 必须含窗口内容）。"""

from __future__ import annotations

from csc import config, windows


def _cfg(**kw):
    defaults = dict(pre_s=2.0, post_s=0.5, merge_gap_s=1.0, max_windows=400)
    defaults.update(kw)
    return config.WindowsCfg(**defaults)


def test_plan_spans_merge_when_close():
    # 相邻死亡 30 tick，间隔 < merge_gap(128 tick) → 合并
    spans = windows.plan_spans([1000, 1030], 128, _cfg())
    assert len(spans) == 1
    assert spans[0].start_tick == 1000 - 256
    assert spans[0].end_tick == 1030 + 64
    assert spans[0].death_ticks == (1000, 1030)


def test_plan_spans_no_merge_when_far():
    spans = windows.plan_spans([1000, 2000], 128, _cfg())
    assert len(spans) == 2


def test_plan_spans_max_windows_guard():
    try:
        windows.plan_spans([1000, 2000, 3000], 128, _cfg(max_windows=2))
        assert False, "应该抛 WindowError"
    except windows.WindowError:
        pass


def test_span_signature_depends_on_player():
    spans = windows.plan_spans([1000], 128, _cfg())
    assert windows._span_signature(spans, 128, 111) != windows._span_signature(spans, 128, 222)


def test_span_signature_depends_on_spans():
    a = windows.plan_spans([1000], 128, _cfg())
    b = windows.plan_spans([2000], 128, _cfg())
    assert windows._span_signature(a, 128, 111) != windows._span_signature(b, 128, 111)


def test_normalize_snap_fills_missing_columns():
    import polars as pl

    df = pl.DataFrame({"tick": [1], "steamid": [111]})
    out = windows.normalize_snap(df)
    for col in windows.TRACK_COLUMNS:
        assert col in out.columns
