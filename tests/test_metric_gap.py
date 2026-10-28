"""先手差（FOV 版）单测：被拉死为正，架死约为 0。"""

from __future__ import annotations

from csc import config, deaths, metrics, synthetic, windows


def _records():
    s = config.load_settings()
    ctx = synthetic.make_context()
    snap = synthetic.make_snap()
    spans = windows.plan_spans(synthetic.death_ticks(), synthetic.TICK_RATE, s.windows)
    geo = synthetic.make_geo(s)
    ws = deaths.build_death_windows(ctx, synthetic.MY_STEAMID, spans, snap)
    return s, ws, geo


def _window_of(s, ws, geo, label):
    return next(w for w in ws if deaths.attribute(w, s.rules, geo).label == label)


def test_peeked_gap_positive():
    """被拉死：他先看见我，gap 应为正。"""
    s, ws, geo = _records()
    w = _window_of(s, ws, geo, "被拉死")
    m = metrics.compute(w, geo, s.rules)
    assert m.first_seen_gap_ms is not None
    assert m.first_seen_gap_ms > 0


def test_holding_gap_not_positive():
    """架死：双方早已通视，gap 应约等于 0（我先架住 = 不亏）。"""
    s, ws, geo = _records()
    w = _window_of(s, ws, geo, "架死")
    m = metrics.compute(w, geo, s.rules)
    assert m.first_seen_gap_ms is not None
    assert m.first_seen_gap_ms <= 0


def test_gap_within_sane_range():
    """评审 4.3：合理区间约 -300~400ms，超出会在 notes 里标可疑。"""
    s, ws, geo = _records()
    for w in ws:
        m = metrics.compute(w, geo, s.rules)
        if m.first_seen_gap_ms is not None:
            if not (metrics.GAP_SANE[0] <= m.first_seen_gap_ms <= metrics.GAP_SANE[1]):
                assert any("可疑" in n for n in m.notes)
