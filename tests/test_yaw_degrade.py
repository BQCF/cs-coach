"""朝向缺失时的降级：gap / predictable 应为 null，而不是悄悄按 0° 算出错值。"""

from __future__ import annotations

import dataclasses

import polars as pl

from csc import config, deaths, metrics, synthetic, windows


def _windows():
    s = config.load_settings()
    ctx = synthetic.make_context()
    snap = synthetic.make_snap()
    spans = windows.plan_spans(synthetic.death_ticks(), synthetic.TICK_RATE, s.windows)
    ws = deaths.build_death_windows(ctx, synthetic.MY_STEAMID, spans, snap)
    return s, ws


def _no_yaw(track: pl.DataFrame) -> pl.DataFrame:
    return track.with_columns(pl.lit(None, dtype=pl.Float64).alias("yaw"))


def test_gap_null_when_track_yaw_missing():
    s, ws = _windows()
    geo = synthetic.make_geo(s)
    for w in ws:
        w2 = dataclasses.replace(w, me_track=_no_yaw(w.me_track), killer_track=_no_yaw(w.killer_track))
        assert metrics.first_seen_gap(w2, geo, s.rules) is None


def test_predictable_null_when_track_yaw_missing():
    s, ws = _windows()
    for w in ws:
        if w.killer_track is None or w.killer_track.is_empty():
            continue
        w2 = dataclasses.replace(w, killer_track=_no_yaw(w.killer_track))
        assert metrics.predictable(w2, s.rules) is None


def test_predictable_null_without_track_and_yaw():
    s = config.load_settings()
    w = deaths.DeathWindow(
        death_tick=1000,
        round_num=1,
        tick_rate=128,
        my_steamid=111,
        my_name="me",
        killer_steamid=222,
        killer_name="killer",
        weapon="ak47",
        headshot=False,
        my_pos=(0.0, 0.0, 0.0),
        killer_pos=(200.0, 0.0, 0.0),
        my_yaw=None,
        killer_yaw=None,
        my_pitch=None,
        killer_pitch=None,
        my_speed=0.0,
        killer_speed=0.0,
        my_crouched=False,
        killer_crouched=False,
    )
    assert metrics.predictable(w, s.rules) is None
    assert metrics.first_seen_gap(w, None, s.rules) is None
