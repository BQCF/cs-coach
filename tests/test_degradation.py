"""降级路径测试矩阵（评审 4.6）。

关键保证：这些模块在**没有 awpy / 没有 mesh / 地图未配 / 空 demo** 时都不该炸，
而是返回 null / 空结果。awpy 是延迟导入的，import 这些包不 import awpy。
"""

from __future__ import annotations

import sys

import polars as pl

from csc import config, deaths, metrics, windows
from csc.geometry import GeoContext
from csc.metrics import compute


def _settings():
    return config.load_settings()


def test_import_does_not_pull_awpy():
    """不装 awpy 时 import csc 也能成功（awpy 只在函数里延迟导入）。"""
    assert "awpy" not in sys.modules
    import csc  # noqa: F401

    import csc.demo_io  # noqa: F401
    import csc.geometry  # noqa: F401
    import csc.metrics  # noqa: F401
    import csc.deaths  # noqa: F401
    import csc.cli  # noqa: F401

    assert "awpy" not in sys.modules


def test_compute_without_geo_returns_nulls():
    """GeoContext(vc=None) 时 compute 返回全 null 且不抛。"""
    s = _settings()
    w = _minimal_window()
    result = compute(w, None, s.rules)
    assert result.exposure is None
    assert result.first_seen_gap_ms is None
    assert result.escape_time is None


def test_empty_map_spec_exposure_is_none():
    """maps.yaml 全空时 exposure 返回 (None, [])。"""
    s = _settings()
    geo = GeoContext(map_name="de_mirage", cfg=s.geom, spec=config.MapSpec(name="de_mirage"), vc=None)
    assert geo.exposure((0.0, 0.0, 0.0)) == (None, [])
    assert geo.escape_time((0.0, 0.0, 0.0)) is None


def test_build_death_windows_without_blinds():
    """缺 blinds 表的 demo 不炸。"""
    ctx = _tiny_context()
    spans = windows.plan_spans([500], 64, _settings().windows)
    result = deaths.build_death_windows(ctx, 111, spans, _tiny_snap())
    assert isinstance(result, list)


# --------------------------------------------------------------------------- fixtures


def _minimal_window() -> deaths.DeathWindow:
    return deaths.DeathWindow(
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
        my_yaw=0.0,
        killer_yaw=180.0,
        my_pitch=0.0,
        killer_pitch=0.0,
        my_speed=0.0,
        killer_speed=0.0,
        my_crouched=False,
        killer_crouched=False,
    )


def _tiny_context():
    from csc.demo_io import DemoContext

    tables = {
        "kills": pl.DataFrame(
            [{"tick": 500, "attacker_steamid": 222, "attacker_name": "k", "victim_steamid": 111, "victim_name": "m", "weapon": "ak47", "headshot": False}],
            schema={"tick": pl.Int64, "attacker_steamid": pl.UInt64, "attacker_name": pl.Utf8, "victim_steamid": pl.UInt64, "victim_name": pl.Utf8, "weapon": pl.Utf8, "headshot": pl.Boolean},
        ),
        "players": pl.DataFrame(
            [{"steamid": 111, "name": "m", "side": "ct"}, {"steamid": 222, "name": "k", "side": "t"}],
            schema={"steamid": pl.UInt64, "name": pl.Utf8, "side": pl.Utf8},
        ),
    }
    return DemoContext(key="x", path=__import__("pathlib").Path("x.dem"), map_name="de_test", tick_rate=64, tables=tables, death_frames=pl.DataFrame(), meta={})


def _tiny_snap() -> pl.DataFrame:
    return pl.DataFrame(
        [{"tick": 500, "steamid": 111, "name": "m", "side": "ct", "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "velocity": 0.0, "health": 100, "armor": 100, "is_crouched": False, "is_blinded": False}],
        schema={c: (pl.Int64 if c in ("tick", "health", "armor") else pl.UInt64 if c == "steamid" else pl.Boolean if c.startswith("is_") else pl.Float64 if c in ("x", "y", "z", "yaw", "pitch", "velocity") else pl.Utf8) for c in ("tick", "steamid", "name", "side", "x", "y", "z", "yaw", "pitch", "velocity", "health", "armor", "is_crouched", "is_blinded")},
    )
