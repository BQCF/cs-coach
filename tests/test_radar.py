"""雷达像素 <-> 世界坐标的换算（纯数学，不需要 awpy / 网络）。"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from csc import radar

MIRAGE = {"pos_x": -3230.0, "pos_y": 1713.0, "scale": 5.0, "rotate": 0.0}
ROOT = Path(__file__).resolve().parents[1]


def mirage_transform(**kwargs) -> radar.MapTransform:
    return radar.make_transform("de_mirage", MIRAGE, version="2000934", **kwargs)


def test_top_left_pixel_is_pos_xy() -> None:
    tf = mirage_transform()
    assert tf.radar_to_world(0, 0) == pytest.approx((-3230.0, 1713.0))
    assert tf.world_to_radar(-3230.0, 1713.0) == pytest.approx((0.0, 0.0))


def test_world_bounds_match_map_size() -> None:
    tf = mirage_transform()
    x0, x1, y0, y1 = tf.world_bounds
    assert (x0, y1) == (-3230.0, 1713.0)
    assert x1 == pytest.approx(-3230.0 + 1024 * 5.0)
    assert y0 == pytest.approx(1713.0 - 1024 * 5.0)


@pytest.mark.parametrize("px,py", [(0, 0), (512, 300.5), (1023, 1023), (200, 800)])
def test_round_trip(px: float, py: float) -> None:
    tf = mirage_transform()
    x, y = tf.radar_to_world(px, py)
    assert tf.world_to_radar(x, y) == pytest.approx((px, py), abs=1e-6)


def test_known_world_point_lands_on_expected_pixel() -> None:
    """A 包点中心 (-600, -1600) -> (526, 662.6)，手算可复核。"""
    tf = mirage_transform()
    assert tf.world_to_radar(-600.0, -1600.0) == pytest.approx((526.0, 662.6))
    assert tf.radar_to_world(526.0, 662.6) == pytest.approx((-600.0, -1600.0))


def test_zero_scale_is_rejected() -> None:
    with pytest.raises(radar.RadarError):
        radar.make_transform("de_mirage", {**MIRAGE, "scale": 0.0})


def test_rotation_round_trip() -> None:
    tf = mirage_transform(rotation_deg=90.0)
    px, py = tf.world_to_radar(-600.0, -1600.0)
    assert tf.radar_to_world(px, py) == pytest.approx((-600.0, -1600.0), abs=1e-6)


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00" * ((width + 1) * height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_png_size_reads_ihdr(ws_tmp: Path) -> None:
    png = ws_tmp / "x.png"
    png.write_bytes(_png(320, 200))
    assert radar.png_size(png) == (320, 200)


@pytest.mark.skipif(not (ROOT / ".awpy").exists(), reason="本机没有 awpy-data 缓存")
def test_load_transform_from_local_cache() -> None:
    tf = radar.load_transform("de_mirage", ROOT / ".awpy", fetch=False)
    assert tf.ready
    assert (tf.width, tf.height) == (1024, 1024)
    assert tf.scale == 5.0
    assert tf.version == "2000934"
