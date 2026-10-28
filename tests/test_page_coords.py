"""JS 与 Python 的坐标换算必须完全一致。

页面里的 ``radarToWorld`` / ``worldToRadar`` 是从 :class:`csc.radar.MapTransform`
翻译过去的第二份实现——两处口径漂移会让"点出来的位置"和指标用的位置对不上，
所以这里用 node 跑一遍页面里的真实代码做交叉验证（没 node 就跳过）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from csc import maps_page, radar

NODE = shutil.which("node")
PIXELS = [(0.0, 0.0), (512.0, 300.5), (1023.0, 1023.0), (200.0, 800.0)]
WORLDS = [(-600.0, -1600.0), (0.0, 0.0), (1500.0, -2800.0)]


def _coordinate_block() -> str:
    html = maps_page.TEMPLATE
    start = html.index("/* ---------- 坐标换算")
    end = html.index("/* ---------- 视图")
    return html[start:end]


def _run_js(transform: radar.MapTransform, ws_tmp: Path) -> dict[str, list[list[float]]]:
    x = {
        "pos_x": transform.pos_x,
        "pos_y": transform.pos_y,
        "scale": transform.scale,
        "rotation_deg": transform.rotation_deg,
        "width": transform.width,
        "height": transform.height,
    }
    js = (
        f"const X = {json.dumps(x)};\n"
        + _coordinate_block()
        + "\nconst fs = require('fs');\n"
        + "const out = {forward: [], back: []};\n"
        + f"{json.dumps(PIXELS)}.forEach(([px, py]) => out.forward.push(radarToWorld(px, py)));\n"
        + f"{json.dumps(WORLDS)}.forEach(([x, y]) => out.back.push(worldToRadar(x, y)));\n"
        + "fs.writeFileSync(process.argv[2], JSON.stringify(out));\n"
    )
    script = ws_tmp / "coords.js"
    script.write_text(js, encoding="utf-8")
    out_file = ws_tmp / "coords.json"
    subprocess.run([NODE, str(script), str(out_file)], check=True, timeout=30)
    return json.loads(out_file.read_text(encoding="utf-8"))


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过 JS/Python 一致性检查")
def test_js_matches_python_without_rotation(ws_tmp: Path) -> None:
    tf = radar.make_transform("de_mirage", {"pos_x": -3230.0, "pos_y": 1713.0, "scale": 4.6})
    got = _run_js(tf, ws_tmp)
    for (px, py), (jx, jy) in zip(PIXELS, got["forward"]):
        assert tf.radar_to_world(px, py) == pytest.approx((jx, jy), abs=1e-6)
    for (x, y), (jpx, jpy) in zip(WORLDS, got["back"]):
        assert tf.world_to_radar(x, y) == pytest.approx((jpx, jpy), abs=1e-6)


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过 JS/Python 一致性检查")
def test_js_matches_python_with_rotation(ws_tmp: Path) -> None:
    tf = radar.make_transform(
        "de_dust2",
        {"pos_x": -2476.0, "pos_y": 3239.0, "scale": 4.4},
        rotation_deg=90.0,
    )
    got = _run_js(tf, ws_tmp)
    for (px, py), (jx, jy) in zip(PIXELS, got["forward"]):
        assert tf.radar_to_world(px, py) == pytest.approx((jx, jy), abs=1e-6)
    for (x, y), (jpx, jpy) in zip(WORLDS, got["back"]):
        assert tf.world_to_radar(x, y) == pytest.approx((jpx, jpy), abs=1e-6)
