"""打点器：payload 组装、HTML 注入、本地保存服务。"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

from csc import config, map_points, maps_edit, radar

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "points-sample.yaml"

DEFAULT_YAML = """\
me: {name: ""}
windows: {pre_s: 2.0, post_s: 0.5, merge_gap_s: 1.0, max_windows: 400}
rules: {}
geom: {}
bench: {}
report: {}
paths: {demos: data/demos, cache: data/cache, output: output, awpy_data: .awpy, tmp: .tmp}
"""
MAPS_YAML = """\
defaults: {gate_min_entry_points: 3, gate_min_cover_points: 1, default_region: A}
maps:
  de_mirage:
    regions:
      A: {center: null, radius: 1100.0}
"""


def _transform():
    return radar.make_transform(
        "de_mirage",
        {"pos_x": -3230.0, "pos_y": 1713.0, "scale": 5.0, "rotate": 0.0},
        version="2000934",
    )


def _config_dir(ws_tmp: Path, *, with_points: bool = True) -> Path:
    cfg = ws_tmp / "config"
    cfg.mkdir()
    (cfg / "default.yaml").write_text(DEFAULT_YAML, encoding="utf-8")
    (cfg / "maps.yaml").write_text(MAPS_YAML, encoding="utf-8")
    if with_points:
        map_points.save_points(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")), cfg)
    return cfg


def test_payload_carries_transform_gate_and_points(ws_tmp: Path) -> None:
    cfg = _config_dir(ws_tmp)
    settings = config.load_settings(cfg)
    payload = maps_edit.build_payload("de_mirage", settings, _transform(), serve=True)

    assert payload["map"] == "de_mirage"
    assert payload["transform"]["scale"] == 5.0
    assert payload["transform"]["pos_x"] == -3230.0
    assert payload["gate"] == {"min_entry": 3, "min_cover": 1}
    assert payload["defaults"]["region"] == "A"
    names = {p["name"] for p in payload["points"]}
    assert {"palace", "chute", "connector", "triple_wall", "bench"} <= names
    assert all(p["source"] == "points" for p in payload["points"])
    assert payload["points_yaml"].endswith("points.yaml")


def test_payload_marks_manual_yaml_points(ws_tmp: Path) -> None:
    """maps.yaml 里手写的点要标成 yaml，编辑器里删不掉才说得清。"""
    cfg = _config_dir(ws_tmp, with_points=False)
    (cfg / "maps.yaml").write_text(
        MAPS_YAML
        + """    entry_points:
      A:
        - name: manual_entry
          pos: [0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )
    settings = config.load_settings(cfg)
    payload = maps_edit.build_payload("de_mirage", settings, _transform(), serve=False)
    sources = {p["name"]: p["source"] for p in payload["points"]}
    assert sources["manual_entry"] == "yaml"


def test_html_embeds_payload_and_is_script_safe() -> None:
    payload = {
        "map": "</script><b>boom",
        "serve": False,
        "transform": {"pos_x": -3230.0, "pos_y": 1713.0, "scale": 5.0, "rotation_deg": 0.0, "width": 1024, "height": 1024},
        "radar": None,
        "regions": {},
        "points": [],
        "gate": {"min_entry": 3, "min_cover": 1},
        "defaults": {"z": 0.0, "region": "A", "kind": "entry"},
        "version": "2000934",
        "lower": False,
        "world_bounds": [-3230.0, 1890.0, -3407.0, 1713.0],
        "points_yaml": "config/points.yaml",
    }
    html = maps_edit.render_html(payload)
    assert "</script><b>boom" not in html
    assert '"pos_x": -3230.0' in html
    assert "api/save" in html


@pytest.mark.skipif(not (ROOT / ".awpy").exists(), reason="本机没有 awpy-data 缓存")
def test_payload_embeds_radar_png_from_cache(ws_tmp: Path) -> None:
    cfg = _config_dir(ws_tmp, with_points=False)
    settings = config.load_settings(cfg)
    tf = radar.load_transform("de_mirage", ROOT / ".awpy", fetch=False)
    payload = maps_edit.build_payload("de_mirage", settings, tf, serve=False)
    assert payload["radar"].startswith("data:image/png;base64,")
    assert payload["version"] == "2000934"


def test_serve_saves_points_and_shuts_down(ws_tmp: Path) -> None:
    cfg = ws_tmp / "config"
    cfg.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    payload = {
        "map": "de_mirage",
        "serve": True,
        "transform": {"pos_x": -3230.0, "pos_y": 1713.0, "scale": 5.0, "rotation_deg": 0.0, "width": 1024, "height": 1024},
        "radar": None,
        "regions": {"A": {"center": None, "radius": 1100.0}},
        "points": [],
        "gate": {"min_entry": 3, "min_cover": 1},
        "defaults": {"z": 0.0, "region": "A", "kind": "entry"},
        "version": "2000934",
        "lower": False,
        "world_bounds": [-3230.0, 1890.0, -3407.0, 1713.0],
        "points_yaml": str(cfg / "points.yaml"),
    }
    thread = threading.Thread(
        target=maps_edit.serve,
        args=(payload, cfg),
        kwargs={"port": port, "open_browser": False, "log": lambda *a, **k: None},
        daemon=True,
    )
    thread.start()

    body = {
        "map": "de_mirage",
        "regions": {"A": {"center": [-600.0, -1600.0, -160.0], "radius": 1100.0}},
        "entry_points": {"A": [{"name": "palace", "pos": [190.0, -1660.0, -184.0]}]},
        "cover_points": {"A": []},
        "exit": True,
    }
    response: dict = {}
    for _ in range(60):
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/save",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            break
        except Exception:  # noqa: BLE001 - 服务还没起来
            time.sleep(0.1)
    else:
        pytest.fail("本地保存服务没起来")

    assert response["ok"] is True
    assert Path(response["path"]) == map_points.points_path(cfg)
    assert any("掩体点" in p for p in response["problems"])

    doc = map_points.load_points(cfg)
    names = {row["name"] for row in doc["maps"]["de_mirage"]["entry_points"]["A"]}
    assert "palace" in names

    thread.join(timeout=5)
    assert not thread.is_alive(), "exit=true 后服务应自行退出"


def test_load_map_file_accepts_three_shapes(ws_tmp: Path) -> None:
    full = ws_tmp / "full.yaml"
    full.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    shape1 = maps_edit.load_map_file(full, "de_mirage")
    assert shape1["map"] == "de_mirage"
    assert len(shape1["entry_points"]["A"]) == 3

    block = ws_tmp / "block.yaml"
    block.write_text(
        "regions: {A: {center: [0, 0, 0], radius: 900}}\nentry_points: {A: [{name: e1, pos: [1, 2, 3]}]}\n",
        encoding="utf-8",
    )
    shape2 = maps_edit.load_map_file(block, "de_mirage")
    assert shape2["entry_points"]["A"][0]["name"] == "e1"

    payload = ws_tmp / "payload.json"
    payload.write_text(
        json.dumps({"map": "de_mirage", "entry_points": {"A": [{"name": "j", "pos": [0, 0, 0]}]}}),
        encoding="utf-8",
    )
    shape3 = maps_edit.load_map_file(payload, "de_mirage")
    assert shape3["entry_points"]["A"][0]["name"] == "j"
