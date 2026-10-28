"""坐标库：points.yaml 的归一化、合并、门禁。"""

from __future__ import annotations

from pathlib import Path

import pytest

from csc import config, map_points

GATES = {"gate_min_entry_points": 3, "gate_min_cover_points": 1, "default_region": "A"}
PAYLOAD = {
    "map": "de_mirage",
    "regions": {"A": {"center": [-600.0, -1600.0, -160.0], "radius": 1100.0}},
    "entry_points": {
        "A": [
            {"name": "palace", "pos": [190.0, -1660.0, -184.0]},
            {"name": "chute", "pos": [-1100.0, -2300.0, -184.0]},
            {"name": "connector", "pos": [-700.0, -900.0, -184.0]},
        ]
    },
    "cover_points": {"A": [{"name": "triple_wall", "pos": [-450.0, -1850.0, -184.0]}]},
}


def test_unnamed_points_get_stable_names() -> None:
    norm = map_points.normalize_payload(
        {"map": "de_mirage", "entry_points": {"A": [{"pos": [0, 0, 0]}, {"pos": [1, 1, 1]}]}}
    )
    assert [row["name"] for row in norm["entry_points"]["A"]] == ["entry_1", "entry_2"]


def test_duplicate_names_are_suffixed() -> None:
    norm = map_points.normalize_payload(
        {
            "map": "de_mirage",
            "entry_points": {"A": [{"name": "x", "pos": [0, 0, 0]}, {"name": "x", "pos": [1, 1, 1]}]},
        }
    )
    assert [row["name"] for row in norm["entry_points"]["A"]] == ["x", "x_2"]


def test_bad_coordinates_rejected() -> None:
    with pytest.raises(map_points.PointsError):
        map_points.normalize_payload({"map": "de_mirage", "entry_points": {"A": [{"pos": [1e9, 0, 0]}]}})
    with pytest.raises(map_points.PointsError):
        map_points.normalize_payload({"map": "de_mirage", "entry_points": {"A": [{"pos": [0, 0]}]}})


def test_validate_reports_missing_cover() -> None:
    payload = {"map": "de_mirage", "regions": {"A": {"center": [0, 0, 0], "radius": 900}}, "entry_points": PAYLOAD["entry_points"]}
    problems = map_points.validate_payload(payload)
    assert any("掩体点" in p for p in problems)


def test_missing_points_file_is_empty(ws_tmp: Path) -> None:
    assert map_points.load_points(ws_tmp) == {"meta": {}, "maps": {}}


def test_apply_then_merge_satisfies_gate(ws_tmp: Path) -> None:
    path, block = map_points.apply_payload(PAYLOAD, ws_tmp, map_name="de_mirage")
    assert path == map_points.points_path(ws_tmp)
    assert len(block["entry_points"]["A"]) == 3

    doc = map_points.load_points(ws_tmp)
    maps = config.parse_maps({"defaults": GATES})
    merged = config.merge_points(maps, doc, defaults=GATES)
    spec = merged["de_mirage"]
    assert spec.ready()
    assert spec.regions["A"]["center"] == (-600.0, -1600.0, -160.0)
    assert {ep.name for ep in spec.entry_points["A"]} == {"palace", "chute", "connector"}
    assert spec.cover_points["A"][0].name == "triple_wall"


def test_merge_overrides_same_name_without_duplicating() -> None:
    spec = config.MapSpec(name="m", entry_points={"A": (config.EntryPoint("a", (0.0, 0.0, 0.0)),)})
    merged = config.merge_points(
        {"m": spec},
        {"maps": {"m": {"entry_points": {"A": [{"name": "a", "pos": [1, 2, 3]}]}}}},
    )
    items = merged["m"].entry_points["A"]
    assert len(items) == 1
    assert items[0].pos == (1.0, 2.0, 3.0)


def test_merge_keeps_manual_entries_from_maps_yaml() -> None:
    spec = config.MapSpec(
        name="m",
        entry_points={"A": (config.EntryPoint("manual", (5.0, 5.0, 5.0)),)},
        cover_points={"A": (config.CoverPoint("wall", (1.0, 1.0, 1.0)),)},
    )
    merged = config.merge_points(
        {"m": spec},
        {"maps": {"m": {"entry_points": {"A": [{"name": "tool", "pos": [0, 0, 0]}]}}}},
    )
    names = [ep.name for ep in merged["m"].entry_points["A"]]
    assert names == ["manual", "tool"]
    assert merged["m"].cover_points["A"][0].name == "wall"


def test_second_apply_keeps_other_maps(ws_tmp: Path) -> None:
    map_points.apply_payload(PAYLOAD, ws_tmp, map_name="de_mirage")
    map_points.apply_payload(
        {"map": "de_inferno", "entry_points": {"A": [{"name": "ramp", "pos": [1, 2, 3]}]}},
        ws_tmp,
        map_name="de_inferno",
    )
    doc = map_points.load_points(ws_tmp)
    assert set(doc["maps"]) == {"de_mirage", "de_inferno"}
    rows = map_points.summarize(doc)
    assert any(r.map_name == "de_inferno" and r.name == "ramp" for r in rows)
