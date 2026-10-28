"""``config/points.yaml``：地图坐标库（``csc maps edit`` 的产物）。

分工：

* ``config/maps.yaml`` —— 人写的注释型配置（区域意图、TODO、门禁），工具**从不改写**；
* ``config/points.yaml`` —— 工具产出的坐标，覆盖写、可重生成；
* ``config.parse_maps`` 把两者合并 -> ``MapSpec``，下游 geometry/metrics 无感知。

这样"我只管点位置"成立：点完保存，指标立刻用得上，且不会把手工注释冲掉。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

POINTS_NAME = "points.yaml"
MAX_POINTS_PER_REGION = 32
WORLD_LIMIT = 20000.0
Z_LIMIT = 5000.0

HEADER = """\
# 自动生成：由 `csc maps edit --map <map>` 写出，请勿手工编辑。
# 手工注释型配置（区域意图 / 门禁 / TODO）在 config/maps.yaml，本文件只存坐标。
"""


class PointsError(RuntimeError):
    """坐标数据结构不合法。"""


@dataclass(frozen=True)
class PointRow:
    """扁平化的一个点（``maps show`` 输出用）。"""

    map_name: str
    kind: str  # entry / cover / region
    region: str
    name: str
    pos: tuple[float, float, float] | None
    radius: float | None = None


# --------------------------------------------------------------------------- 读写


def points_path(config_dir: Path) -> Path:
    return Path(config_dir) / POINTS_NAME


def load_points(config_dir: Path) -> dict[str, Any]:
    """读 points.yaml；不存在返回空骨架。格式坏了就抛出（别悄悄丢数据）。"""
    path = points_path(config_dir)
    if not path.exists():
        return {"meta": {}, "maps": {}}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PointsError(f"points.yaml 解析失败（{path}）：{exc}") from exc
    if not isinstance(doc, dict):
        raise PointsError(f"points.yaml 顶层必须是映射：{path}")
    doc.setdefault("meta", {})
    doc.setdefault("maps", {})
    return doc


def dump_points(doc: Mapping[str, Any]) -> str:
    body = yaml.safe_dump(dict(doc), allow_unicode=True, sort_keys=False, default_flow_style=None)
    return HEADER + body


def save_points(doc: Mapping[str, Any], config_dir: Path) -> Path:
    """原子写：先写临时文件再替换，避免半截文件把配置搞坏。"""
    path = points_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dump_points(doc), encoding="utf-8")
    os.replace(tmp, path)
    return path


def map_block(doc: Mapping[str, Any], map_name: str) -> dict[str, Any]:
    maps = doc.setdefault("maps", {}) if isinstance(doc, dict) else {}
    block = maps.get(map_name)
    if not isinstance(block, dict):
        block = {"regions": {}, "entry_points": {}, "cover_points": {}}
        maps[map_name] = block
    block.setdefault("regions", {})
    block.setdefault("entry_points", {})
    block.setdefault("cover_points", {})
    return block


# --------------------------------------------------------------------------- 校验 / 归一化


def _num(value: Any, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PointsError(f"{what} 不是数字：{value!r}") from exc


def _vec3(value: Any, what: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise PointsError(f"{what} 需要 [x, y, z]，实际 {value!r}")
    vec = [_num(v, what) for v in value]
    if abs(vec[0]) > WORLD_LIMIT or abs(vec[1]) > WORLD_LIMIT:
        raise PointsError(f"{what} 的 x/y 超出合理范围（±{WORLD_LIMIT:.0f}）：{vec}")
    if abs(vec[2]) > Z_LIMIT:
        raise PointsError(f"{what} 的 z 超出合理范围（±{Z_LIMIT:.0f}）：{vec}")
    return [round(vec[0], 1), round(vec[1], 1), round(vec[2], 1)]


def normalize_payload(payload: Mapping[str, Any], *, map_name: str | None = None) -> dict[str, Any]:
    """把编辑器 POST 上来的 JSON 归一化成 points.yaml 的 map 段。"""
    if not isinstance(payload, Mapping):
        raise PointsError("payload 必须是对象")
    name = str(payload.get("map") or map_name or "").strip()
    if not name:
        raise PointsError("缺少 map 名称")

    regions_out: dict[str, dict[str, Any]] = {}
    for region, spec in (payload.get("regions") or {}).items():
        region_key = str(region).strip()
        if not region_key:
            raise PointsError("区域名不能为空")
        spec = spec or {}
        entry: dict[str, Any] = {}
        if spec.get("center"):
            entry["center"] = _vec3(spec["center"], f"{region_key}.center")
        radius = spec.get("radius")
        if radius is not None:
            radius_v = _num(radius, f"{region_key}.radius")
            if radius_v <= 0:
                raise PointsError(f"{region_key}.radius 必须为正")
            entry["radius"] = round(radius_v, 1)
        regions_out[region_key] = entry

    def _points(raw: Any, kind: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for region, items in (raw or {}).items():
            region_key = str(region).strip()
            rows: list[dict[str, Any]] = []
            used: dict[str, int] = {}
            for idx, item in enumerate(items or ()):
                if isinstance(item, Mapping):
                    pos_raw = item.get("pos")
                    label = str(item.get("name") or "").strip()
                else:  # 允许简写 [[x, y, z], ...]
                    pos_raw, label = item, ""
                pos = _vec3(pos_raw, f"{region_key}.{kind}[{idx}].pos")
                label = label or f"{kind}_{idx + 1}"
                if label in used:
                    used[label] += 1
                    label = f"{label}_{used[label]}"
                else:
                    used[label] = 1
                rows.append({"name": label, "pos": pos})
            if len(rows) > MAX_POINTS_PER_REGION:
                raise PointsError(
                    f"{region_key}.{kind} 有 {len(rows)} 个点，超过上限 {MAX_POINTS_PER_REGION}（是不是点歪了）"
                )
            if rows:
                out[region_key] = rows
        return out

    return {
        "regions": regions_out,
        "entry_points": _points(payload.get("entry_points"), "entry"),
        "cover_points": _points(payload.get("cover_points"), "cover"),
        "meta": {
            "radar": payload.get("meta", {}).get("radar") if isinstance(payload.get("meta"), Mapping) else None,
            "transform": payload.get("meta", {}).get("transform") if isinstance(payload.get("meta"), Mapping) else None,
        },
    }


def validate_payload(payload: Mapping[str, Any], *, map_name: str | None = None) -> list[str]:
    """不抛异常的体检：返回问题清单（编辑器保存前展示给用户）。"""
    problems: list[str] = []
    try:
        norm = normalize_payload(payload, map_name=map_name)
    except PointsError as exc:
        return [str(exc)]

    entries = {r: len(p) for r, p in norm["entry_points"].items()}
    covers = {r: len(p) for r, p in norm["cover_points"].items()}
    regions = set(norm["regions"]) | set(entries) | set(covers)
    if not regions:
        problems.append("一个点都还没打")
    for region in sorted(regions):
        if entries.get(region, 0) == 0:
            problems.append(f"区域 {region}: 没有敌方进入点（exposure 会为 null）")
        if covers.get(region, 0) == 0:
            problems.append(f"区域 {region}: 没有掩体点（escape_time 会为 null）")
        if not norm["regions"].get(region, {}).get("center"):
            problems.append(f"区域 {region}: 没有区域中心（落位/分位归属会退化）")
    return problems


# --------------------------------------------------------------------------- 合并 / 汇总


def upsert_map(doc: dict[str, Any], map_name: str, payload: Mapping[str, Any], *, source: str = "csc maps edit") -> dict[str, Any]:
    """把一张图的 payload 写进 points 文档（覆盖该图的坐标，保留其它图）。"""
    norm = normalize_payload(payload, map_name=map_name)
    block = map_block(doc, map_name)
    block["regions"].update(norm["regions"])
    for kind in ("entry_points", "cover_points"):
        for region, rows in norm[kind].items():
            block[kind][region] = rows
    meta = doc.setdefault("meta", {})
    meta.setdefault("maps", {})
    map_meta = dict(meta["maps"].get(map_name) or {})
    map_meta.update({k: v for k, v in norm["meta"].items() if v is not None})
    map_meta["source"] = source
    meta["maps"][map_name] = map_meta
    return doc


def apply_payload(payload: Mapping[str, Any], config_dir: Path, *, map_name: str | None = None, source: str = "csc maps edit") -> tuple[Path, dict[str, Any]]:
    """读-合并-写，返回 (points.yaml 路径, 本次写入的 map 段)。"""
    doc = load_points(config_dir)
    name = str(payload.get("map") or map_name or "").strip() if isinstance(payload, Mapping) else str(map_name or "")
    if not name:
        raise PointsError("缺少 map 名称")
    upsert_map(doc, name, payload, source=source)
    path = save_points(doc, config_dir)
    return path, map_block(doc, name)


def summarize(doc: Mapping[str, Any]) -> list[PointRow]:
    rows: list[PointRow] = []
    for map_name, block in (doc.get("maps") or {}).items():
        block = block or {}
        for region, spec in (block.get("regions") or {}).items():
            center = spec.get("center") if isinstance(spec, Mapping) else None
            centers = tuple(float(v) for v in center) if center else None
            radius = float(spec["radius"]) if isinstance(spec, Mapping) and spec.get("radius") else None
            rows.append(PointRow(str(map_name), "region", str(region), str(region), centers, radius))  # type: ignore[arg-type]
        for kind, key in (("entry", "entry_points"), ("cover", "cover_points")):
            for region, items in (block.get(key) or {}).items():
                for item in items or ():
                    pos = tuple(float(v) for v in item["pos"]) if isinstance(item, Mapping) and item.get("pos") else None
                    name = str(item.get("name") or kind) if isinstance(item, Mapping) else kind
                    rows.append(PointRow(str(map_name), kind, str(region), name, pos))  # type: ignore[arg-type]
    return rows
