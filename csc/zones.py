"""位置体系（N2，计划 §4）：把「A/B 两个 1100u 大圈」细化成 20–40 个具名 zone。

* ``Zone``：具名区域（name / center / radius / site / aliases）
* ``zone_of(zones, pos)``：返回 (zone 名, 到中心距离)；超出所有半径时给最近 zone 加 ``~`` 前缀
* ``suggest_zones(map_name, ...)``：用职业落位点做 DBSCAN，生成候选 zone 草案
* ``load_zones / save_zones``：读写 ``config/zones.yaml``（独立于 maps.yaml，后者只管进入点/掩体）

zone 语义与 ``MapSpec.region_of`` 的大圆**不同**：region 只服务 entry_points 选择，
zone 是「你在哪个具体点位死的」。别混用、别改 region_of。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from .config import Vec3, dist2

ZONES_NAME = "zones.yaml"


@dataclass(frozen=True)
class Zone:
    name: str
    center: tuple[float, float]
    radius: float
    site: str = ""
    aliases: tuple[str, ...] = ()


# --------------------------------------------------------------------------- 读写


def zones_path(config_dir: Path) -> Path:
    return Path(config_dir) / ZONES_NAME


def load_zones(config_dir: Path, map_name: str) -> list[Zone]:
    """读 zones.yaml 里某图的 zone 列表；没有返回空。"""
    import yaml

    p = zones_path(config_dir)
    if not p.exists():
        return []
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = ((doc.get("maps") or {}).get(map_name) or {}).get("zones") or []
    out: list[Zone] = []
    for z in raw:
        c = z.get("center") or [0, 0]
        out.append(Zone(
            name=str(z.get("name") or "zone"),
            center=(float(c[0]), float(c[1])),
            radius=float(z.get("radius") or 300),
            site=str(z.get("site") or ""),
            aliases=tuple(z.get("aliases") or ()),
        ))
    return out


def save_zones(config_dir: Path, map_name: str, zones: Sequence[Zone], *, note: str = "") -> Path:
    """原子写 zones.yaml（保留其它图）。"""
    import os

    import yaml

    p = zones_path(config_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {}
    if p.exists():
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    doc.setdefault("meta", {"version": 1, "note": note})
    maps = doc.setdefault("maps", {})
    maps[map_name] = {
        "zones": [
            {"name": z.name, "aliases": list(z.aliases), "center": [z.center[0], z.center[1]],
             "radius": z.radius, "site": z.site}
            for z in zones
        ]
    }
    tmp = p.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- 归属


def zone_of(zones: Sequence[Zone], pos: Vec3) -> tuple[str, float]:
    """返回 (zone 名, 到中心距离)。超出所有半径时给最近 zone 加 ~ 前缀。"""
    if not zones:
        return "unknown", float("inf")
    xy = (float(pos[0]), float(pos[1]))
    best = min(zones, key=lambda z: dist2(xy, (z.center[0], z.center[1])))
    d = dist2(xy, (best.center[0], best.center[1]))
    return (best.name if d <= best.radius else f"~{best.name}"), d


# --------------------------------------------------------------------------- 候选生成


def suggest_zones(points: Sequence[Vec3], *, n: int = 32, eps: float = 220.0,
                  min_samples: int = 30) -> list[Zone]:
    """用落位点做 DBSCAN，返回候选 zone 草案（按点数降序，名字待人工改）。"""
    if not points:
        return []
    xy = [(float(p[0]), float(p[1])) for p in points]
    try:
        from sklearn.cluster import DBSCAN  # noqa: PLC0415

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    except Exception:  # noqa: BLE001 - 没 sklearn 就退回简单网格聚类
        labels = _grid_cluster(xy, eps)

    clusters: dict[int, list[tuple[float, float]]] = {}
    for (x, y), lab in zip(xy, labels):
        if lab < 0:
            continue
        clusters.setdefault(int(lab), []).append((x, y))

    zones: list[Zone] = []
    for lab, pts in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        r = max(dist2((cx, cy), p) for p in pts)
        r = min(max(r, 120.0), 420.0)
        zones.append(Zone(name=f"zone_{lab:02d}", center=(round(cx, 1), round(cy, 1)),
                          radius=round(r, 1)))
        if len(zones) >= n:
            break

    # 覆盖补齐：给落位点密集但没成簇的空白区补粗网格 zone，避免 zone_of 返回超远距离
    zones.extend(_coverage_fill(xy, zones, n - len(zones)))
    return zones


def _coverage_fill(xy, existing: list[Zone], budget: int) -> list[Zone]:
    """在有落位点、但离现有 zone 都 >350u 的地方补一个粗 zone，提升归属质量。"""
    if budget <= 0 or not xy:
        return []
    grid = 480.0
    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for x, y in xy:
        cells.setdefault((int(x // grid), int(y // grid)), []).append((x, y))

    def near_existing(cx, cy):
        return any(dist2((cx, cy), (z.center[0], z.center[1])) <= z.radius + 350 for z in existing)

    fills: list[Zone] = []
    for (gx, gy), pts in sorted(cells.items(), key=lambda kv: -len(kv[1])):
        if len(fills) >= budget:
            break
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if near_existing(cx, cy):
            continue
        fills.append(Zone(name=f"zone_fill{len(fills):02d}", center=(round(cx, 1), round(cy, 1)),
                          radius=300.0))
    return fills


def _grid_cluster(xy: list[tuple[float, float]], eps: float) -> list[int]:
    """无 sklearn 时的简易网格聚类（按 eps 网格分桶）。"""
    labels = []
    buckets: dict[tuple[int, int], int] = {}
    for x, y in xy:
        key = (int(x // eps), int(y // eps))
        if key not in buckets:
            buckets[key] = len(buckets)
        labels.append(buckets[key])
    return labels
