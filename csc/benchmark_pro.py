"""M4 分位对比接入：把死亡点/站位跟职业落位分布对比。

职责：
1. 从 ``data/demos/pro/`` 下**同地图**的职业 demo 收集落位点（collect_pro_positions）；
2. 建职业落位网格 + 点集；
3. 对每个死亡点算 ``cumulative_percentile``（累积分位）+ ``neighborhood_frequency``（邻域频率）；
4. 同经济匹配（same_economy）：只比装备价值差 <= economy_tolerance 的职业回合。

**持久缓存（"训练"）**：collect_pro_positions 要开 demo 逐回合取快照，38 个 demo
反复解析极慢。所以 ``csc train`` 一次性把每张图的职业落位点 + 网格落盘到
``data/cache/pro_bench/<map>/``，之后 ``ProBenchmark`` 直接读缓存，秒级就绪。
demo 集合 / 网格边长变了（签名不符）才自动重训。

职业 demo 缺失时降级：分位为 None，报告标注"缺职业基准"。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from . import benchmark, bench_roles, demo_io
from .bench_roles import cumulative_percentile, neighborhood_frequency, same_economy
from .config import BenchCfg, Settings

# pro demo 文件名里的地图标识（derive_entry_points 的 DEFAULT_DEMOS 同款）
PRO_DEMO_HINTS = {
    "de_inferno": "inferno",
    "de_mirage": "mirage",
    "de_ancient": "ancient",
}

# collect 阶段：early = 回合前 60s 默认展开，mid = 60~120s
DEFAULT_PHASES = ("early", "mid")


# --------------------------------------------------------------------------- 找 demo


def find_pro_demos(settings: Settings, map_name: str) -> list[Path]:
    """按文件名里的地图标识找该图的职业 demo。"""
    hint = PRO_DEMO_HINTS.get(map_name, map_name.replace("de_", ""))
    pro_dir = settings.paths.demos / "pro"
    if not pro_dir.exists():
        return []
    return sorted(p for p in pro_dir.glob("*.dem") if hint in p.name.lower())


def _demo_signature(demos: Sequence[Path], cfg: BenchCfg) -> str:
    """demo 集合 + 采集参数（网格/降采样/架枪阈值）-> 签名；变了说明要重训。"""
    parts = [f"grid={cfg.grid}", f"stride={getattr(cfg, 'sample_stride', 8)}", f"hold={cfg.holding_speed}"]
    for p in sorted(demos):
        st = p.stat()
        parts.append(f"{p.name}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- 缓存路径


def cache_dir(settings: Settings, map_name: str) -> Path:
    return settings.paths.cache / "pro_bench" / map_name


def save_cache(
    settings: Settings,
    map_name: str,
    pts: Sequence[tuple[float, float, float]],
    equips: Sequence[float],
    grid: pl.DataFrame,
    demo_paths: Sequence[Path],
    *,
    phases: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    """把职业落位点 + 网格 + 元数据落盘。"""
    cdir = cache_dir(settings, map_name)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg = settings.bench

    points_df = pl.DataFrame(
        {
            "x": [p[0] for p in pts],
            "y": [p[1] for p in pts],
            "z": [p[2] for p in pts],
            "equipment_value": list(equips),
        },
        schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "equipment_value": pl.Float64},
    )
    points_df.write_parquet(cdir / "points.parquet")
    grid.write_parquet(cdir / "grid.parquet")

    meta = {
        "map_name": map_name,
        "n_points": len(pts),
        "n_grid_cells": grid.height,
        "grid_size": cfg.grid,
        "phases": list(phases),
        "side": "counter-terrorist",
        "holding_speed": cfg.holding_speed,
        "neighborhood_radius": cfg.neighborhood_radius,
        "economy_tolerance": cfg.economy_tolerance,
        "n_demos": len(demo_paths),
        "demos": [p.name for p in sorted(demo_paths)],
        "signature": _demo_signature(demo_paths, cfg),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    (cdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cdir


def load_cache(settings: Settings, map_name: str, demo_paths: Sequence[Path]) -> dict[str, Any] | None:
    """读缓存；签名不符（demo 变了 / 网格变了）返回 None 触发重训。"""
    cdir = cache_dir(settings, map_name)
    meta_path = cdir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") != _demo_signature(demo_paths, settings.bench):
            return None
        points = pl.read_parquet(cdir / "points.parquet")
        grid = pl.read_parquet(cdir / "grid.parquet")
    except Exception:  # noqa: BLE001 - 缓存坏了就重训
        return None
    pts = [(float(r["x"]), float(r["y"]), float(r["z"])) for r in points.iter_rows(named=True)]
    equips = points["equipment_value"].to_list()
    return {"meta": meta, "points": pts, "equips": equips, "grid": grid}


# --------------------------------------------------------------------------- 采集


def collect_map_points(
    settings: Settings,
    map_name: str,
    *,
    phases: Sequence[str] = DEFAULT_PHASES,
    demo_paths: Sequence[Path] | None = None,
    on_demo=None,
) -> tuple[list[tuple[float, float, float]], list[float], list[Path]]:
    """采集一张图的职业落位点（CT、静止架枪点）。返回 (points, equips, 已处理demo)。

    ``on_demo`` 可传回调 (i, total, path, n_added) 用于进度打印。
    """
    cfg = settings.bench
    paths = list(demo_paths) if demo_paths is not None else find_pro_demos(settings, map_name)
    pts: list[tuple[float, float, float]] = []
    equips: list[float] = []
    for i, demo_path in enumerate(paths):
        added = 0
        try:
            ctx, _ = demo_io.parse_demo(demo_path, settings.paths.cache)
            if getattr(ctx, "map_name", "") and ctx.map_name != map_name:
                # 文件名有地图标识但 demo 实际是别的图（极少见），跳过
                if on_demo:
                    on_demo(i + 1, len(paths), demo_path, 0)
                continue
            for phase in phases:
                df = benchmark.collect_pro_positions(
                    ctx,
                    round_phase=phase,
                    side="counter-terrorist",
                    holding_only=True,
                    holding_speed=cfg.holding_speed,
                )
                # 降密度：每 N tick 取 1 个点。定位基准只需空间分布，
                # per-tick 全量（站立 10s = 上千重复点）会撑爆内存并拖慢邻域扫描。
                stride = max(1, int(getattr(cfg, "sample_stride", 8)))
                if stride > 1 and not df.is_empty() and "tick" in df.columns:
                    df = df.filter((pl.col("tick") % stride) == 0)
                for r in df.iter_rows(named=True):
                    pts.append((float(r["x"]), float(r["y"]), float(r["z"])))
                    equips.append(float(r.get("equipment_value") or 0.0))
                    added += 1
        except Exception:  # noqa: BLE001 - 单个 demo 失败不影响整体
            pass
        if on_demo:
            on_demo(i + 1, len(paths), demo_path, added)
    return pts, equips, paths


# --------------------------------------------------------------------------- 训练


def train_map(
    settings: Settings,
    map_name: str,
    *,
    phases: Sequence[str] = DEFAULT_PHASES,
    force: bool = False,
    on_demo=None,
    verbose: bool = False,
) -> dict[str, Any]:
    """训练一张图的职业基准：采集落位 -> 建网格 -> 落盘缓存。返回 meta。"""
    paths = find_pro_demos(settings, map_name)
    if not paths:
        return {"map_name": map_name, "n_points": 0, "n_demos": 0, "skipped": "no_pro_demos"}

    if not force:
        cached = load_cache(settings, map_name, paths)
        if cached:
            return {**cached["meta"], "cached": True}

    cfg = settings.bench
    pts, equips, _ = collect_map_points(settings, map_name, phases=phases, demo_paths=paths, on_demo=on_demo)
    if not pts:
        return {"map_name": map_name, "n_points": 0, "n_demos": len(paths), "skipped": "no_positions"}
    grid = benchmark.grid_frame(pts, cfg.grid)
    cdir = save_cache(settings, map_name, pts, equips, grid, paths, phases=phases)
    meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
    meta["cached"] = False
    if verbose:
        print(f"[train] {map_name}: {meta['n_points']} 落位点 / {meta['n_grid_cells']} 网格 / {meta['n_demos']} demo -> {cdir}")
    return meta


def train_all(
    settings: Settings,
    *,
    maps: Sequence[str] | None = None,
    phases: Sequence[str] = DEFAULT_PHASES,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """训练所有（或指定）有职业 demo 的地图。"""
    if maps is None:
        # 扫 pro 目录里出现过的地图标识
        pro_dir = settings.paths.demos / "pro"
        found: list[str] = []
        if pro_dir.exists():
            names = " ".join(p.name.lower() for p in pro_dir.glob("*.dem"))
            for map_name, hint in PRO_DEMO_HINTS.items():
                if hint in names:
                    found.append(map_name)
        maps = found

    results: dict[str, dict[str, Any]] = {}
    for map_name in maps:
        if verbose:

            def _mk(m=map_name):
                def cb(i, total, path, added):
                    print(f"  [{m}] {i}/{total} {path.name} +{added}", flush=True)
                return cb

            cb = _mk()
        else:
            cb = None
        try:
            results[map_name] = train_map(settings, map_name, phases=phases, force=force, on_demo=cb, verbose=verbose)
        except Exception as exc:  # noqa: BLE001
            results[map_name] = {"map_name": map_name, "error": str(exc)}
    return results


# --------------------------------------------------------------------------- 查询（懒加载 + 缓存）


class ProBenchmark:
    """同地图职业落位基准（缓存优先，缺失则现场重训并落盘）。"""

    def __init__(self, settings: Settings, map_name: str):
        self.settings = settings
        self.map_name = map_name
        self._points: list[tuple[float, float, float]] | None = None
        self._equip_values: list[float] = []
        self._grid: pl.DataFrame | None = None
        self._meta: dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        paths = find_pro_demos(self.settings, self.map_name)
        if not paths:
            self._points, self._grid, self._meta = [], None, {"skipped": "no_pro_demos"}
            return
        cached = load_cache(self.settings, self.map_name, paths)
        if cached:
            self._points = cached["points"]
            self._equip_values = cached["equips"]
            self._grid = cached["grid"]
            self._meta = cached["meta"]
            return
        # 缓存缺失/过期：现场重训并落盘（下次就走缓存）
        meta = train_map(self.settings, self.map_name, force=True)
        again = load_cache(self.settings, self.map_name, paths)
        if again:
            self._points = again["points"]
            self._equip_values = again["equips"]
            self._grid = again["grid"]
            self._meta = again["meta"]
        else:
            self._points, self._grid, self._meta = [], None, meta

    @property
    def meta(self) -> dict[str, Any]:
        self._load()
        return self._meta

    @property
    def ready(self) -> bool:
        self._load()
        return bool(self._points)

    @property
    def n_points(self) -> int:
        self._load()
        return len(self._points or [])

    def percentile(self, pos: tuple[float, float, float]) -> float | None:
        self._load()
        if self._grid is None:
            return None
        return cumulative_percentile(pos, self._grid, self.settings.bench.grid)

    def neighborhood(self, pos: tuple[float, float, float]) -> float | None:
        self._load()
        if not self._points:
            return None
        return neighborhood_frequency(pos, self._points, self.settings.bench.neighborhood_radius)

    def same_econ_peers(self, equip_value: float) -> int:
        """同经济（装备差<=economy_tolerance）的职业样本数。"""
        self._load()
        tol = self.settings.bench.economy_tolerance
        return sum(1 for e in self._equip_values if same_economy(equip_value, e, tol))


def annotate_pro_percentile(
    records: list[Any],
    settings: Settings,
    map_name: str,
    *,
    equip_lookup: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """给每个死亡点补职业分位列（累积分位 / 邻域频率 / 同经济样本数）。"""
    bench = ProBenchmark(settings, map_name)
    rows: list[dict[str, Any]] = []
    for rec in records:
        w = rec.window
        pos = tuple(w.my_pos)
        pct = bench.percentile(pos) if bench.ready else None
        nb = bench.neighborhood(pos) if bench.ready else None
        equip = (equip_lookup or {}).get(w.my_steamid, 0.0)
        rows.append({
            "pro_cumulative_percentile": round(pct, 4) if pct is not None else None,
            "pro_neighborhood_freq": round(nb, 5) if nb is not None else None,
            "pro_same_econ_samples": bench.same_econ_peers(equip) if bench.ready else 0,
            "pro_benchmark_ready": bench.ready,
        })
    return rows
