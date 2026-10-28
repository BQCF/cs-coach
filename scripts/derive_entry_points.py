"""M2.5：从职业 demo 自动聚类出各地图的区域中心 / 敌方进入点 / 掩体点。

只依赖客观数据（kills + bomb 种包点），不靠手标；demo 无种包记录时退回
BOMBSITE_SEED 手写种子兜底：

* 区域归属  —— 按 T 方 bomb 落点 + 击杀事件的空间分离，把包点分成 A/B 两坨；
* 进入点    —— 该区域内「击杀者（进攻方）」击杀时的位置做 DBSCAN 聚类，
               = 敌方从哪出现/进攻，供 exposure / crossfire 用；
* 掩体点    —— 该区域内「受害者（防守方）」被击杀时的位置聚类中，靠包点、
               出现频次高的点 = 常用防守/掩体位，供 escape_time 用。

产出是 config/points.yaml 的一段（可被 csc maps apply / merge_points 直接吃）。
运行：python scripts/derive_entry_points.py --map de_inferno [--demo <path>]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402

# --------------------------------------------------------------------------- demo 选择

DEFAULT_DEMOS = {
    "de_inferno": "vitality_vs_g2_inferno.dem",
    "de_mirage": "vitality_vs_falcons_mirage.dem",
    "de_ancient": "furia_vs_vitality_ancient.dem",
}

# 每张图包点的已知大致中心（Hammer，用于初始 A/B 归属，聚类会收敛到实际重心）
BOMBSITE_SEED = {
    "de_inferno": {"A": (1220.0, -300.0, 30.0), "B": (-1600.0, 2400.0, 120.0)},
    "de_mirage": {"A": (1100.0, -700.0, -70.0), "B": (-1600.0, -600.0, -80.0)},
    "de_ancient": {"A": (1500.0, -1500.0, 100.0), "B": (-2200.0, -1300.0, 100.0)},
}


# --------------------------------------------------------------------------- 聚类工具


def cluster_points(points: np.ndarray, eps: float = 220.0, min_cluster: int = 3) -> list[tuple[tuple[float, float, float], int]]:
    """单连通 DBSCAN（用 scipy 层次聚类近似），返回 [(中心xyz, 点数)]，按点数降序。"""
    if len(points) == 0:
        return []
    pts = np.asarray(points, dtype=float)
    if len(pts) == 1:
        return [((pts[0][0], pts[0][1], pts[0][2]), 1)]
    # 半径 eps 的单连通：用最短距离层次聚类在阈值 eps 处切簇
    Z = linkage(pts, method="single", metric="euclidean")
    labels = fcluster(Z, t=eps, criterion="distance")
    clusters: list[tuple[tuple[float, float, float], int]] = []
    for lab in set(labels):
        mask = labels == lab
        if mask.sum() < min_cluster:
            continue
        core = pts[mask]
        center = core.mean(axis=0)
        clusters.append(((float(center[0]), float(center[1]), float(center[2])), int(mask.sum())))
    clusters.sort(key=lambda c: -c[1])
    return clusters


def assign_region(pos, seeds: dict[str, tuple[float, float, float]]) -> str:
    best, best_d = None, None
    for name, c in seeds.items():
        d = math.dist(pos, c)
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best


def tick_to_round(tick: int, rounds: list[dict]) -> int | None:
    """tick -> round_num（rounds 表含 start_tick/end_tick）。"""
    for rd in rounds:
        if rd["start_tick"] <= tick <= rd["end_tick"]:
            return rd["round_num"]
    return None


# --------------------------------------------------------------------------- 主流程


def derive(map_name: str, demo_path: Path, *, n_entry: int = 5, n_cover: int = 3) -> dict:
    from awpy import Demo

    print(f"[{map_name}] 解析 {demo_path.name} ...")
    d = Demo(str(demo_path))
    d.load("kills", "bomb", "rounds")

    # 包点权威坐标：用 finish_plant 的 bombsite 位置做 seed（比手写准得多）
    bomb = d.bomb.to_dicts()
    plants = [r for r in bomb if r.get("event") == "finish_plant" and r.get("bombsite") in ("A", "B")]
    seeds: dict[str, tuple[float, float, float]] = {}
    for site in ("A", "B"):
        rows = [p for p in plants if p["bombsite"] == site]
        if rows:
            cx = sum(p["x"] for p in rows) / len(rows)
            cy = sum(p["y"] for p in rows) / len(rows)
            cz = sum(p["z"] for p in rows) / len(rows)
            seeds[site] = (cx, cy, cz)
    if not seeds:
        seeds = dict(BOMBSITE_SEED.get(map_name, {}))
        print(f"  [warn] demo 无 finish_plant，退回手写 seed {seeds}")
    else:
        print(f"  包点 seed（来自 finish_plant）: { {k: tuple(round(v,0) for v in c) for k,c in seeds.items()} }")

    # 按「这回合 C4 种在哪个点」把击杀归到 A/B，而非按 victim 位置最近 seed
    rounds = d.rounds.to_dicts()
    round_site: dict[int, str] = {}  # round_num -> 'A'/'B'
    for p in plants:
        rnum = tick_to_round(p["tick"], rounds)
        if rnum is not None:
            round_site[rnum] = p["bombsite"]
    print(f"  已知包点的回合: {len(round_site)} / {len(rounds)}")

    kills = d.kills.to_dicts()
    kills = [k for k in kills if k.get("attacker_side") and k.get("victim_side")]
    print(f"  有效击杀 {len(kills)} 条")

    entry_by_region: dict[str, list] = defaultdict(list)
    cover_by_region: dict[str, list] = defaultdict(list)

    for k in kills:
        vp = (float(k["victim_x"]), float(k["victim_y"]), float(k["victim_z"]))
        ap = (float(k["attacker_x"]), float(k["attacker_y"]), float(k["attacker_z"]))
        # 优先按回合归属（tick->round->种包点）；该回合没种包时按 victim 位置最近 seed 兜底
        tick = k.get("tick")
        rnum = tick_to_round(tick, rounds) if tick is not None else None
        region = round_site.get(rnum) if rnum is not None else None
        if region is None:
            region = assign_region(vp, seeds)
        entry_by_region[region].append(ap)
        cover_by_region[region].append(vp)

    result: dict = {"map": map_name, "regions": {}, "entry_points": {}, "cover_points": {}}

    for region in sorted(seeds):
        entry_pts = np.array(entry_by_region.get(region, []))
        cover_pts = np.array(cover_by_region.get(region, []))

        # 区域中心 = 包点 seed（权威），而非交火重心
        center = np.array(seeds[region])

        entry_clusters = cluster_points(entry_pts, eps=160.0, min_cluster=2)
        entry_clusters = [c for c in entry_clusters if math.dist(c[0], center) > 350.0][:n_entry]
        cover_clusters = cluster_points(cover_pts, eps=170.0, min_cluster=2)
        cover_clusters = [c for c in cover_clusters if math.dist(c[0], center) > 250.0][:n_cover]

        result["regions"][region] = {
            "center": [round(float(center[0]), 1), round(float(center[1]), 1), round(float(center[2]), 1)],
            "radius": 1100.0,
        }
        result["entry_points"][region] = [
            {"name": f"{region.lower()}_entry{i+1}", "pos": [round(c[0][0], 1), round(c[0][1], 1), round(c[0][2], 1)]}
            for i, c in enumerate(entry_clusters)
        ]
        result["cover_points"][region] = [
            {"name": f"{region.lower()}_cover{i+1}", "pos": [round(c[0][0], 1), round(c[0][1], 1), round(c[0][2], 1)]}
            for i, c in enumerate(cover_clusters)
        ]
        print(f"  区域 {region}: center={result['regions'][region]['center']} "
              f"entry={len(entry_clusters)} cover={len(cover_clusters)}")
        for ep in result["entry_points"][region]:
            print(f"    entry {ep['name']} @ {ep['pos']}")
        for cp in result["cover_points"][region]:
            print(f"    cover {cp['name']} @ {cp['pos']}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, choices=list(DEFAULT_DEMOS))
    ap.add_argument("--demo", default=None)
    ap.add_argument("--n-entry", type=int, default=5)
    ap.add_argument("--n-cover", type=int, default=3)
    ap.add_argument("--write", action="store_true", help="写入 config/points.yaml（默认只打印）")
    args = ap.parse_args()

    demo = Path(args.demo) if args.demo else ROOT / "data/demos/pro" / DEFAULT_DEMOS[args.map]
    if not demo.exists():
        print(f"demo 不存在：{demo}")
        return 1

    result = derive(args.map, demo, n_entry=args.n_entry, n_cover=args.n_cover)

    if args.write:
        from csc import map_points

        payload = {
            "map": args.map,
            "regions": result["regions"],
            "entry_points": result["entry_points"],
            "cover_points": result["cover_points"],
            "meta": {"source": "derive_entry_points (plant-seeded region clustering)"},
        }
        path, block = map_points.apply_payload(payload, ROOT / "config", map_name=args.map,
                                               source="derive_entry_points")
        print(f"\n已写入 {path} 的 {args.map} 段：{len(block.get('entry_points', {}))} 区域")
    else:
        import json
        print("\n--- payload（加 --write 写入 points.yaml）---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
