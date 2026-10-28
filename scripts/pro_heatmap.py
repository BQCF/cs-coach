"""M4：职业落位热图 —— 把 pro demo 的落位点密度叠到雷达底图上。

用 csc.radar.load_transform + world_to_radar 换算（与打点器同源）。
按图 + 区域出热图，供跟"你的站位/死亡点"对比分位。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体统一在这设（标题/colorbar/text 全走 rcParams），否则 colorbar 标签是豆腐块
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from csc import benchmark, demo_io, radar  # noqa: E402
from csc.config import bootstrap_env, load_settings  # noqa: E402

DEMOS = {
    "de_inferno": "vitality_vs_g2_inferno.dem",
    "de_mirage": "vitality_vs_falcons_mirage.dem",
    "de_ancient": "furia_vs_vitality_ancient.dem",
}


def heatmap(map_name: str, out_dir: Path, *, buy_type: str = "full", phase: str = "early") -> Path:
    s = load_settings()
    bootstrap_env()
    ctx, _ = demo_io.parse_demo(ROOT / "data/demos/pro" / DEMOS[map_name], s.paths.cache)

    pts = benchmark.collect_pro_positions(ctx, round_phase=phase, buy_type=buy_type, n_alive=5,
                                          side="counter-terrorist", holding_only=True)
    if pts.is_empty():
        raise SystemExit(f"{map_name}: 无落位点")

    version_dir = radar.find_version_dir(ROOT / ".awpy")
    img_path = radar.find_radar(version_dir, map_name)
    img = plt.imread(img_path)
    transform = radar.load_transform(map_name, ROOT / ".awpy")

    # 落位点 -> 雷达像素
    xs, ys = [], []
    for r in pts.iter_rows(named=True):
        px, py = transform.world_to_radar(r["x"], r["y"])
        xs.append(px)
        ys.append(py)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, origin="upper", alpha=0.9)
    hb = ax.hexbin(np.array(xs), np.array(ys), gridsize=40, cmap="YlOrRd", mincnt=2, alpha=0.7,
                   linewidths=0.1, edgecolors="none")
    fig.colorbar(hb, ax=ax, label="落位频次")
    ax.set_title(f"{map_name} 职业落位热图 — {buy_type}buy {phase} (CT 防守方)", fontfamily="Microsoft YaHei")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pro_heatmap_{map_name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}  (点数={pts.height})")
    return out


def main() -> int:
    out_dir = ROOT / "output" / "m4"
    for m in DEMOS:
        heatmap(m, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
