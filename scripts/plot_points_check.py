"""把 points.yaml 的坐标叠到雷达底图上，供人工目视核对。

用 csc/radar.py 的 world_to_radar 换算（与打点器同源，避免口径漂移）：
* 红 X   = 敌方进入点 (entry)
* 蓝 ●   = 掩体点 (cover)
* 绿圈   = 区域中心 + 半径
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402

# 中文字体统一在这设（标题/text 全走 rcParams），避免默认 DejaVu 渲染成豆腐块
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from csc import map_points, radar  # noqa: E402
from csc.config import load_settings  # noqa: E402

MAPS = ["de_inferno", "de_mirage", "de_ancient"]


def visualize(map_name: str, out_dir: Path) -> Path:
    s = load_settings()
    spec = s.map_spec(map_name)
    pts = map_points.load_points(ROOT / "config")
    block = (pts.get("maps") or {}).get(map_name) or {}

    version_dir = radar.find_version_dir(ROOT / ".awpy")
    img_path = radar.find_radar(version_dir, map_name)
    if img_path is None:
        raise SystemExit(f"找不到 {map_name} 雷达图")

    img = plt.imread(img_path)
    transform = radar.load_transform(map_name, ROOT / ".awpy")
    to_px = transform.world_to_radar

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, origin="upper")
    ax.set_title(f"{map_name} — 进入点(红X) / 掩体点(蓝●) / 区域中心(绿圈)", fontfamily="Microsoft YaHei")

    # 区域中心 + 半径
    for region, rspec in (block.get("regions") or {}).items():
        center = rspec.get("center")
        if not center:
            continue
        cx, cy = to_px(center[0], center[1])
        radius_u = float(rspec.get("radius") or 1100.0)
        radius_px = radius_u / float(transform.scale)
        circ = patches.Circle((cx, cy), radius_px, fill=False, ec="lime", lw=1.5, ls="--")
        ax.add_patch(circ)
        ax.plot(cx, cy, marker="o", ms=10, mfc="none", mec="lime", mew=2)
        ax.text(cx, cy, f" {region}", color="lime", fontsize=13, fontweight="bold", va="center")

    # 进入点
    for region, items in (block.get("entry_points") or {}).items():
        for it in items:
            x, y, z = it["pos"]
            px, py = to_px(x, y)
            ax.plot(px, py, marker="x", color="red", ms=12, mew=2.5)
            ax.text(px + 6, py - 6, it["name"], color="red", fontsize=8)

    # 掩体点
    for region, items in (block.get("cover_points") or {}).items():
        for it in items:
            x, y, z = it["pos"]
            px, py = to_px(x, y)
            ax.plot(px, py, marker="o", color="deepskyblue", ms=9, mec="white", mew=0.8)
            ax.text(px + 6, py - 6, it["name"], color="deepskyblue", fontsize=8)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"check_{map_name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    out_dir = ROOT / "output" / "point_check"
    for m in MAPS:
        p = visualize(m, out_dir)
        print("saved:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
