"""生成带编号的雷达图（N2 §4.3），供人工照图给 zone 改名。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from csc import radar, zones  # noqa: E402
from csc.config import bootstrap_env, load_settings  # noqa: E402


def draw(map_name: str, out_dir: Path) -> Path:
    s = load_settings()
    bootstrap_env()
    zs = zones.load_zones(s.config_dir, map_name)
    if not zs:
        raise SystemExit(f"{map_name} 无 zone，先跑 csc zones suggest --map {map_name} --write")

    img_path = radar.find_radar(radar.find_version_dir(ROOT / ".awpy"), map_name)
    img = plt.imread(img_path)
    transform = radar.load_transform(map_name, ROOT / ".awpy")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, origin="upper")
    for i, z in enumerate(zs):
        cx, cy = transform.world_to_radar(z.center[0], z.center[1])
        rpix = z.radius / float(transform.scale)
        ax.add_patch(plt.Circle((cx, cy), rpix, fill=False, ec="cyan", lw=1.2))
        ax.text(cx, cy, str(i), color="yellow", fontsize=9, fontweight="bold",
                ha="center", va="center")
    ax.set_title(f"{map_name} zone 编号图（黄数字=zone 序号）")
    ax.axis("off")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{map_name}_zones.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = draw("de_ancient", ROOT / "output" / "zones")
    print("saved", out)
