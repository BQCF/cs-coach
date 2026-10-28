"""绘图：死亡点 2D 回放图 + 落位热图。

* matplotlib 用 ``Agg`` 后端（无显示器，CLI 里跑不会弹窗）；
* 延迟导入 matplotlib / awpy：``MPLCONFIGDIR`` 必须在 import matplotlib 前设好，
  由 ``config.bootstrap_env()`` 在 CLI 入口保证；
* MVP 直接画世界坐标散点（等比例），雷达底图/awpy.plot.frame 是 M5 的增强项。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .deaths import DeathRecord


def _plt():
    import matplotlib  # noqa: PLC0415 - 延迟导入

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # 中文字体：报告图里的标签是中文，默认 DejaVu 会渲染成豆腐块
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _draw_arrow(ax, x, y, yaw_deg, length, color, label):
    rad = math.radians(yaw_deg)
    ax.annotate(
        "",
        xy=(x + length * math.cos(rad), y + length * math.sin(rad)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


def death_frame(
    record: DeathRecord,
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """画一个死亡点的 2D 回放：我(红X) + 击杀者(蓝O) + 朝向箭头。"""
    plt = _plt()
    w, v = record.window, record.verdict
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="datalim")

    ax.plot(w.my_pos[0], w.my_pos[1], "rx", markersize=12, label=f"我({v.label})")
    if w.my_yaw is not None:
        _draw_arrow(ax, w.my_pos[0], w.my_pos[1], w.my_yaw, 120, "red", "我的朝向")
    if w.killer_pos is not None:
        ax.plot(w.killer_pos[0], w.killer_pos[1], "bo", markersize=10, label="击杀者")
        if w.killer_yaw is not None:
            _draw_arrow(ax, w.killer_pos[0], w.killer_pos[1], w.killer_yaw, 120, "blue", "击杀者朝向")

    m = record.metrics
    subtitle = (
        f"gap={m.first_seen_gap_ms}ms exposure={m.exposure} escape={m.escape_time}s"
        if m is not None
        else ""
    )
    ax.set_title(title or f"{v.label}  {v.note}  {subtitle}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def heatmap(points: Sequence[tuple[float, float]], out_path: Path, *, title: str = "落位热图") -> Path:
    """世界坐标的 2D 直方热图（MVP，无雷达底图）。"""
    plt = _plt()
    if not points:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title + "（无数据）")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(xs, ys, bins=40, cmap="hot")
    fig.colorbar(h[3], ax=ax, label="频次")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path

def radar_chart(labels: Sequence[str], values: Sequence[float], out_path: Path, *,
                title: str = "能力雷达") -> Path:
    """六维雷达图（径向图）。values 已归一化 0-1，越靠外越好。"""
    plt = _plt()
    import math  # noqa: PLC0415

    n = len(labels)
    angles = [i / n * 2 * math.pi for i in range(n)]
    vals = list(values) + [values[0]] if values else [0.0]
    angs = angles + [angles[0]] if angles else [0.0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angs, vals, "o-", linewidth=2)
    ax.fill(angs, vals, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontfamily="Microsoft YaHei")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontfamily="Microsoft YaHei")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path

def fault_distribution_chart(labels: Sequence[str], counts: Sequence[int], out_path: Path, *,
                             title: str = "责任分布") -> Path:
    """fault 责任分布柱状图。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"position": "#e74c3c", "preaim": "#e67e22", "aim": "#3498db", "info": "#9b59b6", "none": "#95a5a6"}
    cl = [colors.get(l, "#7f8c8d") for l in labels]
    ax.bar(labels, counts, color=cl)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=11)
    ax.set_title(title, fontfamily="Microsoft YaHei")
    ax.set_ylabel("死亡次数", fontfamily="Microsoft YaHei")
    ax.grid(axis="y", alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def preaim_distribution_chart(mine: Sequence[float], theirs: Sequence[float], out_path: Path, *,
                              title: str = "预瞄偏差分布") -> Path:
    """预瞄偏差直方图：你 vs 击杀你的对手。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = 15
    if mine:
        ax.hist(mine, bins=bins, alpha=0.65, label=f"你（中位 {sorted(mine)[len(mine)//2]:.0f}°）", color="#3498db")
    if theirs:
        ax.hist(theirs, bins=bins, alpha=0.65, label=f"对手（中位 {sorted(theirs)[len(theirs)//2]:.0f}°）", color="#e74c3c")
    ax.axvline(10, color="gray", ls="--", lw=1, label="10° 参考线")
    ax.set_xlabel("准星偏差（度）", fontfamily="Microsoft YaHei")
    ax.set_ylabel("次数", fontfamily="Microsoft YaHei")
    ax.set_title(title, fontfamily="Microsoft YaHei")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def duel_damage_chart(ticks: Sequence[int], my_dmg: Sequence[int], his_dmg: Sequence[int],
                      out_path: Path, *, title: str = "对枪伤害对比") -> Path:
    """每次 aim 死亡的我 vs 他伤害对比（分组柱状）。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = range(len(ticks))
    w = 0.35
    ax.bar([x - w / 2 for x in xs], my_dmg, w, label="你造成的伤害", color="#3498db")
    ax.bar([x + w / 2 for x in xs], his_dmg, w, label="他造成的伤害", color="#e74c3c")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax.set_xlabel("死亡 tick", fontfamily="Microsoft YaHei")
    ax.set_ylabel("伤害", fontfamily="Microsoft YaHei")
    ax.set_title(title, fontfamily="Microsoft YaHei")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def death_scatter_chart(xs: Sequence[float], ys: Sequence[float], faults: Sequence[str],
                        out_path: Path, *, title: str = "死亡位置分布") -> Path:
    """死亡位置散点（按 fault 着色）——可视化你死在哪。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"position": "#e74c3c", "preaim": "#e67e22", "aim": "#3498db", "info": "#9b59b6", "none": "#95a5a6"}
    seen = set()
    for x, y, f in zip(xs, ys, faults):
        c = colors.get(f, "#7f8c8d")
        ax.scatter(x, y, c=c, s=70, label=f if f not in seen else None, edgecolors="k", linewidths=0.5)
        seen.add(f)
    ax.set_title(title, fontfamily="Microsoft YaHei")
    ax.set_xlabel("X (Hammer)", fontfamily="Microsoft YaHei")
    ax.set_ylabel("Y (Hammer)", fontfamily="Microsoft YaHei")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path

def radar_dots(df, map_name: str, out_path: Path, *, title: str = "位置分布总览") -> Path | None:
    """独立雷达打点图（雷达底图 + 死亡点按 fault 着色）。供报告「位置分布总览」。"""
    if df is None or df.is_empty() or not {"my_x", "my_y", "fault"} <= set(df.columns):
        return None
    try:
        from . import radar as radar_mod
        from .config import PROJECT_ROOT, bootstrap_env

        bootstrap_env()
        transform = radar_mod.load_transform(map_name, PROJECT_ROOT / ".awpy")
        if not transform.ready:
            return None
    except Exception:  # noqa: BLE001
        return None

    plt = _plt()
    img = plt.imread(Path(transform.radar))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, origin="upper")
    colors = {"position": "#e74c3c", "aim": "#3498db", "preaim": "#e67e22",
              "info": "#9b59b6", "none": "#95a5a6"}
    seen = set()
    for r in df.iter_rows(named=True):
        px, py = transform.world_to_radar(r.get("my_x", 0), r.get("my_y", 0))
        fault = r.get("fault", "none")
        c = colors.get(fault, "#7f8c8d")
        sev = 1 + (r.get("exposure") or 0) * 0.4 + (r.get("crossfire") or 0) * 0.3 + min(2, (r.get("escape_time") or 0) * 0.3)
        ax.scatter(px, py, c=c, s=40 + sev * 22, edgecolors="k", linewidths=0.6,
                   label=fault if fault not in seen else None, zorder=5)
        seen.add(fault)
    ax.set_title(title, fontfamily="Microsoft YaHei")
    ax.legend(loc="lower right", fontsize=9)
    ax.axis("off")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path

