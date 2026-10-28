"""第一人称视角渲染（M6）：从 awpy 的 .mesh 碰撞几何渲染玩家死亡瞬间的视角。

mesh 格式（实测）：`AWMH` 魔数 + 12 字节头（顶点数 u32, 索引数 u32, 面数 u32）
+ float32 顶点数组 (x,y,z) + u32 索引数组（每 3 个 = 一个三角面）。

渲染：正交/透视投影三角面到眼位视角，按距离排序消隐。标注：
* 红 = 敌人方向 / 危险来路
* 绿 = 应该预瞄的方向
* 黄 = 你的准星
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import yaw_forward


@dataclass
class Mesh:
    verts: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32


def load_mesh(path: Path) -> Mesh:
    data = Path(path).read_bytes()
    assert data[:4] == b"AWMH", f"bad magic: {data[:4]!r}"
    n_verts = int.from_bytes(data[8:12], "little")
    n_faces = int.from_bytes(data[12:16], "little")
    off = 16
    verts = np.frombuffer(data[off : off + n_verts * 12], dtype=np.float32).reshape(-1, 3)
    off += n_verts * 12
    faces = np.frombuffer(data[off : off + n_faces * 12], dtype=np.uint32).reshape(-1, 3)
    return Mesh(verts=verts, faces=faces)


def render_first_person(
    mesh: Mesh,
    eye: tuple[float, float, float],
    yaw_deg: float,
    pitch_deg: float = 0.0,
    *,
    fov_deg: float = 90.0,
    size: int = 720,
    max_dist: float = 2500.0,
    marks: list | None = None,
    annotate: str = "",
):
    """渲染第一人称视角。marks = [(pos, color, label, kind)]。

    kind: 'enemy'（红箭头）/ 'aim'（绿十字准星）/ 'mark'（黄圈）。
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # 中文字体
    from matplotlib import font_manager  # noqa: PLC0415

    for _f in font_manager.fontManager.ttflist:
        if any(k in _f.name for k in ("Microsoft YaHei", "SimHei", "DengXian", "Noto Sans CJK", "WenQuanYi")):
            plt.rcParams["font.sans-serif"] = [_f.name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    # 相机基向量
    fwd = np.array(yaw_forward(yaw_deg), dtype=np.float32)
    fwd3 = np.array([fwd[0], fwd[1], 0.0], dtype=np.float32)
    # pitch：正=向下看
    pitch_r = math.radians(pitch_deg)
    fwd3[2] = -math.sin(pitch_r)  # 向下为正 pitch
    fwd3[:2] *= math.cos(pitch_r)
    fwd3 /= np.linalg.norm(fwd3)
    right = np.array([-fwd[1], fwd[0], 0.0], dtype=np.float32)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd3)
    up /= np.linalg.norm(up)

    eye_arr = np.array(eye, dtype=np.float32)

    # 转到相机坐标
    rel = mesh.verts - eye_arr
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ fwd3  # 深度

    # 只保留前方
    faces = mesh.faces
    # 三角面
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    z0, z1, z2 = cam_z[v0], cam_z[v1], cam_z[v2]
    depth = (z0 + z1 + z2) / 3.0
    mask = (depth > 5.0) & (depth < max_dist)
    # 距离衰减着色
    f0, f1, f2 = faces[mask][:, 0], faces[mask][:, 1], faces[mask][:, 2]
    d = depth[mask]
    order = np.argsort(-d)  # 由远及近
    f0, f1, f2, d = f0[order], f1[order], f2[order], d[order]

    # 投影
    focal = (size / 2) / math.tan(math.radians(fov_deg) / 2)

    def proj(ix):
        z = np.maximum(cam_z[ix], 1.0)
        return (size / 2 + cam_x[ix] * focal / z, size / 2 - cam_y[ix] * focal / z)

    x0, y0 = proj(f0)
    x1, y1 = proj(f1)
    x2, y2 = proj(f2)

    # 深度着色：近亮远暗
    shade = np.clip(1.0 - d / max_dist, 0.08, 0.92)
    color = np.stack([shade * 0.42, shade * 0.55, shade * 0.38], axis=1)  # 暗绿

    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.axis("off")
    ax.set_facecolor("#222")

    from matplotlib.collections import PolyCollection  # noqa: PLC0415

    polys = np.stack([np.stack([x0, y0], axis=1),
                      np.stack([x1, y1], axis=1),
                      np.stack([x2, y2], axis=1)], axis=1)
    pc = PolyCollection(polys, facecolors=color, edgecolors="none")
    ax.add_collection(pc)

    # 标注
    for mark in (marks or []):
        pos, mcolor, label, kind = mark
        rel_m = np.array(pos, dtype=np.float32) - eye_arr
        mz = float(rel_m @ fwd3)
        if mz <= 1.0:
            continue
        mx = size / 2 + float(rel_m @ right) * focal / mz
        my = size / 2 - float(rel_m @ up) * focal / mz
        if not (-100 < mx < size + 100 and -100 < my < size + 100):
            continue
        if kind == "aim":
            # 绿色十字 = 推荐瞄准点
            ax.plot([mx], [my], marker="+", ms=30, mew=4, color="#2ecc71")
            ax.annotate(label, (mx, my), color="#0b3d1f", fontsize=11,
                        xytext=(40, -22), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#2ecc71", ec="#0b3d1f", lw=1.2, alpha=0.95),
                        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.8))
        elif kind == "enemy":
            ax.plot([mx], [my], marker="x", ms=20, mew=4, color="#e74c3c")
            ax.annotate(label, (mx, my), color="#7b1c1c", fontsize=11,
                        xytext=(-46, 34), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#e74c3c", ec="#7b1c1c", lw=1.2, alpha=0.95),
                        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.8))
        else:
            ax.plot([mx], [my], marker="o", ms=12, color=mcolor)

    # 你的准星 = 屏幕中心（黄十字）
    cx, cy = size / 2, size / 2
    ax.plot([cx], [cy], marker="+", ms=22, mew=3, color="#f1c40f")
    ax.annotate("你的准星", (cx, cy), color="#7a5c00", fontsize=11,
                xytext=(-78, -24), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f1c40f", ec="#7a5c00", lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#f1c40f", lw=1.8))

    if annotate:
        ax.annotate(annotate, (0.02, 0.02), xycoords="axes fraction", color="white", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#000", ec="none", alpha=0.6))

    return fig, ax


# --------------------------------------------------------------------------- 生成接线
#
# 语义（用户拍板 2026-10-25）：红 X = 敌人死时位置，绿十字 = 敌人**冒头处**
# （killer_emerge_tick 时刻的位置）——教「你该预瞄的是他冒出来的角落」。
# 注意：这是 mesh 示意图，不是游戏截图（.dem 不含画面数据）。


def find_mesh(awpy_root: Path, map_name: str) -> Path | None:
    """找地图碰撞 mesh：``<awpy_root>/<version>/geometry/<map>.mesh``。"""
    for version_dir in sorted(Path(awpy_root).glob("*"), reverse=True):
        p = version_dir / "geometry" / f"{map_name}.mesh"
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- 真截图工作流
#
# .dem 不含游戏画面，真截图只能来自游戏引擎（进游戏回放截一张）。
# 但相机参数（eye/yaw/pitch）demo 里是精确的——所以可以把标注**投影到真截图上**，
# 得到「真游戏画面 + 教学标注」。


def project_marks(
    marks: list,
    eye: tuple[float, float, float],
    yaw_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
    *,
    fov_43: float = 90.0,
) -> list[tuple[float, float, str, str, str]]:
    """把世界坐标标注投影到真实截图的像素坐标（CS 相机模型）。

    CS 的 fov_cs_debug 是 **4:3 基准的水平 FOV**，宽屏按 aspect/(4/3) 放大
    （16:9 ≈ 106.3°），不处理这个宽屏截图会整体偏位。返回 [(mx, my, color, label, kind)]，
    相机后方 / 图外的点自动丢弃。
    """
    fwd = np.array(yaw_forward(yaw_deg), dtype=np.float64)
    fwd3 = np.array([fwd[0], fwd[1], 0.0], dtype=np.float64)
    pitch_r = math.radians(float(pitch_deg or 0.0))
    fwd3[2] = -math.sin(pitch_r)  # 与 render_first_person 同一约定：正 pitch = 向下看
    fwd3[:2] *= math.cos(pitch_r)
    fwd3 /= np.linalg.norm(fwd3)
    right = np.array([-fwd[1], fwd[0], 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd3)
    up /= np.linalg.norm(up)

    aspect = width / max(1, height)
    hfov = 2.0 * math.atan(math.tan(math.radians(fov_43) / 2.0) * (aspect / (4.0 / 3.0)))
    focal = (width / 2.0) / math.tan(hfov / 2.0)

    eye_arr = np.array(eye, dtype=np.float64)
    out: list[tuple[float, float, str, str, str]] = []
    for mark in marks:
        pos, color, label, kind = mark
        rel = np.array(pos, dtype=np.float64) - eye_arr
        mz = float(rel @ fwd3)
        if mz <= 1.0:
            continue
        mx = width / 2.0 + float(rel @ right) * focal / mz
        my = height / 2.0 - float(rel @ up) * focal / mz
        if not (-100 < mx < width + 100 and -100 < my < height + 100):
            continue
        out.append((mx, my, color, label, kind))
    return out


def find_real_shot(shots_dir: Path, tick: int) -> Path | None:
    """找用户放进来的真实截图：文件名含 tick 的 png/jpg（如 shots/121523.png）。"""
    d = Path(shots_dir)
    if not d.is_dir():
        return None
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and str(tick) in p.stem:
            return p
    return None


def overlay_real_shot(
    image_path: Path,
    marks: list,
    eye: tuple[float, float, float],
    yaw_deg: float,
    pitch_deg: float,
    *,
    annotate: str = "",
    fov_43: float = 90.0,
):
    """在真实游戏截图上叠标注（红X=敌人死时位置、绿+=冒头处、黄+=你的准星）。"""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # 中文字体（标注是中文，跟 render_first_person 同一套）
    from matplotlib import font_manager  # noqa: PLC0415

    for _f in font_manager.fontManager.ttflist:
        if any(k in _f.name for k in ("Microsoft YaHei", "SimHei", "DengXian", "Noto Sans CJK", "WenQuanYi")):
            plt.rcParams["font.sans-serif"] = [_f.name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    img = plt.imread(str(image_path))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img)
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    for mx, my, color, label, kind in project_marks(marks, eye, yaw_deg, pitch_deg, w, h, fov_43=fov_43):
        if kind == "aim":
            ax.plot([mx], [my], marker="+", ms=30, mew=4, color="#2ecc71")
            ax.annotate(label, (mx, my), color="#0b3d1f", fontsize=11,
                        xytext=(40, -22), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#2ecc71", ec="#0b3d1f", lw=1.2, alpha=0.95),
                        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.8))
        elif kind == "enemy":
            ax.plot([mx], [my], marker="x", ms=20, mew=4, color="#e74c3c")
            ax.annotate(label, (mx, my), color="#7b1c1c", fontsize=11,
                        xytext=(-46, 34), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#e74c3c", ec="#7b1c1c", lw=1.2, alpha=0.95),
                        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.8))
        else:
            ax.plot([mx], [my], marker="o", ms=12, color=color)

    # 黄十字=你的准星（真截图本身就是这个视角，图心即准星）
    ax.plot([w / 2], [h / 2], marker="+", ms=22, mew=3, color="#f1c40f")
    ax.annotate("你的准星", (w / 2, h / 2), color="#7a5c00", fontsize=11,
                xytext=(-78, -24), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f1c40f", ec="#7a5c00", lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#f1c40f", lw=1.8))

    if annotate:
        ax.annotate(annotate, (0.02, 0.02), xycoords="axes fraction", color="white", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#000", ec="none", alpha=0.6))
    return fig, ax


def write_replay_guide(shots: list[dict], demo_name: str, out_dir: Path) -> Path:
    """生成「进游戏截真图」操作指南；截好的图放 shots/<tick>.png 重跑即自动替换示意图。"""
    lines = [
        "# 进游戏看真实画面（真截图工作流）",
        "",
        "示意图是地图模型渲染，`.dem` 里**没有游戏画面数据**——真截图只能由游戏引擎渲染。",
        "好消息：相机参数（站位/朝向/俯仰）demo 里是精确的，你截的真图回来后，",
        "红 X / 绿 + / 准星标注会**自动投影叠加**到你的截图上。",
        "",
        "## 操作（每张图 30 秒）",
        "",
        f"1. 打开 CS2，控制台播放 demo：`playdemo {demo_name}`（或用完美平台的回放）；",
        "2. 对下面每个 tick：控制台 `demo_gototick <tick>`，跳过去就是你死亡时的第一人称；",
        "3. 按 F12（Steam 截图）或控制台 `jpeg` 截图；",
        f"4. 截图改名为 `<tick>.png` 放进本目录的 `shots/` 文件夹；",
        "5. 重跑 `csc report`（或 `csc analyze`），对应卡片自动换成「真截图 + 标注」版。",
        "",
        "> 建议用原生分辨率截图（别用 4:3 拉伸），标注投影才能对准。",
        "",
        "| 卡片 | tick | 控制台命令 |",
        "| --- | --- | --- |",
    ]
    for shot in shots:
        kind_cn = "站位问题" if shot["kind"] == "position" else "预瞄问题"
        lines.append(f"| {kind_cn} | {shot['tick']} | `demo_gototick {shot['tick']}` |")
    path = Path(out_dir) / "replay_guide.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plan_fp_shots(records, geo, cfg, *, max_each: int = 2) -> list[dict]:
    """挑典型死亡点规划示意图（只算数据不渲染，便于单测）。

    * 站位问题 = fault=position*，预瞄问题 = fault=preaim，各取最严重的 ``max_each`` 个；
    * 找不到冒头点 / 没有 yaw 的死亡点跳过——绿十字是这张图的全部意义。
    """
    from .deaths import killer_emerge_tick  # 局部导入避免循环
    from .windows import frame_at

    groups: dict[str, list[tuple[float, dict]]] = {"position": [], "preaim": []}
    for rec in records:
        w, v = rec.window, rec.verdict
        fault = rec.fault.fault if rec.fault is not None else ""
        kind = "position" if fault.startswith("position") else ("preaim" if fault == "preaim" else "")
        if not kind or w.killer_pos is None or w.my_yaw is None:
            continue
        if geo is None or w.killer_track is None or w.killer_steamid is None:
            continue
        try:
            emerge = killer_emerge_tick(w, geo, cfg)
        except Exception:  # noqa: BLE001
            emerge = None
        row = frame_at(w.killer_track, w.killer_steamid, emerge) if emerge is not None else None
        try:
            emerge_pos = (float(row["x"]), float(row["y"]), float(row["z"])) if row is not None else None
        except (KeyError, TypeError, ValueError):
            emerge_pos = None
        if emerge_pos is None:
            continue

        m = rec.metrics
        if kind == "position":
            score = float(((m.exposure or 0) if m else 0) + ((m.crossfire or 0) if m else 0) * 2
                          + ((m.escape_time or 0.0) if m else 0.0))
        else:
            score = float(rec.duel.my_aim_off_at_see_deg or 0.0) if rec.duel is not None else 0.0

        eye_fn = getattr(geo, "eye", None)
        eye = eye_fn(w.my_pos, w.my_crouched) if eye_fn else w.my_pos
        ev: list[str] = []
        if m is not None:
            if m.exposure is not None:
                ev.append(f"暴露给 {m.exposure} 个敌人")
            if m.escape_time is not None:
                ev.append(f"撤回掩体 {m.escape_time} 秒")
        annotate = (f"{'站位' if kind == 'position' else '预瞄'}问题 | {rec.zone or ''}\n"
                    f"{'、'.join(ev)}\ntick {w.death_tick} | {v.label}")
        groups[kind].append((score, {
            "tick": w.death_tick,
            "kind": kind,
            "eye": eye,
            "yaw": w.my_yaw,
            "pitch": w.my_pitch or 0.0,
            "marks": [
                (w.killer_pos, "#e74c3c", "敌人在这", "enemy"),
                (emerge_pos, "#2ecc71", "该瞄这里", "aim"),
            ],
            "annotate": annotate,
            "file": f"fp_{kind}_{w.death_tick}.png",
        }))

    out: list[dict] = []
    for kind in ("position", "preaim"):
        out += [item for _score, item in sorted(groups[kind], key=lambda kv: -kv[0])[:max_each]]
    return out


def generate_fp_shots(
    records,
    geo,
    cfg,
    out_dir: Path,
    map_name: str,
    awpy_root: Path,
    *,
    max_each: int = 2,
    demo_name: str = "",
    real_shots_dir: Path | None = None,
) -> list[str]:
    """生成 ``fp_*.png`` 并写 ``replay_guide.md``，返回生成的文件名。

    **真截图优先**：用户按指南把游戏截图放进 ``shots/<tick>.png`` 后，自动在真图上
    投影叠加标注；没有真截图才退回 mesh 示意图。示意图是佐证不是主产物，
    mesh 缺失 / 渲染失败都静默跳过，不影响报告主流程。
    """
    if geo is None:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_fp_shots(records, geo, cfg, max_each=max_each)
    if not planned:
        return []
    if demo_name:
        try:
            write_replay_guide(planned, demo_name, out_dir)
        except Exception:  # noqa: BLE001 - 指南写不出不影响出图
            pass

    shots_dir = Path(real_shots_dir) if real_shots_dir else out_dir / "shots"
    mesh = None
    mesh_path = find_mesh(awpy_root, map_name)
    made: list[str] = []
    for shot in planned:
        try:
            real = find_real_shot(shots_dir, shot["tick"])
            if real is not None:
                fig, _ax = overlay_real_shot(
                    real, shot["marks"], shot["eye"], shot["yaw"], shot["pitch"],
                    annotate=shot["annotate"],
                )
            else:
                if mesh is None:
                    if mesh_path is None:
                        continue
                    mesh = load_mesh(mesh_path)
                fig, _ax = render_first_person(
                    mesh, shot["eye"], shot["yaw"], shot["pitch"],
                    marks=shot["marks"], annotate=shot["annotate"],
                )
            fig.savefig(out_dir / shot["file"], dpi=100)
            import matplotlib.pyplot as plt  # noqa: PLC0415

            plt.close(fig)
            made.append(shot["file"])
        except Exception:  # noqa: BLE001 - 单张失败不拖垮整批
            continue
    return made
