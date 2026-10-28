"""雷达像素坐标 <-> Hammer 世界坐标。

变换公式来自 awpy-data 的 ``map_data.json``（字段 ``pos_x`` / ``pos_y`` / ``scale``）：

.. code-block:: text

    radar_x = (world_x - pos_x) / scale
    radar_y = (pos_y   - world_y) / scale

``pos_x/pos_y`` 是雷达图**左上角**对应的世界坐标，``scale`` 是「每像素多少 Hammer unit」。
以 de_mirage 为例：``pos_x=-3230, pos_y=1713, scale=5.0``、图 1024×1024，
覆盖世界范围 x∈[-3230, 1890]、y∈[-3407, 1713]，与地图实际尺寸一致。

依赖说明：**不需要 awpy**。资源直接取自 awpy-data 的 GitHub release
（``map_data.json`` + ``images.zip`` 里的 ``radars/<map>.png``），缓存在
``.awpy/<version>/``——和 awpy 自己的缓存布局一致，装好 awpy 后可以直接共用。
"""

from __future__ import annotations

import io
import json
import math
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_TAG = "2000934"  # 与 .awpy/2000934 对齐（2026-07 的 CS2 ClientVersion）
DEFAULT_BASE = "https://github.com/pnxenopoulos/awpy-data/releases/download"
MINI_ASSETS = ("manifest.json", "map_data.json", "images.zip")
RADAR_NOTES = {
    "rotate": "该图 radar 带旋转标记，像素↔世界换算未在实机上验证；先在编辑器里目测并可用 --rotate-deg 修正",
}


class RadarError(RuntimeError):
    """拿不到 map_data / 雷达图，或参数不合法。"""


# --------------------------------------------------------------------------- 变换


@dataclass(frozen=True)
class MapTransform:
    """一张图的雷达图与像素↔世界变换。"""

    map_name: str
    pos_x: float
    pos_y: float
    scale: float
    width: int = 1024
    height: int = 1024
    rotation_deg: float = 0.0
    version: str = ""
    radar: Path | None = None
    lower: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)

    # -- 校验 ---------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.scale is None or float(self.scale) == 0.0:
            raise RadarError(f"{self.map_name}: scale 不能为 0")
        if self.width <= 0 or self.height <= 0:
            raise RadarError(f"{self.map_name}: 雷达图尺寸非法 {self.width}x{self.height}")

    @property
    def ready(self) -> bool:
        return self.radar is not None and Path(self.radar).exists()

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max)：雷达图覆盖的世界范围，用来 sanity check。"""
        return (
            self.pos_x,
            self.pos_x + self.width * self.scale,
            self.pos_y - self.height * self.scale,
            self.pos_y,
        )

    # -- 旋转 ---------------------------------------------------------------
    def _rotate(self, px: float, py: float, deg: float) -> tuple[float, float]:
        if not deg:
            return px, py
        cx, cy = self.width / 2.0, self.height / 2.0
        a = math.radians(deg)
        dx, dy = px - cx, py - cy
        return (cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a))

    # -- 换算 ---------------------------------------------------------------
    def radar_to_world(self, px: float, py: float) -> tuple[float, float]:
        """像素 (px, py)（左上原点、y 向下）-> 世界 (x, y)。"""
        cx, cy = self._rotate(float(px), float(py), -self.rotation_deg)
        return (self.pos_x + cx * self.scale, self.pos_y - cy * self.scale)

    def world_to_radar(self, x: float, y: float) -> tuple[float, float]:
        """世界 (x, y) -> 像素 (px, py)。"""
        px = (float(x) - self.pos_x) / self.scale
        py = (self.pos_y - float(y)) / self.scale
        return self._rotate(px, py, self.rotation_deg)

    def clamp_pixel(self, px: float, py: float) -> tuple[float, float]:
        return (min(max(px, 0.0), float(self.width - 1)), min(max(py, 0.0), float(self.height - 1)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
            "width": self.width,
            "height": self.height,
            "version": self.version,
            "lower": self.lower,
            "radar": str(self.radar) if self.radar else None,
        }


def png_size(path: Path) -> tuple[int, int]:
    """不依赖 PIL：直接读 PNG 的 IHDR。"""
    with Path(path).open("rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        raise RadarError(f"不是 PNG：{path}")
    return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))


# --------------------------------------------------------------------------- 资源定位


def awpy_data_roots(explicit: Path | None = None) -> list[Path]:
    """候选 awpy-data 根目录，按优先级排列。"""
    roots: list[Path] = []
    if explicit is not None:
        roots.append(Path(explicit))
    env = os.environ.get("AWPY_DATA_DIR")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / ".awpy")
    seen: set[str] = set()
    uniq: list[Path] = []
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(root)
    return uniq


def version_dirs(root: Path) -> list[Path]:
    """``<root>/<version>/`` 列表，新版本优先。"""
    root = Path(root)
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "map_data.json").exists()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def find_version_dir(awpy_root: Path, version: str | None = None) -> Path | None:
    root = Path(awpy_root)
    if version:
        candidate = root / str(version)
        return candidate if (candidate / "map_data.json").exists() else None
    dirs = version_dirs(root)
    return dirs[0] if dirs else None


def read_map_data(version_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(version_dir) / "map_data.json"
    if not path.exists():
        raise RadarError(f"缺少 map_data.json：{version_dir}（先跑 `csc maps fetch`）")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RadarError(f"map_data.json 结构异常：{path}")
    return raw


def available_maps(awpy_root: Path) -> list[str]:
    version_dir = find_version_dir(awpy_root)
    if version_dir is None:
        return []
    return sorted(read_map_data(version_dir))


def find_radar(version_dir: Path, map_name: str, *, lower: bool = False) -> Path | None:
    name = f"{map_name}_lower.png" if lower else f"{map_name}.png"
    for candidate in (Path(version_dir) / "radars" / name, Path(version_dir) / name):
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- 下载


def _download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "cs-coach/0.1 (+csc maps)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 固定 https 源
        data = resp.read()
    dest.write_bytes(data)
    return dest


def latest_tag(base: str = DEFAULT_BASE, *, timeout: float = 30.0) -> str:
    """查最新 awpy-data 版本；失败时退回 DEFAULT_TAG。"""
    api = "https://api.github.com/repos/pnxenopoulos/awpy-data/releases/latest"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "cs-coach/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        tag = str(payload.get("tag_name") or "").strip()
        return tag or DEFAULT_TAG
    except Exception:  # noqa: BLE001 - 离线就退回固定 tag
        return DEFAULT_TAG


def fetch_assets(
    awpy_root: Path,
    *,
    tag: str = DEFAULT_TAG,
    maps: Iterable[str] = (),
    base: str = DEFAULT_BASE,
    log: Any = None,
) -> Path:
    """下载 ``map_data.json`` + ``images.zip``，并解出需要的雷达图。

    只拉 ~3MB（不是几百 MB 的完整资源包），够编辑器用；碰撞网格仍由 ``awpy get`` 负责。
    """
    say = log or (lambda *_a, **_k: None)
    version_dir = Path(awpy_root) / str(tag)
    version_dir.mkdir(parents=True, exist_ok=True)
    for name in MINI_ASSETS:
        target = version_dir / name
        if target.exists() and target.stat().st_size > 0:
            say(f"  已有 {name}")
            continue
        say(f"  下载 {name} ...")
        _download(f"{base}/{tag}/{name}", target)

    zip_path = version_dir / "images.zip"
    wanted = {f"{m}.png" for m in maps} | {f"{m}_lower.png" for m in maps}
    if maps:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if Path(member).name not in wanted:
                    continue
                out = version_dir / "radars" / Path(member).name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(member))
                say(f"  解出 {out.relative_to(version_dir)}")
    return version_dir


# --------------------------------------------------------------------------- 主入口


def make_transform(
    map_name: str,
    raw: Mapping[str, Any],
    *,
    version: str = "",
    radar: Path | None = None,
    lower: bool = False,
    rotation_deg: float | None = None,
) -> MapTransform:
    if map_name not in raw and not {"pos_x", "pos_y", "scale"} <= set(raw):
        raise RadarError(f"map_data.json 里没有 {map_name}")
    entry = raw.get(map_name, raw) if "pos_x" not in raw else raw
    try:
        pos_x, pos_y, scale = float(entry["pos_x"]), float(entry["pos_y"]), float(entry["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RadarError(f"{map_name}: map_data 缺少 pos_x/pos_y/scale（{entry!r}）") from exc

    width = height = 1024
    if radar is not None and Path(radar).exists():
        width, height = png_size(Path(radar))

    rotate = entry.get("rotate")
    if rotation_deg is None:
        # awpy-data 里 rotate 是 0.0 / 1.0 / null；按 90° 一档处理，符号未验证
        rotation_deg = 90.0 if rotate else 0.0

    return MapTransform(
        map_name=map_name,
        pos_x=pos_x,
        pos_y=pos_y,
        scale=scale,
        width=width,
        height=height,
        rotation_deg=float(rotation_deg or 0.0),
        version=version,
        radar=Path(radar) if radar else None,
        lower=lower,
        raw=entry,
    )


def load_transform(
    map_name: str,
    awpy_root: Path,
    *,
    version: str | None = None,
    lower: bool = False,
    fetch: bool = False,
    rotation_deg: float | None = None,
    log: Any = None,
) -> MapTransform:
    """拿一张图的变换 + 雷达图；``fetch=True`` 时自动补下缺失的迷你资源。"""
    say = log or (lambda *_a, **_k: None)
    version_dir = find_version_dir(Path(awpy_root), version)
    if version_dir is None:
        if not fetch:
            raise RadarError(
                f"没有本地地图数据（{Path(awpy_root)}）。跑 `csc maps fetch --map {map_name}` 下载约 3MB 的 map_data.json + 雷达图"
            )
        tag = version or latest_tag()
        version_dir = fetch_assets(awpy_root, tag=tag, maps=[map_name], log=say)

    raw = read_map_data(version_dir)
    if map_name not in raw:
        raise RadarError(f"{version_dir.name}/map_data.json 不含 {map_name}；可用：{', '.join(sorted(raw))}")

    radar = find_radar(version_dir, map_name, lower=lower)
    if radar is None and fetch:
        fetch_assets(awpy_root, tag=version_dir.name, maps=[map_name], log=say)
        radar = find_radar(version_dir, map_name, lower=lower)
    if radar is None:
        say(f"  警告：{map_name} 没有雷达图（lower={lower}），编辑器会退化成纯网格背景")

    if raw[map_name].get("rotate") and rotation_deg is None:
        say(f"  提示：{RADAR_NOTES['rotate']}")

    return make_transform(
        map_name,
        raw[map_name],
        version=version_dir.name,
        radar=radar,
        lower=lower,
        rotation_deg=rotation_deg,
    )
