"""几何与可见性：眼高、烟雾叠加、暴露面、交叉覆盖、退路时间。

``awpy.VisibilityChecker``（>=3.1）只认**静态碰撞几何**：不含烟、不含门、不含穿透。
所以这里在 ``is_visible`` 之上补两层：

* 眼高：demo 里的 x/y/z 是站立点位置（脚底附近），视线查询必须 +64 / +46；
* 烟雾：``smokes`` 是 ``[start_tick, end_tick] + 中心点`` 的区间表，自己叠。

awpy 延迟导入：没有网格资源时抛 :class:`GeoUnavailable`，由调用方降级为
"只出标签、指标留 null"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import polars as pl

from .config import CoverPoint, EntryPoint, GeomCfg, MapSpec, Vec3, dist2, dist3


class GeoUnavailable(RuntimeError):
    """拿不到碰撞网格 / 没有可用几何时的降级信号。"""


# --------------------------------------------------------------------------- 数学


def eye(pos: Vec3, crouched: bool, cfg: GeomCfg) -> Vec3:
    """玩家位置 -> 眼睛位置。**所有视线查询都必须过这一层。**"""
    offset = cfg.eye_crouch if crouched else cfg.eye_stand
    return (float(pos[0]), float(pos[1]), float(pos[2]) + float(offset))


def yaw_forward(yaw_deg: float) -> tuple[float, float]:
    """水平朝向单位向量（CS 约定：yaw 0 = +X，逆时针为正）。"""
    rad = math.radians(float(yaw_deg))
    return (math.cos(rad), math.sin(rad))


def bearing_deg(dx: float, dy: float) -> float:
    """方向 -> 方位角（度，0~360）。"""
    return math.degrees(math.atan2(dy, dx)) % 360.0


def angle_between_deg(a: float, b: float) -> float:
    """两个方位角的最小夹角（0~180）。"""
    diff = abs((float(a) - float(b)) % 360.0)
    return diff if diff <= 180.0 else 360.0 - diff


def facing_gap_deg(pos: Vec3, target: Vec3, yaw_deg: float) -> float:
    """目标相对朝向的夹角：>90 度 = 在背后/侧后。"""
    forward = bearing_deg(*yaw_forward(yaw_deg))
    to_target = bearing_deg(float(target[0]) - float(pos[0]), float(target[1]) - float(pos[1]))
    return angle_between_deg(forward, to_target)


def in_view(observer_pos: Vec3, observer_yaw_deg: float, target_pos: Vec3, fov_deg: float = 90.0) -> bool:
    """目标是否在观察者的水平有效视野内（±fov/2）。

    ``is_visible`` 是 360° 几何通视；人只有约 90° 有效视野。先手差必须用它判断
    「谁先**看见**谁」，否则双方同一 tick 同时"通视"，gap 恒为 0。
    """
    forward = bearing_deg(*yaw_forward(observer_yaw_deg))
    to_target = bearing_deg(
        float(target_pos[0]) - float(observer_pos[0]),
        float(target_pos[1]) - float(observer_pos[1]),
    )
    return angle_between_deg(forward, to_target) <= float(fov_deg) / 2.0


def point_segment_distance(a: Vec3, b: Vec3, p: Vec3) -> float:
    """点 ``p`` 到线段 ``ab`` 的三维距离。"""
    ax, ay, az = (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]), float(b[2]) - float(a[2]))
    px, py, pz = (float(p[0]) - float(a[0]), float(p[1]) - float(a[1]), float(p[2]) - float(a[2]))
    denom = ax * ax + ay * ay + az * az
    if denom == 0.0:
        return math.sqrt(px * px + py * py + pz * pz)
    t = max(0.0, min(1.0, (px * ax + py * ay + pz * az) / denom))
    qx, qy, qz = (a[0] + t * ax, a[1] + t * ay, a[2] + t * az)
    return dist3((float(qx), float(qy), float(qz)), p)


# --------------------------------------------------------------------------- 烟雾


def smoke_blocks(
    smokes: pl.DataFrame | None,
    a: Vec3,
    b: Vec3,
    tick: int | None,
    radius: float,
) -> bool:
    """线段 a->b 是否被某团**当时存在**的烟挡住。

    约定：烟球中心到线段距离 <= radius 视为遮挡。区间表用 ``start_tick/end_tick``。
    """
    if tick is None or smokes is None or not isinstance(smokes, pl.DataFrame):
        return False
    if smokes.is_empty() or not {"start_tick", "end_tick"} <= set(smokes.columns):
        return False
    active = smokes.filter(
        (pl.col("start_tick") <= int(tick)) & (pl.col("end_tick") >= int(tick))
    )
    for row in active.to_dicts():
        try:
            center = (float(row["x"]), float(row["y"]), float(row["z"]))
        except (KeyError, TypeError, ValueError):
            continue
        if point_segment_distance(a, b, center) <= radius:
            return True
    return False


# --------------------------------------------------------------------------- 网格单例


_VC_CACHE: dict[str, Any] = {}


def get_visibility_checker(map_name: str, version: int | None = None) -> Any:
    """进程级单例：BVH 构建很贵，别在循环里 new。"""
    key = f"{map_name}@{version}"
    cached = _VC_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from awpy import VisibilityChecker  # noqa: PLC0415 - 延迟导入
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境路径
        raise GeoUnavailable("没有安装 awpy，无法计算可见性") from exc
    vc = VisibilityChecker(map_name) if version is None else VisibilityChecker(map_name, version=version)
    _VC_CACHE[key] = vc
    return vc


# --------------------------------------------------------------------------- 上下文


@dataclass
class GeoContext:
    """一次分析里复用的几何上下文（一张图的网格 + 配置 + 烟雾表）。"""

    map_name: str
    cfg: GeomCfg
    spec: MapSpec = field(default_factory=lambda: MapSpec(name="unknown"))
    smokes: pl.DataFrame = field(default_factory=pl.DataFrame)
    vc: Any | None = None  # 任何带 is_visible(a, b) 的对象（测试可注入桩）
    _vis_cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- 基础 ---------------------------------------------------------------
    def eye(self, pos: Vec3, crouched: bool = False) -> Vec3:
        return eye(pos, crouched, self.cfg)

    def smoke_hit(self, a: Vec3, b: Vec3, tick: int | None) -> bool:
        return smoke_blocks(self.smokes, a, b, tick, self.cfg.smoke_radius)

    def visible(
        self,
        a: Vec3,
        b: Vec3,
        *,
        tick: int | None = None,
        crouch_a: bool = False,
        crouch_b: bool = False,
    ) -> bool:
        """两点之间是否通视（已含眼高与烟雾）。

        结果按参数记忆化：exposure/crossfire 会对同一批进入点重复查询，
        killer_emerge_tick 也会被 rule_peeked / killer_lead_ms 各调一次，
        缓存后重复查询不打 BVH（BVH 查询是几何路径的大头）。
        """
        key = (tuple(a), tuple(b), tick, bool(crouch_a), bool(crouch_b))
        hit = self._vis_cache.get(key)
        if hit is not None:
            return hit
        ea, eb = self.eye(a, crouch_a), self.eye(b, crouch_b)
        if self.smoke_hit(ea, eb, tick):
            self._vis_cache[key] = False
            return False
        if self.vc is None:
            raise GeoUnavailable("GeoContext 没有碰撞网格，无法判断可见性")
        result = bool(self.vc.is_visible(ea, eb))
        self._vis_cache[key] = result
        return result

    # -- 区域 ---------------------------------------------------------------
    def entry_points(self, pos: Vec3) -> tuple[EntryPoint, ...]:
        return tuple(ep for ep in self.spec.entry_points_for(pos) if ep.usable)

    def cover_points(self, pos: Vec3) -> tuple[CoverPoint, ...]:
        return self.spec.cover_points_for(pos)

    def ready(self, min_entry: int | None = None, min_cover: int | None = None) -> bool:
        """坐标是否配齐（没配齐时指标降级为 null，不报错）。"""
        min_entry = self.spec.gate_min_entry_points if min_entry is None else min_entry
        min_cover = self.spec.gate_min_cover_points if min_cover is None else min_cover
        regions = list(self.spec.regions) or ["A"]
        for region in regions:
            entries = [ep for ep in self.spec.entry_points.get(region, ()) if ep.usable]
            covers = self.spec.cover_points.get(region, ())
            if len(entries) < min_entry or len(covers) < min_cover:
                return False
        return True

    # -- 三个几何指标 -------------------------------------------------------
    def exposure(
        self,
        pos: Vec3,
        *,
        tick: int | None = None,
        crouched: bool = False,
        min_entry: int | None = None,
    ) -> tuple[int | None, list[str]]:
        """能看见你眼睛的敌方进入点数量。坐标没配齐 -> (None, [])。"""
        min_entry = self.spec.gate_min_entry_points if min_entry is None else min_entry
        entries = self.entry_points(pos)
        if len(entries) < min_entry:
            return None, []
        if self.vc is None:
            return None, []
        visible_names: list[str] = []
        for ep in entries:
            assert ep.pos is not None
            if self.visible(pos, ep.pos, tick=tick, crouch_a=crouched):
                visible_names.append(ep.name)
        return len(visible_names), visible_names

    def crossfire(
        self,
        pos: Vec3,
        *,
        tick: int | None = None,
        crouched: bool = False,
        min_entry: int | None = None,
    ) -> int | None:
        """暴露进入点之间两两夹角 > crossfire_angle_deg 的对数。"""
        min_entry = self.spec.gate_min_entry_points if min_entry is None else min_entry
        entries = self.entry_points(pos)
        if len(entries) < min_entry or self.vc is None:
            return None
        bearings: list[float] = []
        for ep in entries:
            assert ep.pos is not None
            if not self.visible(pos, ep.pos, tick=tick, crouch_a=crouched):
                continue
            bearings.append(bearing_deg(ep.pos[0] - pos[0], ep.pos[1] - pos[1]))
        pairs = 0
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                if angle_between_deg(bearings[i], bearings[j]) > self.cfg.crossfire_angle_deg:
                    pairs += 1
        return pairs

    def escape_time(self, pos: Vec3) -> float | None:
        """到最近掩体的秒数（MVP 直线近似，走速 250 u/s；进阶版换 nav 最短路）。"""
        covers = self.cover_points(pos)
        if not covers:
            return None
        nearest = min(dist2(pos, cp.pos) for cp in covers)
        if self.cfg.walk_speed <= 0:  # pragma: no cover - 配置错误
            return None
        return round(nearest / self.cfg.walk_speed, 2)


# --------------------------------------------------------------------------- 构造


def build_geo(
    map_name: str,
    settings: Any,
    *,
    smokes: pl.DataFrame | None = None,
    version: int | None = None,
) -> GeoContext:
    """建几何上下文；拿不到网格就抛 :class:`GeoUnavailable`。"""
    spec = settings.map_spec(map_name)
    vc = get_visibility_checker(map_name, version=version)
    return GeoContext(
        map_name=map_name,
        cfg=settings.geom,
        spec=spec,
        smokes=smokes if isinstance(smokes, pl.DataFrame) else pl.DataFrame(),
        vc=vc,
    )


def try_build_geo(
    map_name: str,
    settings: Any,
    *,
    smokes: pl.DataFrame | None = None,
    version: int | None = None,
) -> tuple[GeoContext | None, str | None]:
    """降级版构造：返回 (geo, 失败原因)。"""
    try:
        return build_geo(map_name, settings, smokes=smokes, version=version), None
    except Exception as exc:  # noqa: BLE001 - 降级路径要吞掉一切
        return None, str(exc)
