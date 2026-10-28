"""配置加载 + 运行环境引导。

这个模块集中处理三件与 DSH 沙箱有关的事（**先读 README 的"沙箱三件套"**）：

* ``AWPY_DATA_DIR``  —— awpy 资源默认缓存在 ``~/.awpy``，workspace 外不可写；
* ``MPLCONFIGDIR``   —— matplotlib 导入期会写字体缓存到用户目录；
* ``TMP`` / ``TEMP`` —— 各类临时文件。

所有阈值、窗口、眼高、速度都从 ``config/default.yaml`` 读，代码里不出现魔数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

Vec3 = tuple[float, float, float]


class ConfigError(RuntimeError):
    """配置缺失或格式错误。"""


# --------------------------------------------------------------------------- 小工具


def dist3(a: Vec3, b: Vec3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def dist2(a: Vec3, b: Vec3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def as_vec3(value: Any, *, what: str = "position") -> Vec3 | None:
    """把 [x, y, z] / None 转成 Vec3。"""
    if value is None:
        return None
    try:
        x, y, z = value
    except (TypeError, ValueError) as exc:  # pragma: no cover - 配置错误路径
        raise ConfigError(f"{what} 需要 [x, y, z] 三元组，实际是 {value!r}") from exc
    return (float(x), float(y), float(z))


@dataclass(frozen=True)
class Paths:
    root: Path
    demos: Path
    cache: Path
    output: Path
    awpy_data: Path
    tmp: Path

    def ensure(self) -> "Paths":
        for p in (
            self.demos,
            self.demos / "me",
            self.demos / "pro",
            self.cache,
            self.output,
            self.awpy_data,
            self.tmp,
            self.tmp / "mpl",
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- 各类配置


@dataclass(frozen=True)
class MeCfg:
    name: str = ""
    steamid: int | None = None


@dataclass(frozen=True)
class WindowsCfg:
    pre_s: float = 2.0
    post_s: float = 0.5
    merge_gap_s: float = 1.0
    max_windows: int = 400


@dataclass(frozen=True)
class RulesCfg:
    blind_lookback_s: float = 2.0
    blind_min_duration_s: float = 1.2
    behind_angle_deg: float = 90.0
    empty_clip_window_s: float = 0.6
    near_miss_window_s: float = 0.15
    moving_speed: float = 60.0
    moving_lookback_s: float = 0.5
    peek_speed_max: float = 30.0
    peek_speed_lookback_s: float = 0.5
    contact_time_s: float = 0.5
    predictable_lookback_s: float = 0.3
    predictable_angle_deg: float = 6.0
    gap_window_s: float = 1.5
    peek_search_s: float = 1.5
    duel_fallback_s: float = 2.0  # 交火起点兜底窗口


@dataclass(frozen=True)
class FaultCfg:
    """责任判定阈值（计划 §5.2）。"""

    preaim_bad_deg: float = 15.0   # 他出现时准星偏这么多度 = 预瞄有问题
    preaim_ok_deg: float = 6.0     # 这以内 = 预瞄没问题
    slow_react_ms: int = 350       # 反应超过这么多毫秒 = 慢
    oneshot_hp: int = 30           # 血量低于此 = 残血被秒，无解
    exposure_bad: int = 2          # 暴露面达到此 = 站位有问题
    escape_bad: float = 1.0        # 退路超过此秒 = 站位有问题
    escape_config_max: float = 5.0  # 退路超过此秒 = 地图配置问题（position_config），非玩家问题


@dataclass(frozen=True)
class GeomCfg:
    eye_stand: float = 64.0
    eye_crouch: float = 46.0
    smoke_radius: float = 144.0
    walk_speed: float = 250.0
    crossfire_angle_deg: float = 60.0
    fov_deg: float = 90.0  # 有效视野：先手差用 FOV 而非 360° 通视


@dataclass(frozen=True)
class BenchCfg:
    grid: float = 96.0
    anchor_radius: float = 200.0
    anchor_hold_s: float = 40.0
    anchor_hold_ratio: float = 0.5
    role_window_s: float = 60.0
    outlier_quantile: float = 0.2
    # M4 规格（用户拍板）：同经济 / 累积分位 / 邻域频率 / 8 角色
    economy_tolerance: float = 500.0  # equipment_value 差 <= 此值 = 同经济
    neighborhood_radius: float = 200.0  # 邻域频率半径（Hammer units）
    holding_speed: float = 80.0  # 架枪速度阈值 u/s
    # 采样降密度：每 N tick 取 1 个落位点（定位基准只需空间分布，per-tick 全量会撑爆内存）
    sample_stride: int = 8


@dataclass(frozen=True)
class ReportCfg:
    min_samples: int = 3
    top_n: int = 5


# --------------------------------------------------------------------------- 地图配置


@dataclass(frozen=True)
class EntryPoint:
    """敌方进入点：用来算暴露面 / 交叉覆盖。"""

    name: str
    pos: Vec3 | None = None

    @property
    def usable(self) -> bool:
        return self.pos is not None


@dataclass(frozen=True)
class CoverPoint:
    """掩体 / 安全点：用来算退路时间。"""

    name: str
    pos: Vec3


@dataclass
class MapSpec:
    """单张地图的区域、进入点、掩体点。坐标允许留空（降级而非报错）。"""

    name: str
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    entry_points: dict[str, tuple[EntryPoint, ...]] = field(default_factory=dict)
    cover_points: dict[str, tuple[CoverPoint, ...]] = field(default_factory=dict)
    default_region: str | None = None
    gate_min_entry_points: int = 3
    gate_min_cover_points: int = 1

    # -- 查询 ---------------------------------------------------------------
    def region_of(self, pos: Vec3) -> str | None:
        best: str | None = None
        best_d: float | None = None
        for region, spec in self.regions.items():
            center = spec.get("center")
            if not center:
                continue
            radius = float(spec.get("radius") or 0.0)
            d = dist2(pos, center)
            if d <= radius and (best_d is None or d < best_d):
                best, best_d = region, d
        if best is not None:
            return best
        if self.default_region in self.regions:
            return self.default_region
        return None

    def entry_points_for(self, pos: Vec3) -> tuple[EntryPoint, ...]:
        region = self.region_of(pos)
        if region and self.entry_points.get(region):
            return self.entry_points[region]
        merged = [ep for eps in self.entry_points.values() for ep in eps]
        return tuple(merged)

    def cover_points_for(self, pos: Vec3) -> tuple[CoverPoint, ...]:
        region = self.region_of(pos)
        if region and self.cover_points.get(region):
            return self.cover_points[region]
        merged = [cp for cps in self.cover_points.values() for cp in cps]
        return tuple(merged)

    # -- 门禁 ---------------------------------------------------------------
    def ready(self, min_entry: int | None = None, min_cover: int | None = None) -> bool:
        return not self.validate(min_entry, min_cover)

    def validate(self, min_entry: int | None = None, min_cover: int | None = None) -> list[str]:
        """返回问题列表；空列表 = 配置可用。默认用 maps.yaml ``defaults`` 里的门禁。"""
        min_entry = self.gate_min_entry_points if min_entry is None else min_entry
        min_cover = self.gate_min_cover_points if min_cover is None else min_cover
        problems: list[str] = []
        regions = list(self.regions) or ["A"]
        for region in regions:
            entries = [ep for ep in self.entry_points.get(region, ()) if ep.usable]
            covers = self.cover_points.get(region, ())
            if len(entries) < min_entry:
                problems.append(
                    f"区域 {region}: 可用进入点 {len(entries)} 个，少于 {min_entry} 个（exposure/crossfire 会为 null）"
                )
            if len(covers) < min_cover:
                problems.append(
                    f"区域 {region}: 可用掩体点 {len(covers)} 个，少于 {min_cover} 个（escape_time 会为 null）"
                )
        return problems


def _parse_entry_points(raw: Any) -> tuple[EntryPoint, ...]:
    out: list[EntryPoint] = []
    for item in raw or ():
        if isinstance(item, Mapping):
            out.append(EntryPoint(name=str(item.get("name") or "entry"), pos=as_vec3(item.get("pos"), what="entry point pos")))
        else:
            out.append(EntryPoint(name="entry", pos=as_vec3(item, what="entry point pos")))
    return tuple(out)


def _parse_cover_points(raw: Any) -> tuple[CoverPoint, ...]:
    out: list[CoverPoint] = []
    for item in raw or ():
        if isinstance(item, Mapping):
            pos = as_vec3(item.get("pos"), what="cover point pos")
            name = str(item.get("name") or "cover")
        else:
            pos, name = as_vec3(item, what="cover point pos"), "cover"
        if pos is not None:
            out.append(CoverPoint(name=name, pos=pos))
    return tuple(out)


def parse_maps(doc: Mapping[str, Any] | None) -> dict[str, MapSpec]:
    """``maps.yaml`` -> {map_name: MapSpec}。"""
    doc = doc or {}
    defaults = doc.get("defaults") or {}
    maps: dict[str, MapSpec] = {}
    for name, raw in (doc.get("maps") or {}).items():
        raw = raw or {}
        regions = raw.get("regions") or {}
        entry = {region: _parse_entry_points(eps) for region, eps in (raw.get("entry_points") or {}).items()}
        cover = {region: _parse_cover_points(cps) for region, cps in (raw.get("cover_points") or {}).items()}
        maps[str(name)] = MapSpec(
            name=str(name),
            regions={str(k): {**v, "center": as_vec3((v or {}).get("center"), what=f"{name}.{k}.center")} for k, v in regions.items()},
            entry_points=entry,
            cover_points=cover,
            default_region=defaults.get("default_region"),
            gate_min_entry_points=int(defaults.get("gate_min_entry_points", 3)),
            gate_min_cover_points=int(defaults.get("gate_min_cover_points", 1)),
        )
    return maps


def _merge_named(current: tuple[Any, ...], incoming: Any, factory: Any) -> tuple[Any, ...]:
    """按 name 合并：同名替换，新名追加（保持既有顺序，输出稳定）。"""
    by_name = {item.name: item for item in current}
    order = [item.name for item in current]
    for row in incoming or ():
        if not isinstance(row, Mapping):
            row = {"pos": row}
        name = str(row.get("name") or "point").strip() or "point"
        item = factory(name, as_vec3(row.get("pos"), what=f"{name}.pos"))
        if item is None:
            continue
        if name not in by_name:
            order.append(name)
        by_name[name] = item
    return tuple(by_name[name] for name in order)


def merge_points(
    maps: dict[str, MapSpec],
    points_doc: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, MapSpec]:
    """把 ``config/points.yaml``（``csc maps edit`` 的产物）合并进 MapSpec。

    * ``maps.yaml`` 保留人工注释与门禁；本函数只**补/覆盖坐标**；
    * 同名点覆盖，新点追加，区域按 region 合并；
    * 目标图若在 maps.yaml 里没写过，也能凭 points.yaml 新建（用 defaults 的门禁值）。
    """
    points_doc = points_doc or {}
    defaults = defaults or {}
    for map_name, block in (points_doc.get("maps") or {}).items():
        map_name = str(map_name)
        spec = maps.get(map_name)
        if spec is None:
            spec = MapSpec(
                name=map_name,
                default_region=defaults.get("default_region"),
                gate_min_entry_points=int(defaults.get("gate_min_entry_points", 3)),
                gate_min_cover_points=int(defaults.get("gate_min_cover_points", 1)),
            )
            maps[map_name] = spec
        block = block or {}

        for region, raw in (block.get("regions") or {}).items():
            region = str(region)
            raw = raw or {}
            merged = dict(spec.regions.get(region) or {})
            center = as_vec3(raw.get("center"), what=f"{map_name}.{region}.center")
            if center is not None:
                merged["center"] = center
            if raw.get("radius") is not None:
                merged["radius"] = float(raw["radius"])
            spec.regions[region] = merged

        for region, rows in (block.get("entry_points") or {}).items():
            region = str(region)
            spec.entry_points[region] = _merge_named(
                spec.entry_points.get(region, ()),
                rows,
                lambda name, pos: EntryPoint(name=name, pos=pos),
            )
        for region, rows in (block.get("cover_points") or {}).items():
            region = str(region)
            spec.cover_points[region] = _merge_named(
                spec.cover_points.get(region, ()),
                rows,
                lambda name, pos: CoverPoint(name=name, pos=pos) if pos is not None else None,
            )
    return maps


# --------------------------------------------------------------------------- 总配置


@dataclass(frozen=True)
class Settings:
    paths: Paths
    me: MeCfg
    windows: WindowsCfg
    rules: RulesCfg
    fault: FaultCfg
    geom: GeomCfg
    bench: BenchCfg
    report: ReportCfg
    maps: dict[str, MapSpec]
    config_dir: Path = DEFAULT_CONFIG_DIR

    def map_spec(self, map_name: str | None) -> MapSpec:
        """取地图配置；没有配置时返回空 MapSpec（指标自动降级）。"""
        if map_name and map_name in self.maps:
            return self.maps[map_name]
        return MapSpec(name=map_name or "unknown", default_region=None)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"找不到配置文件：{path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射：{path}")
    return data


def _read_yaml_optional(path: Path) -> dict[str, Any]:
    """可选配置：不存在就当空（points.yaml 首次运行前并不存在）。"""
    if not path.exists():
        return {}
    return _read_yaml(path)


def _section(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key) or {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"配置段 {key} 必须是映射")
    return dict(value)


def load_paths(config_dir: Path | None = None) -> Paths:
    cdir = Path(config_dir or DEFAULT_CONFIG_DIR)
    doc = _read_yaml(cdir / "default.yaml")
    raw = doc.get("paths") or {}
    root = PROJECT_ROOT

    def resolve(key: str, default: str) -> Path:
        return (root / str(raw.get(key, default))).resolve()

    return Paths(
        root=root,
        demos=resolve("demos", "data/demos"),
        cache=resolve("cache", "data/cache"),
        output=resolve("output", "output"),
        awpy_data=resolve("awpy_data", ".awpy"),
        tmp=resolve("tmp", ".tmp"),
    )


def bootstrap_env(config_dir: Path | None = None, *, force: bool = False) -> Paths:
    """把 awpy / matplotlib / 临时目录钉到 workspace 内。幂等。"""
    paths = load_paths(config_dir).ensure()
    pairs = {
        "AWPY_DATA_DIR": str(paths.awpy_data),
        "MPLCONFIGDIR": str(paths.tmp / "mpl"),
    }
    for key, value in pairs.items():
        if force or not os.environ.get(key):
            os.environ[key] = value
    os.environ.setdefault("TMP", str(paths.tmp))
    os.environ.setdefault("TEMP", str(paths.tmp))
    return paths


def load_settings(config_dir: Path | None = None) -> Settings:
    cdir = Path(config_dir or DEFAULT_CONFIG_DIR)
    doc = _read_yaml(cdir / "default.yaml")
    paths = load_paths(cdir)

    me_doc = _section(doc, "me")
    steamid = me_doc.get("steamid")
    me = MeCfg(name=str(me_doc.get("name") or ""), steamid=int(steamid) if steamid else None)

    windows = WindowsCfg(**_section(doc, "windows"))
    rules = RulesCfg(**_section(doc, "rules"))
    fault = FaultCfg(**_section(doc, "fault"))
    geom = GeomCfg(**_section(doc, "geom"))
    bench = BenchCfg(**_section(doc, "bench"))
    report = ReportCfg(**_section(doc, "report"))

    maps_doc = _read_yaml(cdir / "maps.yaml")
    maps = parse_maps(maps_doc)
    points_doc = _read_yaml_optional(cdir / "points.yaml")
    if points_doc.get("maps"):
        maps = merge_points(maps, points_doc, defaults=maps_doc.get("defaults") or {})
    return Settings(
        paths=paths,
        me=me,
        windows=windows,
        rules=rules,
        fault=fault,
        geom=geom,
        bench=bench,
        report=report,
        maps=maps,
        config_dir=cdir,
    )
