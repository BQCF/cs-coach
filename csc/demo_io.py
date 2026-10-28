"""demo 解析与 parquet 缓存。

设计要点（对应计划 §2 F3/F6）：

* ``Demo.load()`` 返回 None，表要从属性取；
* tickrate 用 ``demo.tick_rate``，header 兜底，配置可覆盖；
* 全量 ``snapshots`` 会爆内存，所以**只缓存表**，玩家轨迹由 ``windows.py`` 按窗口取。
* awpy 是**延迟导入**的：没装 awpy 时本模块仍可被 import（方便跑 selftest/单测）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

from . import schema

DATASETS: tuple[str, ...] = (
    "kills",
    "blinds",
    "shots",
    "smokes",
    "fires",
    "rounds",
    "round_economy",
    "players",
    "damages",
)

# 缓存里 parquet 文件名 -> 表名
_TABLE_FILES = {name: f"{name}.parquet" for name in DATASETS}
_DEATH_FRAMES = "death_frames.parquet"
_META = "meta.json"


class DemoError(RuntimeError):
    """demo 相关错误的基类。"""


class DemoNotFound(DemoError):
    """给的路径上没有 .dem。"""


class ParseError(DemoError):
    """awpy 解析失败。"""


class PlayerNotFound(DemoError):
    """demo 里找不到这个玩家。"""


# --------------------------------------------------------------------------- 上下文


@dataclass(frozen=True)
class DemoContext:
    """一个 demo 的解析结果（全部是 polars DataFrame）。"""

    key: str
    path: Path
    map_name: str
    tick_rate: int
    tables: dict[str, pl.DataFrame]
    death_frames: pl.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def table(self, name: str) -> pl.DataFrame:
        return self.tables.get(name, pl.DataFrame())

    @property
    def kills(self) -> pl.DataFrame:
        return self.table("kills")

    @property
    def players(self) -> pl.DataFrame:
        return self.table("players")

    def win_rate(self) -> dict[str, int]:  # pragma: no cover - 便捷方法
        rounds = self.table("rounds")
        if rounds.is_empty() or "winner_side" not in rounds.columns:
            return {}
        return {str(r["winner_side"]): r["len"] for r in rounds.group_by("winner_side").len().to_dicts()}


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame()


def _versions() -> tuple[str, str]:
    try:  # pragma: no cover - 依赖环境
        import awpy

        awpy_version = getattr(awpy, "__version__", "unknown")
    except Exception:  # pragma: no cover
        awpy_version = "not-installed"
    return awpy_version, pl.__version__


# --------------------------------------------------------------------------- 缓存键


def cache_key(path: Path) -> str:
    """文件名 + size + mtime + 依赖版本 -> 12 位短键。"""
    path = Path(path)
    stat = path.stat()
    awpy_version, polars_version = _versions()
    raw = "|".join(
        [
            path.name,
            str(stat.st_size),
            str(stat.st_mtime_ns),
            awpy_version,
            polars_version,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def cache_dir_for(key: str, cache_root: Path) -> Path:
    return Path(cache_root) / key


def has_cache(key: str, cache_root: Path) -> bool:
    return (cache_dir_for(key, cache_root) / _META).exists()


# --------------------------------------------------------------------------- 解析


def _read_demo_attr(demo: Any, name: str) -> pl.DataFrame:
    value = getattr(demo, name, None)
    if isinstance(value, pl.DataFrame):
        return value
    if value is None:
        return _empty_frame()
    try:  # pragma: no cover - awpy 可能返回惰性对象
        return pl.DataFrame(value)
    except Exception:
        return _empty_frame()


def parse_demo(
    path: Path | str,
    cache_root: Path,
    *,
    datasets: Iterable[str] = DATASETS,
    force: bool = False,
) -> tuple[DemoContext, list[schema.ContractIssue]]:
    """解析 demo 并写 parquet 缓存；已缓存且非 force 时直接读缓存。"""
    from .config import bootstrap_env  # 局部导入避免循环

    bootstrap_env()
    path = Path(path).resolve()
    if not path.exists():
        raise DemoNotFound(f"找不到 demo 文件：{path}（CS2 的录像在 game/csgo/replays，或从比赛历史重新下载）")
    if path.suffix.lower() != ".dem":
        raise DemoNotFound(f"不是 .dem 文件：{path.name}")

    key = cache_key(path)
    cdir = cache_dir_for(key, cache_root)
    if not force and (cdir / _META).exists():
        ctx = load_context(key, cache_root)
        return ctx, schema.validate({**ctx.tables, "snap": ctx.death_frames})

    try:
        from awpy import Demo  # noqa: PLC0415 - 延迟导入
    except ModuleNotFoundError as exc:  # pragma: no cover - 环境路径
        raise ParseError("没有安装 awpy，先在 .venv 里 pip install 'awpy[plot]'") from exc

    datasets = tuple(datasets)
    try:
        demo = Demo(path)
        demo.load(*datasets)
    except Exception as exc:  # pragma: no cover - awpy 内部错误
        raise ParseError(f"awpy 解析失败：{exc}") from exc

    tables: dict[str, pl.DataFrame] = {}
    warnings: list[str] = []
    for name in datasets:
        try:
            tables[name] = _read_demo_attr(demo, name)
        except Exception as exc:  # pragma: no cover
            tables[name] = _empty_frame()
            warnings.append(f"{name}: {exc}")

    tick_rate = int(getattr(demo, "tick_rate", 0) or 0)
    if tick_rate <= 0:  # pragma: no cover - 老 demo 兜底
        header = getattr(demo, "header", {}) or {}
        tick_rate = int(header.get("tickrate") or 64)
        warnings.append(f"tick_rate 缺失，兜底用 {tick_rate}")

    map_name = str(getattr(demo, "header", {}).get("map_name") or "unknown")

    try:  # pragma: no cover - 需要 awpy 资源
        death_frames = demo.snapshots(events="player_death")
    except Exception as exc:
        death_frames = _empty_frame()
        warnings.append(f"death_frames 取不到（{exc}），回退到窗口内轨迹")

    cdir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pl.DataFrame):
            df.write_parquet(cdir / _TABLE_FILES.get(name, f"{name}.parquet"))
    if isinstance(death_frames, pl.DataFrame):
        death_frames.write_parquet(cdir / _DEATH_FRAMES)

    awpy_version, polars_version = _versions()
    meta: dict[str, Any] = {
        "key": key,
        "path": str(path),
        "map_name": map_name,
        "tick_rate": tick_rate,
        "datasets": list(datasets),
        "awpy_version": awpy_version,
        "polars_version": polars_version,
        "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": warnings,
        "rows": {name: df.height for name, df in tables.items() if isinstance(df, pl.DataFrame)},
    }
    (cdir / _META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx = DemoContext(
        key=key,
        path=path,
        map_name=map_name,
        tick_rate=tick_rate,
        tables=tables,
        death_frames=death_frames,
        meta=meta,
    )
    issues = schema.validate({**tables, "snap": death_frames})
    return ctx, issues


def load_context(key: str, cache_root: Path) -> DemoContext:
    """从缓存读回 DemoContext。"""
    cdir = cache_dir_for(key, cache_root)
    meta_path = cdir / _META
    if not meta_path.exists():
        raise DemoError(f"缓存不存在：{cdir}（先跑 python -m csc parse --demo <文件>）")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tables: dict[str, pl.DataFrame] = {}
    for name in meta.get("datasets", DATASETS):
        fpath = cdir / _TABLE_FILES.get(name, f"{name}.parquet")
        tables[name] = pl.read_parquet(fpath) if fpath.exists() else _empty_frame()
    death_frames = pl.read_parquet(cdir / _DEATH_FRAMES) if (cdir / _DEATH_FRAMES).exists() else _empty_frame()
    return DemoContext(
        key=str(meta.get("key") or key),
        path=Path(str(meta.get("path") or "")),
        map_name=str(meta.get("map_name") or "unknown"),
        tick_rate=int(meta.get("tick_rate") or 64),
        tables=tables,
        death_frames=death_frames,
        meta=meta,
    )


def open_demo(ctx: DemoContext) -> Any:
    """重新打开 awpy Demo（取快照用）。只在需要轨迹时调用。"""
    try:
        from awpy import Demo  # noqa: PLC0415 - 延迟导入
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ParseError("没有安装 awpy，无法取玩家轨迹") from exc
    if not ctx.path.exists():
        raise DemoNotFound(f"原始 demo 不在了：{ctx.path}（parquet 缓存仍在，可只跑归因）")
    return Demo(ctx.path)


def snapshot_windows(demo: Any, spans: Iterable[tuple[int, int]]) -> pl.DataFrame:
    """按 tick 区间取快照并纵向拼起来。每个 demo 只在这里碰 awpy。"""
    frames: list[pl.DataFrame] = []
    for start, end in spans:
        try:
            df = demo.snapshots(start_tick=int(start), end_tick=int(end))
        except Exception:  # pragma: no cover - 空区间等
            continue
        if isinstance(df, pl.DataFrame) and not df.is_empty():
            frames.append(df)
    if not frames:
        return _empty_frame()
    return pl.concat(frames, how="vertical_relaxed").unique(subset=["tick", "steamid"], keep="first")


# --------------------------------------------------------------------------- 玩家解析


def demos_in(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.dem"))


def resolve_player(ctx: DemoContext, query: str | int | None) -> tuple[int, str]:
    """昵称或 steamid -> (steamid, name)。"""
    players = ctx.players
    if players.is_empty():
        raise PlayerNotFound("demo 里没有 players 表，无法识别玩家")
    if query is None or str(query).strip() == "":
        raise PlayerNotFound("没指定玩家：用 --player <昵称或 steamid>，或写进 config/default.yaml 的 me.name")
    text = str(query).strip()

    if text.isdigit():
        steamid = int(text)
        hit = players.filter(pl.col("steamid") == steamid)
        if hit.is_empty():
            raise PlayerNotFound(f"demo 里没有 steamid {steamid}")
        return steamid, str(hit.row(0, named=True).get("name") or "")

    lowered = text.casefold()
    names = players.select("steamid", "name").to_dicts()
    exact = [row for row in names if str(row.get("name") or "").casefold() == lowered]
    if len(exact) == 1:
        return int(exact[0]["steamid"]), str(exact[0]["name"])
    partial = [row for row in names if lowered in str(row.get("name") or "").casefold()]
    if len(partial) == 1:
        return int(partial[0]["steamid"]), str(partial[0]["name"])
    if len(partial) > 1:
        options = ", ".join(str(row.get("name")) for row in partial)
        raise PlayerNotFound(f"昵称 '{text}' 匹配到多个玩家：{options}；请用完整昵称或 steamid")
    raise PlayerNotFound(f"demo 里找不到玩家 '{text}'")
