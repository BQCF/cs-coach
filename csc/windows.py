"""死亡窗口规划与轨迹缓存。

文档 §6 的第 4 个坑：全量 ``snapshots`` 会爆内存。做法是三步：

1. 只读 ``kills`` 拿死亡 tick；
2. 把 ``[death-pre_s, death+post_s]`` 按时间合并（相邻间隔 < ``merge_gap_s`` 就并），
   这样同一回合的一串死亡只调一次 awpy；
3. 每段窗口调 ``demo.snapshots(start_tick=, end_tick=)``，拼起来缓存成
   ``<cache>/<demo_key>/windows.parquet``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from .config import WindowsCfg
from .demo_io import DemoContext, open_demo, snapshot_windows

WINDOWS_FILE = "windows.parquet"

# 内部统一使用的轨迹列（awpy 少给哪列就补 null）
TRACK_COLUMNS: tuple[str, ...] = (
    "tick",
    "steamid",
    "name",
    "side",
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "velocity",
    "health",
    "armor",
    "is_crouched",
    "is_blinded",
)


class WindowError(RuntimeError):
    """窗口规划失败（例如窗口数量爆炸）。"""


@dataclass(frozen=True)
class Span:
    """一段连续的 tick 窗口，以及它包含的死亡 tick。"""

    start_tick: int
    end_tick: int
    death_ticks: tuple[int, ...]


# --------------------------------------------------------------------------- 规划


def plan_spans(death_ticks: Iterable[int], tick_rate: int, cfg: WindowsCfg) -> list[Span]:
    """把死亡 tick 扩成窗口并合并。返回按时间排序、互不重叠的 Span 列表。"""
    pre = int(round(cfg.pre_s * tick_rate))
    post = int(round(cfg.post_s * tick_rate))
    gap = int(round(cfg.merge_gap_s * tick_rate))

    raw = sorted((max(0, int(t) - pre), int(t) + post, int(t)) for t in set(int(x) for x in death_ticks))
    spans: list[Span] = []
    cur_start = cur_end = None
    group: list[int] = []
    for start, end, tick in raw:
        if cur_start is None:
            cur_start, cur_end, group = start, end, [tick]
            continue
        if start <= cur_end + gap:
            cur_end = max(cur_end, end)
            group.append(tick)
        else:
            spans.append(Span(cur_start, cur_end, tuple(sorted(group))))
            cur_start, cur_end, group = start, end, [tick]
    if cur_start is not None:
        spans.append(Span(cur_start, cur_end, tuple(sorted(group))))

    if len(spans) > cfg.max_windows:
        raise WindowError(
            f"窗口数 {len(spans)} 超过上限 {cfg.max_windows}："
            f"检查 deaths 是否被误当成全体玩家，或调大 windows.merge_gap_s"
        )
    return spans


# --------------------------------------------------------------------------- 轨迹规范化


def normalize_snap(df: pl.DataFrame) -> pl.DataFrame:
    """把 awpy 快照表裁成内部列集，缺列补 null。"""
    if not isinstance(df, pl.DataFrame) or df.is_empty():
        return pl.DataFrame(schema={col: pl.Null for col in TRACK_COLUMNS})
    exprs = []
    for col in TRACK_COLUMNS:
        exprs.append(pl.col(col).alias(col) if col in df.columns else pl.lit(None).alias(col))
    return df.select(exprs).sort(["tick", "steamid"])


def slice_track(snap: pl.DataFrame, steamid: int, start_tick: int, end_tick: int) -> pl.DataFrame:
    """取某玩家在某段 tick 内的轨迹。"""
    if snap.is_empty():
        return snap
    return snap.filter(
        (pl.col("steamid") == int(steamid))
        & (pl.col("tick") >= int(start_tick))
        & (pl.col("tick") <= int(end_tick))
    ).sort("tick")


def frame_at(snap: pl.DataFrame, steamid: int, tick: int) -> dict | None:
    """取某玩家 <= tick 的最后一帧（死亡那一 tick 玩家可能已无实体）。"""
    if snap.is_empty():
        return None
    rows = snap.filter((pl.col("steamid") == int(steamid)) & (pl.col("tick") <= int(tick))).sort("tick")
    if rows.is_empty():
        return None
    return rows.row(rows.height - 1, named=True)


# --------------------------------------------------------------------------- 缓存


def windows_path(key: str, cache_root: Path) -> Path:
    return Path(cache_root) / key / WINDOWS_FILE


def _span_signature(spans: Sequence[Span], tick_rate: int, player: int | None) -> str:
    """窗口内容签名：换玩家 / 改窗口参数 / 换 tickrate 都会得到不同 key。

    否则缓存里存的是玩家 A 的窗口，玩家 B 一读就拿到错轨迹，指标全错还不报错。
    """
    body = "|".join(f"{s.start_tick}-{s.end_tick}" for s in spans)
    extra = f"tr{tick_rate}" + (f"p{int(player)}" if player is not None else "")
    return hashlib.sha1(f"{body}@{extra}".encode()).hexdigest()[:8]


def ensure_windows(
    ctx: DemoContext,
    death_ticks: Sequence[int],
    cache_root: Path,
    cfg: WindowsCfg,
    *,
    player: int | None = None,
    force: bool = False,
) -> tuple[list[Span], pl.DataFrame]:
    """规划窗口并返回轨迹（优先读缓存）。"""
    spans = plan_spans(death_ticks, ctx.tick_rate, cfg)
    sig = _span_signature(spans, ctx.tick_rate, player)
    path = windows_path(f"{ctx.key}_{sig}", cache_root)
    if not force and path.exists():
        return spans, normalize_snap(pl.read_parquet(path))

    demo = open_demo(ctx)
    snap = snapshot_windows(demo, [(s.start_tick, s.end_tick) for s in spans])
    snap = normalize_snap(snap)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 空结果也落盘：否则「无死亡点/空窗口」的 demo 每次都重新 open_demo 取快照
    snap.write_parquet(path)
    return spans, snap
