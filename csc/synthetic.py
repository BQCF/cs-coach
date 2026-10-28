"""selftest 用合成数据：不需要 demo、不需要 awpy 资源，跑通整条管线。

8 个死亡点各靶向一条 v2 归因规则（含已启用的规则 7「信息缺失硬拉」与
「正面遭遇」），selftest 会显式断言标签序列逐个命中。

几何用一个方向感知的 :class:`ScriptedGeo` 桩：按 tick 与「观察者是我还是他」返回
不同的可见性，从而能测出**正数的先手差**（他先看见我）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from . import metrics, report, plot
from .config import EntryPoint, GeomCfg, MapSpec, Settings, Vec3, dist3
from .deaths import DeathRecord, attribute, build_death_windows, records_to_frame
from .demo_io import DemoContext
from .windows import Span, plan_spans

TICK_RATE = 128
MY_STEAMID = 111
KILLER_STEAMID = 222
MY_NAME = "me"
KILLER_NAME = "killer"
MAP = "de_test"


@dataclass(frozen=True)
class Scenario:
    label: str
    my_yaw: float = 0.0                       # 我的朝向（0 = +X）
    killer_offset: tuple[float, float] = (200.0, 0.0)  # 击杀者相对我的位置
    my_vel: float = 0.0
    killer_vel: float = 0.0
    blind_s: float = 0.0
    empty_clip: bool = False
    me_from: float | None = None              # 秒（相对死亡，负数=死前）：我何时能看见他
    him_from: float | None = None             # 秒：他何时能看见我


# 8 个剧本，各靶向一条 v2 归因规则（计划 §5.3）：
# blind / behind / fallback(空弹匣只留 clip_state) / moving / peeked / holding / info_missing / encounter
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("被闪", blind_s=2.0),
    Scenario("背身/侧身", killer_offset=(-200.0, 200.0)),
    Scenario("其他/对枪", empty_clip=True),                              # v2：空弹匣不再是死因标签，只进 clip_state
    Scenario("移动中对枪", my_vel=200.0, me_from=-0.2, him_from=-0.2),   # 看得见→跳过信息缺失；跑动→跳过 encounter/holding
    Scenario("被拉死", me_from=-0.05, him_from=-0.30),
    Scenario("架死", me_from=-2.0, him_from=-2.0),
    Scenario("信息缺失硬拉", my_vel=200.0),                              # 推进中完全看不见击杀者、无队友掩护
    Scenario("正面遭遇", my_vel=45.0, me_from=-0.2, him_from=-0.2),      # gap≈0 且未跑动（45>30 躲开被拉死）
)


def death_ticks() -> list[int]:
    return [1000 + i * 600 for i in range(len(SCENARIOS))]


# --------------------------------------------------------------------------- 数据


def _df(data: list[dict], schema: dict) -> pl.DataFrame:
    return pl.DataFrame(data, schema=schema) if data else pl.DataFrame(schema=schema)


def make_context() -> DemoContext:
    kills_rows, blinds_rows, shots_rows = [], [], []
    for i, sc in enumerate(SCENARIOS):
        t = 1000 + i * 600
        kx, ky = sc.killer_offset
        my_x = float(i * 400)  # 每个死亡点错开位置，热图不至于退化成单点
        kills_rows.append(
            {
                "tick": t,
                "attacker_steamid": KILLER_STEAMID,
                "attacker_name": KILLER_NAME,
                "attacker_x": my_x + float(kx),
                "attacker_y": float(ky),
                "attacker_z": 0.0,
                "victim_steamid": MY_STEAMID,
                "victim_name": MY_NAME,
                "victim_x": my_x,
                "victim_y": 0.0,
                "victim_z": 0.0,
                "weapon": "ak47",
                "headshot": False,
            }
        )
        if sc.blind_s > 0:
            blinds_rows.append(
                {
                    "tick": t - 100,
                    "victim_steamid": MY_STEAMID,
                    "victim_name": MY_NAME,
                    "attacker_steamid": KILLER_STEAMID,
                    "attacker_name": KILLER_NAME,
                    "duration": sc.blind_s,
                }
            )
        if sc.empty_clip:
            shots_rows.append(
                {
                    "tick": t - 30,
                    "steamid": MY_STEAMID,
                    "name": MY_NAME,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "yaw": float(sc.my_yaw),
                    "pitch": 0.0,
                    "weapon": "ak47",
                    "num_bullets_remaining": 0,
                }
            )

    tables = {
        "kills": _df(
            kills_rows,
            {
                "tick": pl.Int64,
                "attacker_steamid": pl.UInt64,
                "attacker_name": pl.Utf8,
                "attacker_x": pl.Float64,
                "attacker_y": pl.Float64,
                "attacker_z": pl.Float64,
                "victim_steamid": pl.UInt64,
                "victim_name": pl.Utf8,
                "victim_x": pl.Float64,
                "victim_y": pl.Float64,
                "victim_z": pl.Float64,
                "weapon": pl.Utf8,
                "headshot": pl.Boolean,
            },
        ),
        "blinds": _df(
            blinds_rows,
            {
                "tick": pl.Int64,
                "victim_steamid": pl.UInt64,
                "victim_name": pl.Utf8,
                "attacker_steamid": pl.UInt64,
                "attacker_name": pl.Utf8,
                "duration": pl.Float64,
            },
        ),
        "shots": _df(
            shots_rows,
            {
                "tick": pl.Int64,
                "steamid": pl.UInt64,
                "name": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
                "yaw": pl.Float64,
                "pitch": pl.Float64,
                "weapon": pl.Utf8,
                "num_bullets_remaining": pl.Int64,
            },
        ),
        "smokes": _df([], {"start_tick": pl.Int64, "end_tick": pl.Int64, "x": pl.Float64, "y": pl.Float64, "z": pl.Float64}),
        "fires": _df([], {"start_tick": pl.Int64, "end_tick": pl.Int64, "x": pl.Float64, "y": pl.Float64, "z": pl.Float64}),
        "rounds": _df(
            [{"round_num": 1, "start_tick": 0, "freeze_end_tick": 100, "end_tick": 100000}],
            {"round_num": pl.Int64, "start_tick": pl.Int64, "freeze_end_tick": pl.Int64, "end_tick": pl.Int64},
        ),
        "players": _df(
            [
                {"steamid": MY_STEAMID, "name": MY_NAME, "side": "counter-terrorist"},
                {"steamid": KILLER_STEAMID, "name": KILLER_NAME, "side": "terrorist"},
            ],
            {"steamid": pl.UInt64, "name": pl.Utf8, "side": pl.Utf8},
        ),
        "damages": _df([], {"tick": pl.Int64, "attacker_steamid": pl.UInt64, "victim_steamid": pl.UInt64}),
        "round_economy": _df([], {"round_num": pl.Int64, "side": pl.Utf8, "equipment_value": pl.Int64, "buy_type": pl.Utf8}),
    }
    return DemoContext(
        key="synthetic",
        path=Path("data/demos/me/synthetic.dem"),
        map_name=MAP,
        tick_rate=TICK_RATE,
        tables=tables,
        death_frames=pl.DataFrame(schema={"steamid": pl.UInt64, "tick": pl.Int64}),
        meta={},
    )


def _killer_yaw(kx: float, ky: float) -> float:
    """击杀者面向我（0,0,0）的 yaw。"""
    return math.degrees(math.atan2(0.0 - ky, 0.0 - kx)) % 360.0


def make_snap() -> pl.DataFrame:
    pre, post = 256, 64
    rows: list[dict] = []
    for i, sc in enumerate(SCENARIOS):
        t = death_ticks()[i]
        kx, ky = sc.killer_offset
        k_yaw = _killer_yaw(kx, ky)
        my_x = float(i * 400)
        for tick in range(t - pre, t + post + 1):
            rows.append(
                {
                    "tick": tick,
                    "steamid": MY_STEAMID,
                    "name": MY_NAME,
                    "side": "counter-terrorist",
                    "x": my_x,
                    "y": 0.0,
                    "z": 0.0,
                    "yaw": float(sc.my_yaw),
                    "pitch": 0.0,
                    "velocity": sc.my_vel,
                    "health": 100,
                    "armor": 100,
                    "is_crouched": False,
                    "is_blinded": sc.blind_s > 0,
                }
            )
            rows.append(
                {
                    "tick": tick,
                    "steamid": KILLER_STEAMID,
                    "name": KILLER_NAME,
                    "side": "terrorist",
                    "x": my_x + float(kx),
                    "y": float(ky),
                    "z": 0.0,
                    "yaw": k_yaw,
                    "pitch": 0.0,
                    "velocity": sc.killer_vel,
                    "health": 100,
                    "armor": 100,
                    "is_crouched": False,
                    "is_blinded": False,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "tick": pl.Int64,
            "steamid": pl.UInt64,
            "name": pl.Utf8,
            "side": pl.Utf8,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
            "yaw": pl.Float64,
            "pitch": pl.Float64,
            "velocity": pl.Float64,
            "health": pl.Int64,
            "armor": pl.Int64,
            "is_crouched": pl.Boolean,
            "is_blinded": pl.Boolean,
        },
    )


# --------------------------------------------------------------------------- 几何桩


class ScriptedGeo:
    """方向感知的几何桩：按 tick 与「观察者是我还是他」返回可见性。"""

    def __init__(self, cfg: GeomCfg, spec: MapSpec, meta: list[dict]):
        self.cfg = cfg
        self.spec = spec
        self.smokes = pl.DataFrame()
        self.meta = meta
        self._entries = [
            EntryPoint("entryA", (100.0, 0.0, 0.0)),
            EntryPoint("entryB", (0.0, 100.0, 0.0)),
            EntryPoint("entryC", (-100.0, 0.0, 0.0)),
        ]

    def _m(self, tick: int) -> dict | None:
        for m in self.meta:
            if m["start"] <= tick <= m["end"]:
                return m
        return None

    def visible(self, a, b, *, tick=None, crouch_a=False, crouch_b=False) -> bool:
        if tick is None:
            return True
        m = self._m(int(tick))
        if m is None:
            return True
        a3 = (float(a[0]), float(a[1]), float(a[2]))
        is_me_observer = dist3(a3, m["my_pos"]) <= dist3(a3, m["killer_pos"])
        threshold = m["me_from"] if is_me_observer else m["him_from"]
        return threshold is not None and int(tick) >= threshold

    def entry_points(self, pos: Vec3) -> list[EntryPoint]:
        return list(self._entries)

    def exposure(self, pos, *, tick=None, crouched=False):
        return 2, ["entryA", "entryB"]

    def crossfire(self, pos, *, tick=None, crouched=False):
        return 0

    def escape_time(self, pos) -> float:
        return 0.5


def make_geo(settings: Settings) -> ScriptedGeo:
    spec = settings.map_spec(MAP)
    pre, post = 256, 64
    meta: list[dict] = []
    for i, sc in enumerate(SCENARIOS):
        t = death_ticks()[i]
        kx, ky = sc.killer_offset
        my_x = float(i * 400)
        meta.append(
            {
                "start": t - pre,
                "end": t + post,
                "my_pos": (my_x, 0.0, 0.0),
                "killer_pos": (my_x + float(kx), float(ky), 0.0),
                "me_from": t + int(round(sc.me_from * TICK_RATE)) if sc.me_from is not None else None,
                "him_from": t + int(round(sc.him_from * TICK_RATE)) if sc.him_from is not None else None,
            }
        )
    return ScriptedGeo(settings.geom, spec, meta)


# --------------------------------------------------------------------------- 主流程


def run_selftest(settings: Settings, out_dir: Path | None = None) -> dict:
    ctx = make_context()
    snap = make_snap()
    spans = plan_spans(death_ticks(), TICK_RATE, settings.windows)
    geo = make_geo(settings)

    windows = build_death_windows(ctx, MY_STEAMID, spans, snap)
    records: list[DeathRecord] = []
    for w in windows:
        verdict = attribute(w, settings.rules, geo)
        rec = DeathRecord(w, verdict)
        rec.metrics = metrics.compute(w, geo, settings.rules)
        records.append(rec)

    frame = records_to_frame(records, demo=ctx.path.name)
    labels = frame["label"].to_list()
    expected = [s.label for s in SCENARIOS]

    errors: list[str] = []
    if labels != expected:
        errors.append(f"标签顺序不符：{labels} != {expected}")
    if "信息缺失硬拉" not in labels:
        errors.append("规则7（信息缺失硬拉）已启用，合成场景应命中它")
    if "打空/换弹" in labels:
        errors.append("v2 里空弹匣不再是死因标签，不应产出「打空/换弹」")

    peek = frame.filter(pl.col("label") == "被拉死")
    if peek.is_empty():
        errors.append("没有「被拉死」样本")
    else:
        gap = peek["first_seen_gap_ms"][0]
        if gap is None or gap <= 0:
            errors.append(f"「被拉死」的先手差应为正（他先看见我），实得 {gap}")

    out = Path(out_dir or settings.paths.output / "selftest")
    out.mkdir(parents=True, exist_ok=True)
    report.write_deaths_csv(frame, out / "deaths.csv")
    summary = report.summarize(frame)
    report.write_report_md(
        frame,
        summary,
        out / "report.md",
        player=f"{MY_NAME}(synthetic)",
        caveats=["selftest 合成数据，几何为桩；先手差用 FOV(90°)"],
        cfg=settings.report,
    )
    plot.heatmap([(r.window.my_pos[0], r.window.my_pos[1]) for r in records], out / "heatmap_me.png")
    for idx, rec in enumerate(records):
        plot.death_frame(rec, out / f"death_{idx}.png")

    return {
        "labels": labels,
        "errors": errors,
        "total": frame.height,
        "out": str(out),
        "gap_peeked": peek["first_seen_gap_ms"][0] if not peek.is_empty() else None,
    }
