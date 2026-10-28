"""死亡 tick -> 回合号：落在回合间隙时回退到最近开始的回合，不再查空。"""

from __future__ import annotations

import polars as pl

from csc import deaths

ROUNDS = pl.DataFrame(
    [
        {"round_num": 1, "start_tick": 0, "end_tick": 1000},
        {"round_num": 2, "start_tick": 1200, "end_tick": 2000},
    ]
)


def test_tick_inside_round():
    assert deaths._round_for_tick(ROUNDS, 500) == 1
    assert deaths._round_for_tick(ROUNDS, 1500) == 2


def test_tick_in_gap_falls_back_to_previous_round():
    # 1000~1200 是回合间隙（回合结束到下回合开始之间）
    assert deaths._round_for_tick(ROUNDS, 1100) == 1


def test_tick_before_first_round_is_none():
    assert deaths._round_for_tick(ROUNDS, -5) is None
    assert deaths._round_for_tick(pl.DataFrame(), 5) is None
