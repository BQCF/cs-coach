"""对枪重建单测（N1，计划 §5.4）：用固化合成 shots/damages 断言 first_fire / my_hits / 伤害口径。"""

from __future__ import annotations

import polars as pl

from csc import config, duel, fault
from csc.deaths import DeathWindow


def _mk_window(shots, damages, *, my=111, killer=222):
    return DeathWindow(
        death_tick=1000, round_num=1, tick_rate=64,
        my_steamid=my, my_name="me", killer_steamid=killer, killer_name="him",
        weapon="ak47", headshot=False,
        my_pos=(0.0, 0.0, 0.0), killer_pos=(200.0, 0.0, 0.0),
        my_yaw=0.0, killer_yaw=180.0, my_pitch=0.0, killer_pitch=0.0,
        my_speed=0.0, killer_speed=0.0, my_crouched=False, killer_crouched=False,
        shots=shots, damages=damages,
        span_start=700, span_end=1010,
    )


def _shots_df(rows):
    return pl.DataFrame(rows, schema={
        "tick": pl.Int64, "steamid": pl.UInt64, "num_bullets_remaining": pl.Int64,
        "x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "pitch": pl.Float64, "yaw": pl.Float64,
    })


def _dmg_df(rows):
    return pl.DataFrame(rows, schema={
        "tick": pl.Int64, "attacker_steamid": pl.UInt64, "victim_steamid": pl.UInt64,
        "dmg_health": pl.Int64, "hitgroup_name": pl.Utf8, "health_pre": pl.Int64,
    })


def test_first_fire_me_and_hits():
    cfg = config.load_settings().rules
    # 我先开火 3 枪、2 中 50 伤；他后开火 2 枪、1 中 30 伤
    shots = _shots_df([
        {"tick": 950, "steamid": 111, "num_bullets_remaining": 25, "x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0},
        {"tick": 955, "steamid": 111, "num_bullets_remaining": 24, "x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0},
        {"tick": 960, "steamid": 111, "num_bullets_remaining": 23, "x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0},
        {"tick": 970, "steamid": 222, "num_bullets_remaining": 25, "x": 200, "y": 0, "z": 0, "pitch": 0, "yaw": 180},
        {"tick": 980, "steamid": 222, "num_bullets_remaining": 24, "x": 200, "y": 0, "z": 0, "pitch": 0, "yaw": 180},
    ])
    dmg = _dmg_df([
        {"tick": 952, "attacker_steamid": 111, "victim_steamid": 222, "dmg_health": 30, "hitgroup_name": "chest", "health_pre": 100},
        {"tick": 958, "attacker_steamid": 111, "victim_steamid": 222, "dmg_health": 20, "hitgroup_name": "chest", "health_pre": 70},
        {"tick": 975, "attacker_steamid": 222, "victim_steamid": 111, "dmg_health": 30, "hitgroup_name": "chest", "health_pre": 100},
    ])
    w = _mk_window(shots, dmg)
    d = duel.build_duel(w, None, cfg)
    assert d.first_fire == "me"
    assert d.my_shots == 3 and d.his_shots == 2
    assert d.my_hits == 2 and d.his_hits == 1
    assert d.my_dmg == 50 and d.his_dmg == 30
    # health_pre 归属：我打他=他的血(100)，他打我=我的血(100)
    assert d.his_hp_at_engage == 100 and d.my_hp_at_engage == 100


def test_whiff_fault():
    """我先开火 5 枪 0 中 = aim/whiff。"""
    s = config.load_settings()
    shots = _shots_df([
        {"tick": 950 + i, "steamid": 111, "num_bullets_remaining": 30 - i, "x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0}
        for i in range(5)
    ])
    dmg = _dmg_df([
        {"tick": 960, "attacker_steamid": 222, "victim_steamid": 111, "dmg_health": 100, "hitgroup_name": "head", "health_pre": 100},
    ])
    w = _mk_window(shots, dmg)
    d = duel.build_duel(w, None, s.rules)
    fv = fault.classify_fault(w, None, d, s.rules, s.fault, None)
    assert fv.fault == "aim" and fv.sub == "whiff"
    assert "0 中" in fv.evidence


def test_health_pre_direction():
    """health_pre 由 attacker 决定归属，别搞混（计划 §9 坑2）。"""
    cfg = config.load_settings().rules
    dmg = _dmg_df([
        # 我打他：health_pre 是他的血 60
        {"tick": 950, "attacker_steamid": 111, "victim_steamid": 222, "dmg_health": 20, "hitgroup_name": "chest", "health_pre": 60},
        # 他打我：health_pre 是我的血 40
        {"tick": 960, "attacker_steamid": 222, "victim_steamid": 111, "dmg_health": 20, "hitgroup_name": "chest", "health_pre": 40},
    ])
    w = _mk_window(_shots_df([]), dmg)
    d = duel.build_duel(w, None, cfg)
    assert d.his_hp_at_engage == 60
    assert d.my_hp_at_engage == 40
