"""第一人称示意图规划：绿十字=敌人冒头处（预瞄点），红X=死时位置。"""

from __future__ import annotations

import dataclasses

import polars as pl

from csc import config, deaths, fault, firstperson, synthetic, windows


def _recs():
    s = config.load_settings()
    ctx = synthetic.make_context()
    snap = synthetic.make_snap()
    spans = windows.plan_spans(synthetic.death_ticks(), synthetic.TICK_RATE, s.windows)
    geo = synthetic.make_geo(s)
    ws = deaths.build_death_windows(ctx, synthetic.MY_STEAMID, spans, snap)
    t_pos, t_pre = synthetic.death_ticks()[4], synthetic.death_ticks()[7]
    recs = []
    for w in ws:
        # 击杀者轨迹整体平移：冒头位置 != 死时位置，绿十字才有区分意义
        kt = w.killer_track.with_columns((pl.col("x") - 300.0).alias("x")) if w.killer_track is not None else None
        w2 = dataclasses.replace(w, killer_track=kt)
        rec = deaths.DeathRecord(w2, deaths.attribute(w2, s.rules, geo))
        if w2.death_tick == t_pos:
            rec.fault = fault.FaultVerdict(fault="position", sub="exposed", conf="high", evidence="")
        elif w2.death_tick == t_pre:
            rec.fault = fault.FaultVerdict(fault="preaim", sub="", conf="mid", evidence="")
        else:
            rec.fault = fault.FaultVerdict(fault="none", sub="", conf="low", evidence="")
        recs.append(rec)
    return s, geo, recs, t_pos, t_pre


def test_plan_marks_files_and_ticks():
    s, geo, recs, t_pos, t_pre = _recs()
    shots = firstperson.plan_fp_shots(recs, geo, s.rules)
    by_kind = {sh["kind"]: sh for sh in shots}
    assert set(by_kind) == {"position", "preaim"}
    assert by_kind["position"]["tick"] == t_pos
    assert by_kind["position"]["file"] == f"fp_position_{t_pos}.png"
    assert by_kind["preaim"]["file"] == f"fp_preaim_{t_pre}.png"
    assert f"tick {t_pos}" in by_kind["position"]["annotate"]
    for sh in shots:
        enemy_pos, _c1, _l1, enemy_kind = sh["marks"][0]
        aim_pos, _c2, _l2, aim_kind = sh["marks"][1]
        assert enemy_kind == "enemy" and aim_kind == "aim"
        assert aim_pos != enemy_pos  # 绿十字=冒头点，不是死时位置


def test_plan_skips_unlabeled_and_respects_cap():
    s, geo, recs, t_pos, t_pre = _recs()
    shots = firstperson.plan_fp_shots(recs, geo, s.rules, max_each=1)
    assert len(shots) == 2  # 每类最多 1 张
    assert all(sh["tick"] in (t_pos, t_pre) for sh in shots)  # fault=none 不出图


# --------------------------------------------------------------------------- 真截图投影


def test_project_marks_center_and_sides():
    eye = (0.0, 0.0, 64.0)
    marks = [
        ((500.0, 0.0, 64.0), "#e74c3c", "敌人在这", "enemy"),     # 正前方 -> 图心
        ((500.0, 500.0, 64.0), "#2ecc71", "该瞄这里", "aim"),    # 右前方 -> 图右
        ((-500.0, 0.0, 64.0), "#fff", "背后", "mark"),           # 背后 -> 丢弃
    ]
    out = firstperson.project_marks(marks, eye, 0.0, 0.0, 1600, 900)
    assert len(out) == 2  # 背后的点被丢弃
    front = next(m for m in out if m[3] == "敌人在这")
    right = next(m for m in out if m[3] == "该瞄这里")
    assert abs(front[0] - 800) < 1 and abs(front[1] - 450) < 1  # 正前方在图心
    assert right[0] > 800  # 正右方在图右半边


def test_project_marks_wide_aspect_pulls_toward_center():
    """CS 宽屏 FOV 放大：同一方向在宽屏图上更靠中心。"""
    eye = (0.0, 0.0, 64.0)
    marks = [((500.0, 200.0, 64.0), "#e74c3c", "敌人在这", "enemy")]
    narrow = firstperson.project_marks(marks, eye, 0.0, 0.0, 1600, 1200)   # 4:3
    wide = firstperson.project_marks(marks, eye, 0.0, 0.0, 2400, 1000)     # 21:9
    assert abs(wide[0][0] - 1200) < abs(narrow[0][0] - 800) * (2400 / 1600)


def test_find_real_shot_by_tick(ws_tmp):
    (ws_tmp / "121523.png").write_bytes(b"png")
    (ws_tmp / "other.jpg").write_bytes(b"jpg")
    hit = firstperson.find_real_shot(ws_tmp, 121523)
    assert hit is not None and hit.name == "121523.png"
    assert firstperson.find_real_shot(ws_tmp, 999) is None
    assert firstperson.find_real_shot(ws_tmp / "missing", 1) is None
