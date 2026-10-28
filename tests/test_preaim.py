"""N3 预瞄度量单测：pitch 符号 + 分布统计。"""

from __future__ import annotations

from csc.preaim import aim_stats, pitch_err_deg, pitch_needed_deg


def test_pitch_sign_down_positive():
    """目标在下方（dz<0）→ 向下看 pitch>0（符号已实测验证 30/30）。"""
    need = pitch_needed_deg((0.0, 0.0, 100.0), (300.0, 0.0, 0.0))  # 目标低 100u
    assert need > 0  # 向下为正


def test_pitch_sign_up_negative():
    need = pitch_needed_deg((0.0, 0.0, 0.0), (300.0, 0.0, 100.0))  # 目标高 100u
    assert need < 0  # 向上为负


def test_pitch_err_zero_when_on_target():
    need = pitch_needed_deg((0.0, 0.0, 100.0), (300.0, 0.0, 0.0))
    err = pitch_err_deg((0.0, 0.0, 100.0), need, (300.0, 0.0, 0.0))
    assert err is not None and err < 1.0


def test_aim_stats():
    st = aim_stats([1.0, 2.0, 3.0, 20.0, 30.0])
    assert st["n"] == 5
    assert st["median"] == 3.0
    assert st["over_pct"] == 40.0  # 20,30 超 10°
    assert aim_stats([])["n"] == 0
