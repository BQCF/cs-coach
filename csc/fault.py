"""责任判定（N1）：这一枪输在谁身上？

fault 五类（用户要的正是这个）：
* ``position`` 站位有问题（他先看见我 + 暴露面/交叉/退路超标）
* ``preaim``   预瞄有问题（他出现时我准星不在他能出现的方向）
* ``aim``      枪法有问题（我先看见/先开火，但打空 or 伤害不足）——用户第三条诉求
* ``info``     信息有问题（没看到就推进/被拉）
* ``none``     无解（被闪、残血被秒、peeker's advantage）

严格按证据顺序判定，命中即停。没有数字支撑不给结论。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FaultCfg, RulesCfg
from .duel import DuelFacts


@dataclass(frozen=True)
class FaultVerdict:
    fault: str          # position / preaim / aim / info / none
    sub: str            # whiff / low_dmg / late / wide / exposed / no_escape / ""
    conf: str           # high / mid / low
    evidence: str       # 人话证据串（含数字）

    def to_dict(self) -> dict[str, str]:
        return {"fault": self.fault, "fault_sub": self.sub, "fault_conf": self.conf,
                "fault_evidence": self.evidence}


def classify_fault(w: Any, m: Any, d: DuelFacts, cfg: RulesCfg, fcfg: FaultCfg, verdict: Any = None) -> FaultVerdict:
    """返回 (fault, sub, conf, evidence)。m=Metrics, d=DuelFacts。"""
    gap = getattr(m, "first_seen_gap_ms", None)
    mine_first = gap is not None and gap < 0   # 严格为负才算我先看见（0=同时）

    vlabel = getattr(verdict, "label", "") or ""
    blind = (vlabel == "被闪") or (getattr(w, "is_blinded", False))
    hp = d.my_hp_at_engage if d.my_hp_at_engage is not None else 100

    # A. 完全无解：被闪 / 残血被秒
    if blind or hp <= fcfg.oneshot_hp:
        why = "被闪致盲" if blind else f"残血 {hp}"
        return FaultVerdict("none", "", "high", f"{why}，无交火空间")

    # B0. 明确打空（whiff）：我开了枪却一枪没中 = 枪法问题，优先于先手判定
    if d.my_shots >= 2 and d.my_hits == 0:
        return FaultVerdict("aim", "whiff", "high",
                            f"我开 {d.my_shots} 枪 0 中（他反打 {d.his_dmg} 伤）")

    # B. 枪法：我先看见 或 我先开火，但打空/伤害不够（用户第三条诉求核心）
    if mine_first or d.first_fire == "me":
        lead = "先看见" if mine_first else "先开火"
        if d.my_shots >= 2 and d.my_hits == 0:
            return FaultVerdict("aim", "whiff", "high",
                                f"{lead} {d.my_shots} 枪 0 中（他反打 {d.his_dmg} 伤）")
        if d.my_hits >= 1 and d.his_shots >= 1:
            need = d.his_hp_at_engage if d.his_hp_at_engage is not None else 100
            if d.my_dmg < need:
                return FaultVerdict("aim", "low_dmg", "high",
                                    f"打中 {d.my_hits} 枪共 {d.my_dmg} 伤，他 {need} 血没死")
            if d.his_dmg >= hp:
                return FaultVerdict("aim", "low_dmg", "mid",
                                    f"{lead} {d.my_dmg} 伤，他 {d.his_dmg} 伤先到致命")
        if d.first_fire == "me" and d.my_shots >= 1 and d.my_react_ms is not None \
                and d.my_react_ms > fcfg.slow_react_ms:
            return FaultVerdict("aim", "late", "mid", f"他出现 {d.my_react_ms}ms 后我才开枪")
        if d.my_shots >= 1:
            return FaultVerdict("aim", "", "mid",
                                f"{lead}但交换不利（我 {d.my_dmg} 对他 {d.his_dmg}）")
        return FaultVerdict("none", "", "mid",
                            f"{lead}但我未及开枪就被秒（他 {d.his_dmg} 伤）")

    # C. 预瞄：他先看见我，且他出现的方向不在我准星上
    if gap is not None and gap > 0:
        off = d.my_aim_off_at_see_deg
        if off is not None and off > fcfg.preaim_bad_deg:
            return FaultVerdict("preaim", "wide", "high", f"他出现时我准星偏 {off:.0f}°")
        if off is not None and off <= fcfg.preaim_ok_deg:
            if d.my_react_ms is not None and d.my_react_ms > fcfg.slow_react_ms:
                return FaultVerdict("aim", "late", "mid", f"准星对了但 {d.my_react_ms}ms 没开枪")
            return FaultVerdict("none", "", "mid", "准星到位、反应正常，被 peeker's advantage 吃掉")

    # D0. 配置问题：退路超 5s = 掩体点没配对，不是玩家站位问题
    if (getattr(m, "escape_time", 0) or 0) > fcfg.escape_config_max:
        return FaultVerdict("position_config", "no_cover", "high",
                            f"退路 {m.escape_time}s 过长，是地图掩体配置问题（非站位）")

    # D. 站位：他先看见我 + 几何三项任一超标
    flags = []
    if (getattr(m, "exposure", 0) or 0) >= fcfg.exposure_bad:
        flags.append(f"暴露面 {m.exposure}")
    if (getattr(m, "crossfire", 0) or 0) >= 1:
        flags.append(f"交叉 {m.crossfire}")
    if (getattr(m, "escape_time", 0) or 0) > fcfg.escape_bad:
        flags.append(f"退路 {m.escape_time}s")
    if flags:
        return FaultVerdict("position", "exposed", "high", "；".join(flags))

    # E. 信息：全程未见 / 未开枪
    if d.my_aim_off_at_see_deg is None or d.my_shots == 0:
        return FaultVerdict("info", "", "low", "全程未见有效信息 / 未开枪")
    return FaultVerdict("position", "no_escape", "low", "无掩体可退")
