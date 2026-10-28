"""采集计划生成（N7-B / T1，纯 Python 不碰 CS2）。

每个死亡点出 2 张：
* contact —— 他首次进我视野那一刻（看预瞄）
* death    —— 死亡前 pre_ticks tick（看结果，避开 killcam）

写到 output/<demo>/capture_plan.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def build_capture_plan(deaths_csv: Path, *, pre_ticks: int = 4) -> list[dict]:
    df = pl.read_csv(deaths_csv)
    plan: list[dict] = []
    for r in df.iter_rows(named=True):
        tick = int(r["death_tick"])
        plan.append({
            "tick": tick,
            "kind": "death",
            "fault": r.get("fault"),
            "zone": r.get("zone"),
            "mechanism": r.get("mechanism"),
        })
        t_see = r.get("t_see_him")
        if t_see is not None and not (isinstance(t_see, float) and t_see != t_see):
            try:
                plan.append({
                    "tick": int(t_see),
                    "kind": "contact",
                    "fault": r.get("fault"),
                    "zone": r.get("zone"),
                    "mechanism": r.get("mechanism"),
                })
            except (TypeError, ValueError):
                pass
    return plan


def write_plan(plan: list[dict], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "capture_plan.json"
    p.write_text(json.dumps({"jobs": plan}, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--deaths", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan = build_capture_plan(Path(args.deaths))
    p = write_plan(plan, Path(args.out))
    n_death = sum(1 for j in plan if j["kind"] == "death")
    n_contact = sum(1 for j in plan if j["kind"] == "contact")
    print(f"采集计划：{len(plan)} 个任务（death {n_death} + contact {n_contact}）-> {p}")


if __name__ == "__main__":
    main()
