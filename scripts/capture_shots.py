"""netcon 驱动采集（N7-B / T2+T3）。

按 capture_plan.json 逐个采集：playdemo → demo_gototick → spec_player → demo_pause → jpeg。
截图从 CS2 截图目录收集改名到 output/<demo>/shots/<tick>_<kind>.jpg。

用法：
  .venv python scripts/capture_shots.py --plan <capture_plan.json> --demo <demo名> --out <目录> --player <你的昵称> [--resume]

CS2 截图目录默认探测常见 Steam 库位置，也可用环境变量 ``CS2_CSGO_DIR``
或 ``--csgo-dir`` 指定（指向 game/csgo）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csc.netcon import CS2Netcon, ping


def _default_csgo() -> Path:
    env = os.environ.get("CS2_CSGO_DIR")
    if env:
        return Path(env)
    for cand in (
        r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo",
        r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo",
    ):
        if Path(cand).exists():
            return Path(cand)
    return Path(cand)  # 都不存在就用标准路径，报错信息里能看出来


CS2_CSGO = _default_csgo()
SHOT_DIRS = [CS2_CSGO, CS2_CSGO / "screenshots"]


def find_new_shots(before: set[str]) -> list[Path]:
    out = []
    for d in SHOT_DIRS:
        if not d.exists():
            continue
        for pat in ("*.jpg", "*.png", "*.tga"):
            out.extend(Path(p) for p in glob.glob(str(d / pat)))
    return [p for p in out if str(p) not in before]


def snapshot_state() -> set[str]:
    s = set()
    for d in SHOT_DIRS:
        if not d.exists():
            continue
        for pat in ("*.jpg", "*.png", "*.tga"):
            s.update(glob.glob(str(d / pat)))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--demo", required=True, help="demo 文件名（相对 game/csgo）或绝对路径")
    ap.add_argument("--out", required=True, help="输出目录（shots/ 子目录）")
    ap.add_argument("--player", required=True, help="你的游戏昵称（spec_player 用）")
    ap.add_argument("--csgo-dir", default=None, help="CS2 的 game/csgo 目录（默认探测常见 Steam 库）")
    ap.add_argument("--port", type=int, default=2121)
    ap.add_argument("--password", default=None)
    ap.add_argument("--settle", type=float, default=0.8, help="跳转后等待渲染秒数")
    ap.add_argument("--resume", action="store_true", help="已存在同名就跳过")
    args = ap.parse_args()

    if args.csgo_dir:
        _csgo = Path(args.csgo_dir)
        SHOT_DIRS[:] = [_csgo, _csgo / "screenshots"]

    out_dir = Path(args.out)
    shots_dir = out_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))["jobs"]
    print(f"[*] 计划 {len(plan)} 个任务")

    if not ping(port=args.port):
        print(f"[!] netcon 未监听。启动 CS2：cs2.exe -console -novid -netconport {args.port}")
        return 1

    done = 0
    skipped = 0
    with CS2Netcon(port=args.port, password=args.password) as nc:
        nc.send("cl_draw_only_deathnotices 1")
        nc.send("spec_freeze_time 0")
        print(f"[*] playdemo {args.demo}")
        nc.send(f"playdemo {args.demo}")
        time.sleep(5)

        for i, job in enumerate(plan, 1):
            tick = job["tick"]
            kind = job["kind"]
            dest = shots_dir / f"{tick}_{kind}.jpg"
            if args.resume and dest.exists():
                skipped += 1
                continue

            before = snapshot_state()
            nc.send(f"demo_gototick {tick}")
            time.sleep(args.settle)
            nc.send(f"spec_player {args.player}")
            nc.send("spec_mode 1")
            nc.send("demo_pause")
            time.sleep(0.3)
            fname = f"{tick}_{kind}"
            nc.send(f"jpeg {fname} 100")
            time.sleep(0.5)

            new = find_new_shots(before)
            if new:
                # 取最新的
                src = max(new, key=lambda p: p.stat().st_mtime)
                shutil.copy2(src, dest)
                src.unlink(missing_ok=True)
                done += 1
                print(f"  [{i}/{len(plan)}] {tick}_{kind} -> {dest.name}")
            else:
                print(f"  [{i}/{len(plan)}] {tick}_{kind} [!] 未捕获到截图")

        nc.send("stopdemo")

    print(f"\n[完成] 采集 {done} 张，跳过 {skipped} 张 -> {shots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
