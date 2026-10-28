"""一键报告脚本（N7 全自动）：生成报告时自动启动 CS2 截真图。

流程：
  1. 启动 CS2（-netconport 2121）—— 已在跑就复用
  2. 等 netcon 端口就绪
  3. 生成采集计划（shots.py）
  4. 自动采集真截图（capture_shots.py）
  5. 重跑报告（真图优先，几何图兜底）
  6. 重跑 profile（六维图）

用法：
  python scripts/make_report.py --demo <demo.dem> --player <你的昵称> --map de_inferno --out <输出目录>

注意：
  - demo 文件名传给 CS2 的 playdemo，相对 game/csgo 目录（会自动复制进去）
  - CS2 首次启动约需 10-20 秒进入主菜单，脚本会等端口
  - 采集时 CS2 窗口必须前台（脚本会提示）
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csc.netcon import ping

CS2_ROOT = Path(r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive")
CS2_EXE = CS2_ROOT / "game" / "bin" / "win64" / "cs2.exe"
CS2_CSGO = CS2_ROOT / "game" / "csgo"
NETCON_PORT = 2121

CS_COACH = Path(__file__).resolve().parents[1]


def is_cs2_running() -> bool:
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq cs2.exe"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return "cs2.exe" in (r.stdout or "")


def launch_cs2() -> None:
    print(f"[*] 启动 CS2：{CS2_EXE}")
    print(f"    参数：-console -novid -netconport {NETCON_PORT}")
    subprocess.Popen(
        [str(CS2_EXE), "-console", "-novid", f"-netconport{NETCON_PORT}"],
        cwd=str(CS2_EXE.parent),
    )
    print("[*] CS2 启动中（首次约需 10-20 秒进主菜单）...")


def wait_for_netcon(timeout_s: int = 60) -> bool:
    print(f"[*] 等待 netcon 端口 {NETCON_PORT} 就绪（最多 {timeout_s} 秒）...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if ping(port=NETCON_PORT):
            print("[+] netcon 就绪")
            return True
        time.sleep(2)
    print(f"[!] {timeout_s} 秒内 netcon 未就绪")
    return False


def copy_demo_to_csgo(demo: Path) -> str:
    """把 demo 复制进 game/csgo（失败就用绝对路径，playdemo 支持绝对路径）。"""
    dest = CS2_CSGO / demo.name
    try:
        if not dest.exists():
            print(f"[*] 复制 demo 到 game/csgo：{demo.name}")
            shutil.copy2(demo, dest)
        return demo.name
    except PermissionError:
        print(f"[*] game/csgo 不可写，用绝对路径 playdemo")
        return str(demo)


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd or CS_COACH))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="demo 文件路径")
    ap.add_argument("--player", required=True, help="玩家名或 steamid")
    ap.add_argument("--map", required=True, help="地图名（de_inferno/de_mirage/de_ancient）")
    ap.add_argument("--out", required=True, help="报告输出目录")
    ap.add_argument("--skip-shots", action="store_true", help="跳过真截图采集（直接用已有）")
    ap.add_argument("--no-launch", action="store_true", help="不自动启动 CS2（假定已带 netcon 运行）")
    args = ap.parse_args()

    demo = Path(args.demo)
    if not demo.exists():
        print(f"[!] demo 不存在：{demo}")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    py = str(CS_COACH / ".venv" / "Scripts" / "python.exe")

    # ---- 1. 启动 CS2 ----
    if not args.no_launch:
        if is_cs2_running():
            print("[*] CS2 已在运行")
            if not ping(port=NETCON_PORT):
                print(f"[!] 但 netcon 端口 {NETCON_PORT} 未开——请关闭 CS2 后重跑（netcon 必须启动时带端口参数）")
                return 1
        else:
            launch_cs2()
            if not wait_for_netcon():
                print("[!] CS2 未就绪，请手动启动后用 --no-launch 重跑")
                return 1

    # ---- 2. 复制 demo 到 game/csgo ----
    demo_name = copy_demo_to_csgo(demo)

    # ---- 3. 采集真截图 ----
    if not args.skip_shots:
        plan = out_dir / "capture_plan.json"
        print("\n===== 生成采集计划 =====")
        run([py, "-m", "csc.shots", "--deaths", str(out_dir / "deaths.csv"), "--out", str(out_dir)])
        if not plan.exists():
            print("[!] 采集计划生成失败")
            return 1
        print("\n===== 采集真截图（约 1-2 分钟，请勿切走 CS2 窗口）=====")
        rc = run([py, str(CS_COACH / "scripts" / "capture_shots.py"),
                  "--plan", str(plan), "--demo", demo_name,
                  "--out", str(out_dir), "--player", args.player, "--resume"])
        if rc != 0:
            print("[!] 采集失败（CS2 需前台，netcon 需就绪）")

    # ---- 4. 重跑报告 ----
    print("\n===== 重跑报告（真图优先）=====")
    run([py, "-m", "csc", "--output", str(out_dir), "report",
         "--deaths", str(out_dir / "deaths.csv"), "--player", args.player, "--map", args.map])

    # ---- 5. 重跑 profile ----
    print("\n===== 重跑画像（六维图）=====")
    run([py, "-m", "csc", "profile", "--deaths", str(out_dir / "deaths.csv"),
         "--player", args.player, "--out", str(out_dir / "profile")])

    shots = list((out_dir / "shots").glob("*.jpg")) if (out_dir / "shots").exists() else []
    print(f"\n[完成] 报告目录：{out_dir}")
    print(f"  真截图：{len(shots)} 张")
    print(f"  打开：{out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
