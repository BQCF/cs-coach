"""N7-A / T0：netcon 自检脚本。

用法：
  1. 启动 CS2：cs2.exe -console -novid -netconport 2121
  2. 把 demo 复制到 game/csgo/（或用绝对路径）
  3. .venv python scripts/netcon_selfcheck.py --demo <demo名>
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csc.netcon import CS2Netcon, ping

CS2_CSGO = Path(r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo")
SCREENSHOT_HINTS = [CS2_CSGO / "*.jpg", CS2_CSGO / "screenshots" / "*.jpg",
                    CS2_CSGO / "*.tga", CS2_CSGO / "*.png"]


def find_screenshots() -> list[Path]:
    out = []
    for pat in SCREENSHOT_HINTS:
        out.extend(glob.glob(str(pat)))
    return [Path(p) for p in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="demo 文件名（相对 game/csgo）或绝对路径")
    ap.add_argument("--player", required=True, help="你的游戏昵称（spec_player 用）")
    ap.add_argument("--tick", type=int, default=3301)
    ap.add_argument("--port", type=int, default=2121)
    ap.add_argument("--password", default=None)
    args = ap.parse_args()

    print(f"[*] 检查 netcon 端口 {args.port} ...")
    if not ping(port=args.port):
        print(f"[!] 未监听。请启动 CS2：cs2.exe -console -novid -netconport {args.port}")
        return 1
    print("[+] 端口在监听")

    before = set(find_screenshots())
    print(f"[*] 截图前已有 {len(before)} 张 jpg")

    with CS2Netcon(port=args.port, password=args.password) as nc:
        print("[+] 已连接 netcon")
        nc.send_wait("echo cs-coach T0 selfcheck", 0.3)
        print("[*] 已发 echo，游戏控制台应能看到")

        demo_cmd = args.demo if str(args.demo).startswith(("C:", "D:", "E:", "/")) else args.demo
        print(f"[*] playdemo {demo_cmd} ...")
        nc.send(f"playdemo {demo_cmd}")
        time.sleep(4)  # 等 demo 加载

        print(f"[*] demo_gototick {args.tick} ...")
        nc.send(f"demo_gototick {args.tick}")
        time.sleep(1.0)
        nc.send("demo_pause")
        time.sleep(0.5)

        print(f"[*] spec_player {args.player} ...")
        nc.send(f"spec_player {args.player}")
        time.sleep(0.3)

        for mode in (1, 0, 2):
            print(f"[*] spec_mode {mode}")
            nc.send(f"spec_mode {mode}")
            time.sleep(0.3)

        print("[*] jpeg test 100 ...")
        nc.send("jpeg test 100")
        time.sleep(1.5)

        nc.send("stopdemo")
        time.sleep(0.5)

    time.sleep(1.0)
    after = set(find_screenshots())
    new = after - before
    print(f"\n[结果]")
    print(f"  新增截图 {len(new)} 张：")
    for p in sorted(new):
        print(f"    {p}")
    if new:
        print("\n[+] T0 通过：把上面的路径和文件名模式写死进 config（T3 采集的前提）")
    else:
        print("[!] 没找到新截图，可能 jpeg 输出格式不同（试试 .tga/.png），或需要 CS2 前台运行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
