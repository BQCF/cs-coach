"""SendKeys 方案 A 实测：启动 CS2 → 开控制台 → 发 echo → 截一张 jpeg。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csc import sendkeys

CS2_ROOT = Path(r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive")
CS2_EXE = CS2_ROOT / "game" / "bin" / "win64" / "cs2.exe"


def main():
    print("[1] 检查 CS2 进程")
    pid = sendkeys.find_cs2_pid()
    if pid:
        print(f"  CS2 已在运行 PID={pid}")
    else:
        print("  启动 CS2（-console -novid -insecure）...")
        import subprocess

        subprocess.Popen([str(CS2_EXE), "-console", "-novid", "-insecure"],
                         cwd=str(CS2_EXE.parent))
        print("  等 CS2 启动 30 秒...")
        time.sleep(30)
        pid = sendkeys.find_cs2_pid()
        print(f"  CS2 PID={pid}")

    if not pid:
        print("[!] CS2 没起来")
        return 1

    print("[2] 带到前台")
    sendkeys.focus_window(pid=pid)
    time.sleep(1)

    print("[3] 开控制台（按 `~`）")
    sendkeys.open_console()
    time.sleep(1)

    print("[4] 发 echo test")
    sendkeys.send_command("echo cs-coach-sendkeys-test")
    time.sleep(0.5)

    print("[5] 截一张 jpeg")
    sendkeys.send_command("jpeg sendkeys_test 100")
    time.sleep(2)

    print("[6] 关控制台")
    sendkeys.open_console()

    # 查找截图
    print("[7] 查找截图...")
    import glob

    hits = []
    for d in [CS2_ROOT / "game" / "csgo", CS2_ROOT / "game" / "csgo" / "screenshots"]:
        if d.exists():
            hits.extend(glob.glob(str(d / "*.jpg")))
            hits.extend(glob.glob(str(d / "*.png")))
    hits = [h for h in hits if "sendkeys_test" in Path(h).name]
    if hits:
        print(f"  [+] 找到截图：{hits}")
    else:
        print("  [!] 未找到 sendkeys_test 截图，列出最新 5 张：")
        all_shots = []
        for d in [CS2_ROOT / "game" / "csgo", CS2_ROOT / "game" / "csgo" / "screenshots"]:
            if d.exists():
                all_shots.extend(glob.glob(str(d / "*.jpg")) + glob.glob(str(d / "*.png")))
        all_shots.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
        for p in all_shots[:5]:
            print(f"    {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
