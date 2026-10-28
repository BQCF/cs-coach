"""SendInput 诊断：验证键盘注入是否被 Windows 接收（先用记事本做基准）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csc import sendkeys


def main():
    print("[1] 启动记事本")
    import subprocess

    subprocess.Popen(["notepad.exe"])
    time.sleep(2)

    print("[2] 找记事本 PID")
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq notepad.exe"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("  tasklist:", (r.stdout or "")[:200])

    print("[3] 带到前台（AppActivate by PID）")
    pid = sendkeys.find_cs2_pid()  # 临时复用
    # 直接找 notepad
    r2 = subprocess.run(["tasklist", "/FI", "IMAGENAME eq notepad.exe"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    npid = None
    for line in (r2.stdout or "").splitlines():
        if "notepad.exe" in line:
            parts = line.split()
            try:
                npid = int(parts[1])
                break
            except (IndexError, ValueError):
                pass
    print(f"  notepad PID={npid}")
    if npid:
        sendkeys.focus_window(pid=npid)
        time.sleep(1)

    print("[4] 发 HELLO_TEST")
    sendkeys.type_text("HELLO_TEST_ABC")
    time.sleep(1)

    print("[5] 截屏看记事本")
    import subprocess as sp

    from pathlib import Path as _Path

    _out = _Path(__file__).resolve().parents[1] / ".tmp" / "sendkeys_diag.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    sp.run(["powershell", "-Command",
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
            "$b = New-Object System.Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, "
            "[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); "
            "$g = [System.Drawing.Graphics]::FromImage($b); "
            "$g.CopyFromScreen(0,0,0,0,$b.Size); "
            f"$b.Save('{str(_out)}')"],
           capture_output=True)
    print(f"  saved {_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
