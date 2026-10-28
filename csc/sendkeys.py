"""Windows 键盘模拟（方案 A）：用 SendInput 直接驱动 CS2 开发者控制台。

不依赖第三方库（纯 ctypes + Win32 SendInput）。
CS2 用 -console 启动后，按 `~` 开控制台 → 输入命令 → 回车。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import time

# Win32 常量
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
VK_BACK = 0x08
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_OEM_3 = 0xC0  # `~` 键（键盘左上角）
VK_ESCAPE = 0x1B

ULONG_PTR = ctypes.c_ulonglong


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.wintypes.WORD),
                ("wScan", ctypes.wintypes.WORD),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("time", ctypes.wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]
    _anonymous_ = ("_INPUT",)
    _fields_ = [("type", ctypes.wintypes.DWORD), ("_INPUT", _INPUT)]


user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.SendInput.argtypes = (ctypes.c_uint32, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = ctypes.c_uint32


def _send_key(vk: int, *, up: bool = False, unicode_char: str | None = None) -> None:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    if unicode_char is not None:
        inp.ki.wVk = 0
        inp.ki.wScan = ord(unicode_char)
        inp.ki.dwFlags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    else:
        inp.ki.wVk = vk
        inp.ki.wScan = 0
        inp.ki.dwFlags = KEYEVENTF_KEYUP if up else 0
    inp.ki.time = 0
    inp.ki.dwExtraInfo = 0
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def tap_key(vk: int, delay: float = 0.03) -> None:
    _send_key(vk)
    time.sleep(delay)
    _send_key(vk, up=True)
    time.sleep(delay)


def type_text(text: str, delay: float = 0.02) -> None:
    """逐字符输入（Unicode 方式，支持中文/符号）。"""
    for ch in text:
        _send_key(0, unicode_char=ch)
        time.sleep(delay)
        _send_key(0, up=True, unicode_char=ch)
        time.sleep(delay)


def open_console() -> None:
    """按 `~` 开/关开发者控制台。"""
    tap_key(VK_OEM_3, delay=0.05)


def send_command(cmd: str, *, submit: bool = True, settle: float = 0.15) -> None:
    """发一条控制台命令（假定控制台已开）。"""
    # 清行
    tap_key(VK_BACK, delay=0.01)
    type_text(cmd)
    time.sleep(settle)
    if submit:
        tap_key(VK_RETURN, delay=0.05)


def focus_window(pid: int | None = None, title_contains: str | None = None) -> bool:
    """把 CS2 窗口带到前台。"""
    try:
        import subprocess

        if pid is not None:
            subprocess.run(["powershell", "-Command",
                            f"(New-Object -ComObject WScript.Shell).AppActivate({pid})"],
                           capture_output=True)
            time.sleep(0.3)
            return True
        if title_contains:
            subprocess.run(["powershell", "-Command",
                            f"(New-Object -ComObject WScript.Shell).AppActivate('{title_contains}')"],
                           capture_output=True)
            time.sleep(0.3)
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def find_cs2_pid() -> int | None:
    r = __import__("subprocess").run(["tasklist", "/FI", "IMAGENAME eq cs2.exe"],
                                     capture_output=True, text=True,
                                     encoding="utf-8", errors="replace")
    for line in (r.stdout or "").splitlines():
        if "cs2.exe" in line:
            parts = line.split()
            try:
                return int(parts[1])
            except (IndexError, ValueError):
                continue
    return None
