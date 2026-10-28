"""netcon 客户端（N7-A）：用 CS2 的网络控制台远程驱动游戏截真图。

CS2 必须**带 netcon 端口启动**（不能附加到已在跑的实例）：
    cs2.exe -console -novid -netconport 2121 [-netconpassword <密码>]

安全提醒：netcon 只监听本地，别把端口暴露到公网。
"""

from __future__ import annotations

import socket
import time


class CS2Netcon:
    """CS2 netcon TCP 客户端。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 2121,
                 password: str | None = None, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        if self.password:
            self.send(self.password)
            time.sleep(0.1)

    def send(self, cmd: str) -> None:
        """发一行命令（自动加 \\n）。"""
        if self.sock is None:
            raise RuntimeError("未连接，先调 connect()")
        self.sock.sendall((cmd + "\n").encode("utf-8", errors="replace"))

    def send_wait(self, cmd: str, wait_s: float = 0.3) -> str:
        """发命令 + 短暂等待，返回读到的回显（尽力而为，netcon 不保证回显）。"""
        self.send(cmd)
        time.sleep(wait_s)
        try:
            self.sock.settimeout(0.05)
            return self.sock.recv(4096).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
        finally:
            try:
                self.sock.settimeout(self.timeout)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def ping(host: str = "127.0.0.1", port: int = 2121) -> bool:
    """检查 netcon 端口是否在监听。"""
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except Exception:  # noqa: BLE001
        return False
