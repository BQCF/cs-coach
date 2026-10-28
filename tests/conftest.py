"""测试夹具。

**为什么不用 pytest 的 ``tmp_path``**：DSH 的 Windows 沙箱用受限令牌跑命令，而
Python 的 ``os.mkdir(path, 0o700)``（pytest 的 ``tmp_path`` / ``.pytest_cache``
都会走这条）会写出只授权 owner SID 的 DACL；受限令牌里用户 SID 是 deny-only，
于是那种目录连本进程自己都进不去（``PermissionError: [WinError 5]``）。

所以测试统一用 ``ws_tmp``：在工作区 ``.tmp/tests/`` 下用**默认权限**建目录。
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Iterator

import pytest

BASE = Path(__file__).resolve().parents[1] / ".tmp" / "tests"


@pytest.fixture()
def ws_tmp() -> Iterator[Path]:
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / uuid.uuid4().hex[:10]
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except OSError:
            pass  # 沙箱里偶发句柄未释放，不因清理失败挂测试
