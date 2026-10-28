"""``csc maps edit``：在雷达图上点位置，自动换算成 Hammer 坐标。

为什么是浏览器应用而不是 matplotlib 窗口：

* 不用装 Tk / 不抢焦点，滚轮缩放、拖动平移的手感比 ginput 好；
* 页面能直接 POST 回本地小服务 → **点完自动写 config/points.yaml**，不用复制粘贴；
* 同一份 HTML 也能离线打开（``--no-serve``），导出 YAML 走剪贴板/下载。

坐标系：页面里所有坐标都是**世界坐标**（Hammer units）；像素↔世界的换算由
``csc.radar.MapTransform`` 在 Python 侧算好、注入页面，JS 只做同一套公式，避免两处口径漂移。
"""

from __future__ import annotations

import base64
import html as _html
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import map_points
from .config import MapSpec, Settings
from .maps_page import TEMPLATE as _TEMPLATE
from .radar import MapTransform


class EditError(RuntimeError):
    """打点器启动失败（没雷达图 / 没配置等）。"""


# --------------------------------------------------------------------------- payload


def _load_points_names(config_dir: Path, map_name: str) -> dict[str, set[str]]:
    """points.yaml 里已有的点名，用来区分"工具写的"和"maps.yaml 手写的"。"""
    try:
        doc = map_points.load_points(config_dir)
    except map_points.PointsError:
        return {"entry": set(), "cover": set()}
    block = (doc.get("maps") or {}).get(map_name) or {}
    return {
        "entry": {str(row.get("name")) for items in (block.get("entry_points") or {}).values() for row in (items or ())},
        "cover": {str(row.get("name")) for items in (block.get("cover_points") or {}).values() for row in (items or ())},
    }


def build_payload(
    map_name: str,
    settings: Settings,
    transform: MapTransform,
    *,
    serve: bool = True,
    default_z: float = 0.0,
) -> dict[str, Any]:
    """把「变换 + 雷达图 + 现有坐标 + 门禁」打包给页面。"""
    spec: MapSpec = settings.map_spec(map_name)
    from_points = _load_points_names(settings.config_dir, map_name)

    radar_data: str | None = None
    if transform.ready and transform.radar is not None:
        raw = Path(transform.radar).read_bytes()
        radar_data = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

    regions: dict[str, dict[str, Any]] = {}
    for region, raw in spec.regions.items():
        regions[region] = {
            "center": list(raw["center"]) if raw.get("center") else None,
            "radius": float(raw.get("radius") or 1100.0),
        }
    for region in list(spec.entry_points) + list(spec.cover_points):
        regions.setdefault(region, {"center": None, "radius": 1100.0})

    points: list[dict[str, Any]] = []
    for region, items in spec.entry_points.items():
        for ep in items:
            if ep.pos is None:
                continue
            points.append(
                {
                    "kind": "entry",
                    "region": region,
                    "name": ep.name,
                    "pos": [round(ep.pos[0], 1), round(ep.pos[1], 1), round(ep.pos[2], 1)],
                    "source": "points" if ep.name in from_points["entry"] else "yaml",
                }
            )
    for region, items in spec.cover_points.items():
        for cp in items:
            points.append(
                {
                    "kind": "cover",
                    "region": region,
                    "name": cp.name,
                    "pos": [round(cp.pos[0], 1), round(cp.pos[1], 1), round(cp.pos[2], 1)],
                    "source": "points" if cp.name in from_points["cover"] else "yaml",
                }
            )

    default_region = spec.default_region or (sorted(regions)[0] if regions else "A")
    return {
        "map": map_name,
        "serve": bool(serve),
        "transform": {
            "pos_x": transform.pos_x,
            "pos_y": transform.pos_y,
            "scale": transform.scale,
            "rotation_deg": transform.rotation_deg,
            "width": transform.width,
            "height": transform.height,
        },
        "radar": radar_data,
        "regions": regions,
        "points": points,
        "gate": {
            "min_entry": spec.gate_min_entry_points,
            "min_cover": spec.gate_min_cover_points,
        },
        "defaults": {"z": float(default_z), "radius": 1100.0, "region": default_region, "kind": "entry"},
        "points_yaml": str(map_points.points_path(settings.config_dir)),
        "version": transform.version,
        "lower": transform.lower,
        "world_bounds": list(transform.world_bounds),
    }


# --------------------------------------------------------------------------- HTML


def render_html(payload: Mapping[str, Any]) -> str:
    """生成单文件页面（雷达图内联为 base64）。"""
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = _html.escape(str(payload.get("map") or ""), quote=False)
    return _TEMPLATE.replace("__PAYLOAD__", blob).replace("__MAP__", title)


def write_html(payload: Mapping[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(payload), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- 本地保存服务


def serve(
    payload: Mapping[str, Any],
    config_dir: Path,
    *,
    port: int = 0,
    open_browser: bool = True,
    log: Any = print,
) -> int:
    """起一个只监听 127.0.0.1 的小服务：GET 页面，POST /api/save 写 points.yaml。"""
    html = render_html(payload)
    state: dict[str, Any] = {"httpd": None}
    map_name = str(payload.get("map") or "")
    config_dir = Path(config_dir)

    class Handler(BaseHTTPRequestHandler):
        server_version = "csc-maps-edit"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 静音
            return

        def _send(self, code: int, body: bytes | str, ctype: str = "application/json; charset=utf-8") -> None:
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except Exception:  # noqa: BLE001 - 客户端提前断开
                pass

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
            if self.path in ("/", "/index.html"):
                self._send(200, html, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(payload, ensure_ascii=False))
            elif self.path == "/favicon.ico":
                self._send(204, b"")
            else:
                self._send(404, json.dumps({"ok": False, "error": "not found"}))

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/save":
                self._send(404, json.dumps({"ok": False, "error": "not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                path, block = map_points.apply_payload(body, config_dir, map_name=map_name)
                problems = map_points.validate_payload(body, map_name=map_name)
            except Exception as exc:  # noqa: BLE001 - 全部回给页面显示
                self._send(400, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
                return
            counts = {
                "entry": {r: len(v) for r, v in (block.get("entry_points") or {}).items()},
                "cover": {r: len(v) for r, v in (block.get("cover_points") or {}).items()},
            }
            self._send(
                200,
                json.dumps({"ok": True, "path": str(path), "problems": problems, "counts": counts}, ensure_ascii=False),
            )
            if body.get("exit"):
                httpd = state.get("httpd")
                if httpd is not None:
                    threading.Thread(target=httpd.shutdown, daemon=True).start()

    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    state["httpd"] = httpd
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    log(f"打点器已启动: {url}")
    log(f"地图: {map_name}   雷达: {'有' if payload.get('radar') else '无(纯网格)'}   保存到: {payload.get('points_yaml')}")
    log("在页面里点位置，Ctrl+S 保存；点「保存并关闭」会同时结束本服务。")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - 没浏览器就手动开
            log("（没能自动打开浏览器，请手动访问上面的地址）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("\n收到 Ctrl+C，退出。")
    finally:
        httpd.server_close()
    return 0


# --------------------------------------------------------------------------- 导入已有坐标

def load_map_file(path: Path, map_name: str) -> dict[str, Any]:
    """读一个导出的 YAML/JSON，转成编辑器/apply 用的 payload。

    支持三种形态：
    1. 完整 points 文档（含 ``maps:``）；
    2. 单图 block（``regions`` / ``entry_points`` / ``cover_points``）；
    3. 编辑器导出的 payload（含 ``map`` 字段）。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = yaml.safe_load(text) or {}
    if not isinstance(doc, Mapping):
        raise EditError(f"{path} 不是有效的地图坐标文件")

    if "maps" in doc and isinstance(doc["maps"], Mapping):
        block = (doc["maps"] or {}).get(map_name) or {}
        return {"map": map_name, **dict(block)}
    if doc.get("map") and ({"entry_points", "cover_points", "regions"} & set(doc)):
        return dict(doc)
    if {"entry_points", "cover_points", "regions"} & set(doc):
        return {"map": map_name, **dict(doc)}
    raise EditError(f"{path} 里没找到可用坐标（期望 maps.<图名> 或 entry_points/cover_points）")
