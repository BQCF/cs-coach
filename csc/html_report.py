"""单文件 HTML 报告（N6 空间化版）：按位置组织，雷达图 + 悬停证据 + 每个 zone 一个故事。

用户核心诉求（原话）：「我的问题是出现在那个位置，而不是只是告诉我预瞄差了多少」。

产出：output/<demo>/report.html —— 图 base64 内嵌、雷达 SVG 悬停、无外部依赖、可分享。
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import polars as pl

FAULT_COLOR = {
    "position": "#e74c3c", "aim": "#3498db", "preaim": "#e67e22",
    "info": "#9b59b6", "none": "#95a5a6",
}



# 大白话映射（用户要求：读者是普通 CS 玩家/比赛粉丝，不用专业英文术语）
FAULT_CN = {
    "position": "站位", "position_config": "地图设置", "aim": "枪法",
    "preaim": "预瞄", "info": "信息", "none": "被阴",
}
EVID_CN = {
    "exposure_mean": "暴露给几个敌人", "crossfire_mean": "被多角度打",
    "escape_mean": "撤回掩体要几秒", "pro_zone_pct": "职业常站度",
    "gap_mean": "谁先看见谁差毫秒", "aim_off_mean": "准星偏了几度",
    "exposure": "暴露给几个敌人", "crossfire": "被多角度打",
    "escape_time": "撤回掩体要几秒", "pro_cumulative_percentile": "职业常站度",
}


def _cn(fault: str) -> str:
    return FAULT_CN.get(fault, fault)


def _ev_cn(key: str) -> str:
    return EVID_CN.get(key, key)




def shot_or_fallback(out_dir: Path, tick: int, kind: str = 'death'):
    # 有真图就用真图（base64），没有返回 None
    p = Path(out_dir) / 'shots' / (str(tick) + '_' + kind + '.jpg')
    if p.exists():
        return _img_b64(p)
    return None


def _img_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _radar_svg(df: pl.DataFrame, map_name: str, width: int = 640) -> str | None:
    """T1：把死亡点画在雷达图上，SVG 圆点可悬停查看证据。"""
    if df.is_empty() or not {"my_x", "my_y", "fault"} <= set(df.columns):
        return None
    try:
        from . import radar as radar_mod  # 局部导入
        from .config import bootstrap_env, PROJECT_ROOT

        bootstrap_env()
        transform = radar_mod.load_transform(map_name, PROJECT_ROOT / ".awpy")
        if not transform.ready:
            return None
        img_b64 = _img_b64(Path(transform.radar))
        w, h = transform.width, transform.height
    except Exception:  # noqa: BLE001
        return None

    # 缩放到显示宽度
    scale = width / w
    disp_w, disp_h = width, int(h * scale)

    circles = []
    for r in df.iter_rows(named=True):
        px, py = transform.world_to_radar(r.get("my_x", 0), r.get("my_y", 0))
        px, py = transform.clamp_pixel(px, py)
        sx, sy = px * scale, py * scale
        fault = r.get("fault", "none")
        color = FAULT_COLOR.get(fault, "#7f8c8d")
        # 大小 = 严重度（exposure+crossfire+escape 超阈值程度）
        sev = 1.0
        sev += (r.get("exposure") or 0) * 0.4
        sev += (r.get("crossfire") or 0) * 0.3
        sev += min(2.0, (r.get("escape_time") or 0) * 0.3)
        rad = 5 + sev * 2.5

        tip = (f"第 {r.get('death_tick')} tick\n"
               f"{r.get('mechanism')} / {_cn(fault)}\n"
               f"暴露给 {r.get('exposure')} 个敌人｜被多角度打 {r.get('crossfire')} 次｜撤回掩体 {r.get('escape_time')} 秒\n"
               f"准星偏了 {r.get('my_aim_off_at_see_deg')}°｜谁先看见谁差 {r.get('first_seen_gap_ms')} 毫秒")
        tip_esc = html.escape(tip, quote=True)
        circles.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{rad:.1f}" fill="{color}" '
            f'fill-opacity="0.72" stroke="#fff" stroke-width="1.2" style="cursor:pointer">'
            f'<title>{tip_esc}</title></circle>'
        )

    legend = "　".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{c};display:inline-block"></span>{_cn(k)}</span>'
        for k, c in FAULT_COLOR.items()
    )
    return f'''<div style="text-align:center">
<svg width="{disp_w}" height="{disp_h}" viewBox="0 0 {disp_w} {disp_h}" xmlns="http://www.w3.org/2000/svg">
  <image href="{img_b64}" x="0" y="0" width="{disp_w}" height="{disp_h}"/>
  {''.join(circles)}
</svg>
<div style="margin-top:8px;font-size:12px;color:#666">{legend}<br>圆点大小=严重度，悬停查看证据</div>
</div>'''


def build(out_dir: Path, *, player: str = "玩家", map_name: str = "de_ancient",
          findings: list | None = None, fp_shots: list | None = None) -> Path:
    out_dir = Path(out_dir)
    deaths_csv = out_dir / "deaths.csv"
    report_md = out_dir / "report.md"
    profile_dir = out_dir / "profile"

    df = pl.read_csv(deaths_csv) if deaths_csv.exists() else pl.DataFrame()
    n = df.height

    # T1 雷达打点
    radar_svg = _radar_svg(df, map_name)

    # 第一人称截图
    fp_html = ""
    if fp_shots:
        cards = ""
        for fp in fp_shots:
            fname = fp.get("file", "")
            caption = fp.get("caption", "")
            real_b64 = None
            if '_' in Path(fname).stem:
                try:
                    _tick = int(Path(fname).stem.split('_')[-1])
                    real_b64 = shot_or_fallback(out_dir, _tick, 'death')
                except Exception:
                    real_b64 = None
            if real_b64:
                tag = '<span style="background:#27ae60;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px">游戏内真截图</span>'
                cards += f'<div class="card"><h3>{html.escape(caption)} {tag}</h3><img src="{real_b64}" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px"></div>'
            elif (out_dir / fname).exists():
                b64 = _img_b64(out_dir / fname)
                tag = '<span style="background:#95a5a6;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px">几何示意</span>'
                cards += f'<div class="card"><h3>{html.escape(caption)} {tag}</h3><img src="{b64}" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px"></div>'
        if cards:
            hdr = "\U0001F3AF 你看这一枪是怎么输的（第一人称视角）"
            note = ('<div style="max-width:1100px;margin:0 auto 10px;padding:0 16px;font-size:12px;color:#888">'
                    '⚠️ 标「游戏内截图」的是<b>真截图 + 标注</b>；其余是地图模型渲染的示意图——.dem 只记录位置/朝向/事件数据，不含游戏画面。'
                    '想换成真截图：按 <code>replay_guide.md</code> 操作（CS2 控制台 <code>demo_gototick &lt;tick&gt;</code> 截图放进 <code>shots/&lt;tick&gt;.png</code>），重跑报告自动替换。</div>')
            fp_html = (
                '<h2 style="max-width:1100px;margin:24px auto 12px;padding:0 16px;font-size:20px">'
                + hdr + '</h2>\n  ' + note
                + '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;max-width:1100px;margin:0 auto;padding:0 16px">'
                + cards + '</div>'
            )

    # 六维：直接与雷达图同源（compute_dims + DIMS）。
    # 不读 profile.csv——它的列是「维度/你的值/样本」，跟旧版 name/value/n/formula
    # 对不上，整张表会渲染成 "-"（2026-10-25 修）。
    prof_html = ""
    try:
        from . import profile as profile_mod  # 局部导入避免循环

        dims = profile_mod.compute_dims(df) if n else {}
        for _key, dim_name, _good, formula in profile_mod.DIMS:
            d = dims.get(_key) or {}
            v = d.get("value")
            vdisp = f"{v:.2f}" if isinstance(v, float) else (str(v) if v is not None else "-")
            prof_html += (f"<tr><td>{html.escape(dim_name)}</td><td><b>{vdisp}</b></td>"
                          f"<td>{d.get('n', 0)}</td><td>{html.escape(str(formula))}</td></tr>")
    except Exception:  # noqa: BLE001 - 画像算不出就留空表，不炸报告
        prof_html = ""
    if not prof_html:
        prof_html = '<tr><td colspan="4" style="color:#999">（deaths.csv 数据不足，画像无法计算）</td></tr>'

    def img(name: str) -> str | None:
        for base in (out_dir, profile_dir):
            p = base / name
            if p.exists():
                return _img_b64(p)
        return None

    # T2：按 zone 讲故事（findings）
    zone_cards = ""
    for f in (findings or []):
        z = f.get("zone", "?")
        nfault = f.get("n_deaths", 0)
        dom = f.get("dominant_fault", "?")
        color = FAULT_COLOR.get(dom, "#7f8c8d")
        ev = f.get("evidence", {})
        ev_html = "　".join(f"<b>{_ev_cn(str(k))}</b>：{round(v,2) if isinstance(v,float) else v}" for k, v in ev.items() if v is not None)
        pro = f.get("pro_compare", {})
        pro_html = ""
        if pro.get("zone_pct") is not None:
            pro_html = f"职业常站度 <b>{pro['zone_pct']}</b>（1=常站，0=冷门）"
            if pro.get("cold_rank"):
                pro_html += f"（冷门程度排第 {pro['cold_rank']}）"

        typical_html = ""
        for t in f.get("typical_deaths", []):
            typical_html += (f"<div style='margin:3px 0;font-size:12px;color:#555'>"
                            f"tick {t.get('tick')}｜{html.escape(str(t.get('mechanism')))}｜"
                            f"{html.escape(str(t.get('evidence'))[:60])}</div>")

        advice = f.get("advice") or {}
        if advice.get("action"):
            verify = advice.get("verify") or {}
            verify_html = ""
            for k, pair in verify.items():
                if pair and pair[0] is not None and pair[1] is not None:
                    verify_html += f"{k} {pair[0]}→{pair[1]}　"
            advice_html = (f"<div style='background:#e8f5e9;border-left:3px solid #27ae60;"
                          f"padding:8px 12px;border-radius:4px;margin-top:8px;font-size:13px'>"
                          f"💊 <b>处方</b>：{html.escape(advice['action'])}"
                          f"<br>📐 验算：{html.escape(verify_html)}"
                          f"<br>{html.escape(advice.get('note',''))}</div>")
        elif advice.get("note"):
            advice_html = (f"<div style='background:#fff8e1;border-left:3px solid #f39c12;"
                          f"padding:8px 12px;border-radius:4px;margin-top:8px;font-size:13px'>"
                          f"📋 {html.escape(advice['note'])}</div>")
        else:
            advice_html = ""

        zone_cards += f'''
  <div class="card">
    <h2><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{color};margin-right:8px"></span>
    {html.escape(z)} <span style="font-size:14px;color:#888;font-weight:normal">（{nfault} 次死亡，主要是 <b style="color:{color}">{_cn(dom)}</b> 问题）</span></h2>
    <div style="font-size:13px;color:#555;margin:8px 0">📊 量化：{ev_html}</div>
    <div style="font-size:13px;color:#555;margin:8px 0">🏆 {pro_html}</div>
    <div style="margin:10px 0"><b style="font-size:13px">典型死亡：</b>{typical_html}</div>
    {advice_html}
  </div>'''

    # 其它章节（图 + 明细 + 原文）
    other_charts = {
        "预瞄偏差分布（你 vs 对手）": img("preaim_distribution.png"),
        "枪法问题：对枪伤害对比": img("duel_damage.png"),
        "六维能力雷达": img("radar.png"),
    }
    chart_html = ""
    for title, b64 in other_charts.items():
        if b64:
            chart_html += (f'<div class="card"><h3>{html.escape(title)}</h3>'
                          f'<img src="{b64}" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px"></div>')

    detail_rows = ""
    if n:
        cols = ["death_tick", "zone", "mechanism", "fault", "fault_sub", "first_fire",
                "my_shots", "my_hits", "my_dmg", "his_dmg", "my_aim_off_at_see_deg", "first_seen_gap_ms"]
        cols = [c for c in cols if c in df.columns]
        for r in df.select(cols).iter_rows(named=True):
            def fmt(v):
                if v is None:
                    return "-"
                return f"{v:.1f}" if isinstance(v, float) else str(v)
            color = FAULT_COLOR.get(r.get("fault", ""), "#7f8c8d")
            detail_rows += (f"<tr><td>{fmt(r.get('death_tick'))}</td><td>{fmt(r.get('zone'))}</td>"
                            f"<td><span style='color:{color};font-weight:bold'>{fmt(r.get('mechanism'))}</span></td>"
                            f"<td><span style='color:{color};font-weight:bold'>{_cn(fmt(r.get('fault')))}</span></td>"
                            f"<td>{fmt(r.get('fault_sub'))}</td><td>{fmt(r.get('first_fire'))}</td>"
                            f"<td>{fmt(r.get('my_shots'))}</td><td>{fmt(r.get('my_hits'))}</td>"
                            f"<td>{fmt(r.get('my_dmg'))}</td><td>{fmt(r.get('his_dmg'))}</td>"
                            f"<td>{fmt(r.get('my_aim_off_at_see_deg'))}</td><td>{fmt(r.get('first_seen_gap_ms'))}</td></tr>")

    md_text = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
    md_text = "\n".join(l for l in md_text.splitlines() if not l.strip().startswith("!["))

    html_doc = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(player)} · 空间化体检报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #f5f6fa; color: #2c3e50; }}
  .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: #fff; padding: 30px 24px; }}
  .header h1 {{ margin: 0 0 8px; font-size: 25px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 22px 16px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .card h2 {{ margin-top: 0; font-size: 18px; }}
  .card h3 {{ font-size: 15px; color: #555; margin: 14px 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 7px 9px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f0f2f5; font-weight: 600; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
</style>
</head>
<body>
<div class="header">
  <h1>📍 {html.escape(player)} · 空间化体检报告</h1>
  <div style="opacity:0.9;font-size:14px">地图：{html.escape(map_name)}　|　死亡点：{n}　|　按位置组织，每个位置讲一个完整故事</div>
</div>
<div class="container">

  <div class="card">
    <h2>🗺️ 你在哪死的（雷达图，悬停查看证据）</h2>
    {radar_svg or '<div style="color:#999">（雷达图生成需要 awpy 资源）</div>'}
  </div>

  {fp_html}

  <h2 style="max-width:1100px;margin:24px auto 12px;padding:0 16px;font-size:20px">📍 位置体检（按问题严重度排序）</h2>
  {zone_cards or '<div class="card">（暂无 zone findings）</div>'}

  <div class="card">
    <h2>📊 六维能力画像</h2>
    <table><thead><tr><th>维</th><th>值</th><th>n</th><th>公式</th></tr></thead><tbody>{prof_html}</tbody></table>
  </div>

  <div class="charts">{chart_html}</div>

  <div class="card">
    <h2>📋 死亡明细（证据链）</h2>
    <table>
      <thead><tr><th>时间点</th><th>位置</th><th>怎么死的</th><th>问题出在</th><th>细节</th><th>谁先开枪</th>
      <th>开几枪</th><th>中几枪</th><th>我打多少</th><th>他打多少</th><th>准星偏几度</th><th>谁先看见谁(毫秒)</th></tr></thead>
      <tbody>{detail_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>📄 完整报告（原文）</h2>
    <pre style="white-space:pre-wrap;font-family:inherit;font-size:12px;background:#f8f9fb;padding:14px;border-radius:8px;max-height:420px;overflow:auto">{html.escape(md_text)}</pre>
  </div>

</div>
</body>
</html>"""

    out = out_dir / "report.html"
    out.write_text(html_doc, encoding="utf-8")
    return out
