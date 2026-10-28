"""CLI：``python -m csc <子命令>``。

子命令：

* ``env``       —— 环境自检（版本 / 资源 / demo 存量）
* ``parse``     —— 解析 demo -> parquet 缓存 + 列契约核对（M1 第一步）
* ``analyze``   —— 归因 + 指标 -> output/deaths.csv + report.md（M1 交付物）
* ``report``    —— 对已有 deaths.csv 出报告 + 图
* ``selftest``  —— 合成数据跑通整条管线（无 demo / 无 awpy 资源也能跑）
* ``maps``      —— 地图配置：list / check / edit

每个子命令都有 ``--dry-run``：不碰 awpy，只做配置 / 契约校验。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from . import benchmark_pro, deaths, demo_io, duel, fault, findings, firstperson, geometry, html_report, metrics, plot, profile, report, report_charts, schema, synthetic, windows, zones
from .config import PROJECT_ROOT, bootstrap_env, load_settings


# --------------------------------------------------------------------------- 通用


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="csc", description="CS2 demo 离线体检")
    p.add_argument("--config-dir", default=None, help="config 目录（默认 cs-coach/config）")
    p.add_argument("--output", default=None, help="输出目录（默认 config/default.yaml 里的 output）")
    p.add_argument("--verbose", action="store_true", help="打印更多过程信息")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("env", help="环境自检")

    sp = sub.add_parser("parse", help="解析 demo + 列契约核对")
    sp.add_argument("--demo", required=True)
    sp.add_argument("--player", default=None)
    sp.add_argument("--force", action="store_true", help="忽略缓存重新解析")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("analyze", help="归因 + 指标 -> deaths.csv")
    sp.add_argument("--demo", required=True)
    sp.add_argument("--player", default=None)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("report", help="对 deaths.csv 出报告")
    sp.add_argument("--deaths", required=True)
    sp.add_argument("--player", default=None)
    sp.add_argument("--map", default=None, help="地图名（雷达打点用）")

    sp = sub.add_parser("eval", help="M3：自动归因 vs 人工标注，算准确率")
    sp.add_argument("--labels", required=True, help="人工标注 labels.csv (demo,death_tick,label)")
    sp.add_argument("--deaths", required=True, help="analyze 产出的 deaths.csv")
    sp.add_argument("--out", default=None, help="评估报告输出路径（默认 output/eval_report.md）")

    sp = sub.add_parser("profile", help="N4：跨场画像 + 六维雷达（我的定位）")
    sp.add_argument("--deaths", action="append", required=True,
                    help="deaths.csv（可多次，跨场聚合）")
    sp.add_argument("--player", default=None)
    sp.add_argument("--out", default=None, help="输出目录（默认 output/profile）")

    sp = sub.add_parser("findings", help="N5：findings.json + verify_pose 处方")
    sp.add_argument("--deaths", required=True)
    sp.add_argument("--map", default=None, help="地图名（加载 geo + zone）")
    sp.add_argument("--out", default=None, help="输出目录（默认 output）")

    sub.add_parser("selftest", help="合成数据跑通管线")

    sp = sub.add_parser("zones", help="N2：位置体系（suggest/check/list）")
    sp.add_argument("action", choices=["suggest", "check", "list"])
    sp.add_argument("--map", default=None)
    sp.add_argument("--demo", default=None, help="用哪个职业 demo 生成候选（suggest）")
    sp.add_argument("--n", type=int, default=32, help="候选 zone 数")
    sp.add_argument("--eps", type=float, default=220.0, help="DBSCAN 半径")
    sp.add_argument("--write", action="store_true", help="suggest 后写入 zones.yaml")

    sp = sub.add_parser("maps", help="地图配置 / 雷达打点")
    sp.add_argument("action", choices=["list", "check", "show", "fetch", "edit", "apply"])
    sp.add_argument("--map", default=None)
    sp.add_argument("--version", default=None, help="awpy-data 版本（默认用已缓存的最新）")
    sp.add_argument("--lower", action="store_true", help="使用 lower 雷达图（nuke/train/vertigo）")
    sp.add_argument("--no-fetch", action="store_true", help="不自动下载缺失的地图资源")
    sp.add_argument("--rotate-deg", type=float, default=None, help="强制像素↔世界旋转角（默认按 map_data 推断）")
    sp.add_argument("--z", type=float, default=0.0, help="新增点的默认 z（玩家脚下高度）")
    sp.add_argument("--port", type=int, default=0, help="保存服务端口（0=随机）")
    sp.add_argument("--no-serve", action="store_true", help="只生成静态 HTML，不起本地保存服务")
    sp.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    sp.add_argument("--html", default=None, help="静态页输出路径（默认 output/maps/edit-<map>.html）")
    sp.add_argument("--points", default=None, help="apply：要导入的 YAML / JSON 文件")

    sp = sub.add_parser("train", help="训练职业基准网格（解析 pro demo，缓存落位，供报告对比）")
    sp.add_argument("--map", default=None, help="只训练指定图（默认扫 pro 目录里所有出现过的图）")
    sp.add_argument("--phases", default="early,mid", help="采集的回合阶段，逗号分隔（early/mid/late）")
    sp.add_argument("--force", action="store_true", help="忽略已有缓存强制重训")

    return p


def _out_dir(args: argparse.Namespace, settings: Any) -> Path:
    return Path(args.output or settings.paths.output)


# --------------------------------------------------------------------------- 子命令


def _cmd_env(args: argparse.Namespace, settings: Any) -> int:
    import platform

    bootstrap_env()
    print("== 运行环境 ==")
    print(f"python        : {platform.python_version()} ({sys.executable})")
    for mod in ("polars", "numpy", "scipy", "matplotlib", "yaml", "pytest"):
        try:
            m = __import__(mod)
            print(f"{mod:<13}: {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{mod:<13}: 未安装 ({exc.__class__.__name__})")
    try:
        import awpy  # noqa: PLC0415

        print(f"{'awpy':<13}: {getattr(awpy, '__version__', '?')}")
        awpy_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"{'awpy':<13}: 未安装 ({exc})")
        awpy_ok = False

    print("\n== 资源 ==")
    print(f"AWPY_DATA_DIR : {settings.paths.awpy_data}")
    print(f"MPLCONFIGDIR  : {settings.paths.tmp / 'mpl'}")
    if awpy_ok:
        try:
            from awpy import data  # noqa: PLC0415

            maps = data.available_maps()
            print(f"awpy-data maps: {len(maps)} 张 ({', '.join(maps[:8])}...)" if len(maps) > 8 else f"awpy-data maps: {', '.join(maps)}")
            try:
                buildid = data.manifest().get("buildid")
                print(f"awpy-data buildid: {buildid}")
            except Exception:  # noqa: BLE001
                print("awpy-data buildid: 未知（可能还没 awpy get）")
        except Exception as exc:  # noqa: BLE001
            print(f"awpy-data 状态: 读取失败 ({exc})；先跑 `awpy get`")

    print("\n== demo 存量 ==")
    me = demo_io.demos_in(settings.paths.demos / "me")
    pro = demo_io.demos_in(settings.paths.demos / "pro")
    print(f"data/demos/me : {len(me)} 个")
    for d in me:
        print(f"  - {d.name}")
    print(f"data/demos/pro: {len(pro)} 个")
    for d in pro:
        print(f"  - {d.name}")
    return 0


def _cmd_parse(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    demo_path = Path(args.demo)
    if args.dry_run:
        print(f"[dry-run] demo={demo_path} 存在={demo_path.exists()}")
        if not demo_path.exists():
            print("[dry-run] 提示：先放 demo 到 data/demos/me/，或 --demo 指向 Steam 录像目录")
        return 0

    ctx, issues = demo_io.parse_demo(demo_path, settings.paths.cache, force=args.force)
    print(f"解析完成: {ctx.map_name}  tickrate={ctx.tick_rate}  key={ctx.key}")
    for name, df in ctx.tables.items():
        print(f"  {name:<16} rows={df.height}")
    print(f"  death_frames    rows={ctx.death_frames.height}")
    if ctx.meta.get("warnings"):
        print("warnings:")
        for w in ctx.meta["warnings"]:
            print(f"  - {w}")

    md = schema.render_markdown({**ctx.tables, "snap": ctx.death_frames}, issues)
    out = _out_dir(args, settings)
    out.mkdir(parents=True, exist_ok=True)
    (out / "columns.md").write_text(md, encoding="utf-8")
    print(f"\n列契约核对 -> {out / 'columns.md'}（问题 {len(issues)} 条）")
    for i in issues:
        print(f"  - {i}")
    return 0


def run_analysis(
    settings: Any,
    demo_path: Path,
    player_query: str | None,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """归因 + 指标 完整链路（analyze 用，也供 selftest 复用思路）。"""
    ctx, issues = demo_io.parse_demo(demo_path, settings.paths.cache, force=force)
    steamid, name = demo_io.resolve_player(ctx, player_query or settings.me.steamid or settings.me.name)

    death_rows = deaths.extract_deaths(ctx.table("kills"), steamid)
    ticks = [int(t) for t in death_rows["tick"].to_list()] if "tick" in death_rows.columns else []
    spans, snap = windows.ensure_windows(ctx, ticks, settings.paths.cache, settings.windows, player=steamid, force=force)

    dwindows = deaths.build_death_windows(ctx, steamid, spans, snap, my_name=name)
    geo, geo_err = geometry.try_build_geo(ctx.map_name, settings, smokes=ctx.table("smokes"))

    zlist = zones.load_zones(settings.config_dir, ctx.map_name)
    records: list[deaths.DeathRecord] = []
    for w in dwindows:
        verdict = deaths.attribute(w, settings.rules, geo)
        rec = deaths.DeathRecord(w, verdict)
        rec.metrics = metrics.compute(w, geo, settings.rules)
        rec.duel = duel.build_duel(w, geo, settings.rules)
        rec.fault = fault.classify_fault(w, rec.metrics, rec.duel, settings.rules, settings.fault, verdict)
        if zlist:
            rec.zone, rec.zone_dist = zones.zone_of(zlist, w.my_pos)
        records.append(rec)

    frame = deaths.records_to_frame(records, demo=demo_path.name)

    # M4：给死亡点补职业分位（累积分位 / 邻域频率 / 同经济样本数）
    try:
        pro_rows = benchmark_pro.annotate_pro_percentile(records, settings, ctx.map_name)
        if pro_rows and frame.height == len(pro_rows):
            extra = pl.DataFrame(pro_rows)
            frame = pl.concat([frame, extra], how="horizontal")
    except Exception as exc:  # noqa: BLE001 - 缺职业 demo 时降级
        print(f"[M4 分位] 职业基准不可用（{exc}），分位列留空")

    out_dir.mkdir(parents=True, exist_ok=True)
    report.write_deaths_csv(frame, out_dir / "deaths.csv")
    summary = report.summarize(frame)
    caveats: list[str] = []
    if geo is None:
        caveats.append(f"未加载几何（{geo_err}）：exposure/crossfire/escape/gap 为 null")
    if not settings.map_spec(ctx.map_name).ready():
        caveats.append(f"{ctx.map_name} 的 maps.yaml 坐标未配齐，几何指标为 null；用 `csc maps edit --map {ctx.map_name}` 补")
    # N5：findings（位置体检 + verify 处方）
    try:
        zmap = {z.name: z for z in zones.load_zones(settings.config_dir, ctx.map_name)}
        findings_list = findings.build_findings(frame, geo, zmap)
    except Exception:  # noqa: BLE001
        findings_list = []
    # 可视化配图
    try:
        charts = report_charts.generate(frame, out_dir)
    except Exception:  # noqa: BLE001
        charts = {}
    try:
        _rd = plot.radar_dots(frame, ctx.map_name, out_dir / "radar_dots.png")
        radar_dots_img = _rd.name if _rd else None
    except Exception:  # noqa: BLE001
        radar_dots_img = None
    # M6 第一人称示意图（fp_*.png）：红X=敌人死时位置、绿十字=敌人冒头处（预瞄点）。
    # 先清掉上一次运行的旧图，避免 report 把过期 fp 再嵌进新报告。
    for old in out_dir.glob("fp_*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    if records:
        try:
            made = firstperson.generate_fp_shots(
                records, geo, settings.rules, out_dir, ctx.map_name, settings.paths.awpy_data,
                demo_name=demo_path.name,
            )
            if made:
                print(f"第一人称视角图: {len(made)} 张（真截图优先，缺图为模型示意）；真截图工作流见 replay_guide.md")
        except Exception as exc:  # noqa: BLE001 - 示意图是佐证，失败不影响主产物
            print(f"[fp 示意图跳过] {exc}")

    report.write_report_md(frame, summary, out_dir / "report.md", player=name, caveats=caveats,
                           findings_list=findings_list, charts=charts,
                           radar_dots_img=radar_dots_img, cfg=settings.report,
                           fp_shots=_fp_shots_for_report(out_dir))

    if records:
        plot.heatmap([(r.window.my_pos[0], r.window.my_pos[1]) for r in records], out_dir / "heatmap_me.png")
        for idx, rec in enumerate(records[:50]):  # 只画前 50 张，防止一个 demo 几百张
            plot.death_frame(rec, out_dir / f"death_{idx}.png")

    # 评审 4.1：验收看「非 null 率」，不是「有没有输出」
    coverage = {}
    for col in ("exposure", "crossfire", "first_seen_gap_ms", "predictable", "escape_time"):
        if col in frame.columns and frame.height:
            coverage[col] = f"{len(frame[col].drop_nulls())}/{frame.height}"

    return {
        "map": ctx.map_name,
        "tick_rate": ctx.tick_rate,
        "key": ctx.key,
        "player": name,
        "deaths": frame.height,
        "geo": geo is not None,
        "labels": frame["label"].to_list() if not frame.is_empty() else [],
        "issues": len(issues),
        "coverage": coverage,
        "out": str(out_dir),
    }


def _cmd_analyze(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    demo_path = Path(args.demo)
    if args.dry_run:
        print(f"[dry-run] demo={demo_path} 存在={demo_path.exists()}")
        return 0
    result = run_analysis(settings, demo_path, args.player, _out_dir(args, settings), force=args.force)
    print(f"analyze 完成: {result['map']}  tickrate={result['tick_rate']}  死亡点={result['deaths']}  几何={'有' if result['geo'] else '无(降级)'}")
    if result["coverage"]:
        print("指标非 null 率:", "  ".join(f"{k}={v}" for k, v in result["coverage"].items()))
    print(f"输出目录: {result['out']}")
    if result["labels"]:
        from collections import Counter

        print("标签分布:", dict(Counter(result["labels"])))
    return 0


def _cmd_report(args: argparse.Namespace, settings: Any) -> int:
    import polars as pl

    deaths_csv = Path(args.deaths)
    if not deaths_csv.exists():
        print(f"找不到 {deaths_csv}")
        return 1
    frame = pl.read_csv(deaths_csv)
    summary = report.summarize(frame)
    out = _out_dir(args, settings)
    out.mkdir(parents=True, exist_ok=True)
    try:
        map_name = (frame["zone"].drop_nulls().head(1).to_list() or [""])[0]
        geo, _ = geometry.try_build_geo(map_name or "de_ancient", settings)
        zmap = {z.name: z for z in zones.load_zones(settings.config_dir, map_name or "de_ancient")}
        findings_list = findings.build_findings(frame, geo, zmap)
    except Exception:  # noqa: BLE001
        findings_list = []
    try:
        charts = report_charts.generate(frame, out)
    except Exception:  # noqa: BLE001
        charts = {}
    # 雷达打点图（位置分布总览）
    radar_dots_img = None
    try:
        _map = args.map or "de_ancient"
        _rd = plot.radar_dots(frame, _map, out / "radar_dots.png")
        if _rd:
            radar_dots_img = _rd.name
    except Exception:  # noqa: BLE001
        radar_dots_img = None
    try:
        _dims = profile.compute_dims(frame)
        _labels = [d[1] for d in profile.DIMS]
        _vals = profile.radar_points(_dims)
        _rc = out / "profile" / "radar.png"
        _rc.parent.mkdir(parents=True, exist_ok=True)
        plot.radar_chart(_labels, _vals, _rc, title=f"{args.player or chr(29616)+chr(23478)} "+chr(20845)+chr(32500)+chr(33021)+chr(21147))
        radar_chart_img = f"profile/{_rc.name}"
    except Exception:
        radar_chart_img = None
    md = report.write_report_md(frame, summary, out / "report.md", player=args.player,
                                findings_list=findings_list, charts=charts,
                                radar_dots_img=radar_dots_img, radar_chart_img=radar_chart_img,
                           fp_shots=_fp_shots_for_report(out),
                                cfg=settings.report)
    # 落盘 findings.json
    try:
        import json as _json
        _fp = out / "findings.json"
        _fp.write_text(_json.dumps({"player": args.player, "n_deaths": frame.height,
                                    "findings": findings_list}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    # N6 空间化 HTML（按 zone 讲故事 + 雷达打点）
    try:
        map_name2 = args.map or (frame["zone"].drop_nulls().head(1).to_list() or [""])[0] or "de_ancient"
        html_path = html_report.build(out, player=args.player or "玩家", map_name=map_name2, findings=findings_list, fp_shots=_fp_shots_for_report(out))
        print(f"HTML 报告: {html_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[HTML 生成失败] {exc}")
    print(f"报告已生成: {md}")
    print(f"死亡点总数: {summary['total']}")
    for row in summary["by_label"]:
        print(f"  {row['label']}: {row['n']}")
    return 0


def _cmd_eval(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    from . import evaluate  # 局部导入

    labels = evaluate.load_labels(Path(args.labels))
    preds = evaluate.load_predictions(Path(args.deaths))
    result = evaluate.evaluate(labels, preds)

    md = evaluate.render_report(result)
    out = Path(args.out) if args.out else (_out_dir(args, settings) / "eval_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(f"标注样本: {result['n']}（对齐: {result['matched_on']}）")
    if not result["n"]:
        print("没有任何标注能对上自动结果，准确率无意义", file=sys.stderr)
        print("检查 --deaths 是不是标注的那场 demo（新 deaths.csv 带 demo 列会自动按场对齐）", file=sys.stderr)
        return 1
    print(f"准确率: {result['accuracy']*100:.1f}%")
    for r in result["per_label"]:
        print(f"  {r['label']}: support={r['support']} precision={r['precision']} recall={r['recall']}")
    if result["mismatches"]:
        print(f"不一致 {len(result['mismatches'])} 个（详见 {out}）")
    else:
        print("全部一致 ✅")
    print(f"报告: {out}")
    return 0




def _fp_shots_for_report(out):
    """扫描 out 目录下的 fp_*.png，返回 [{file, caption}]。

    shots/<tick>.png 有真截图的卡片标「游戏内截图」，否则标「示意图」。
    """
    from . import firstperson as _fp

    shots = []
    shots_dir = Path(out) / 'shots'
    for p in sorted(Path(out).glob('fp_*.png')):
        tick = p.stem.split('_')[-1]
        kind = '预瞄' if 'preaim' in p.stem else '站位'
        real = _fp.find_real_shot(shots_dir, tick) is not None
        tag = '（游戏内截图）' if real else '（示意图）'
        shots.append({'file': p.name, 'caption': f'{kind}问题 tick {tick}{tag}'})
    return shots

def _cmd_profile(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    frames = []
    for dpath in args.deaths:
        pth = Path(dpath)
        if not pth.exists():
            print(f"找不到 {pth}")
            return 1
        f = pl.read_csv(pth)
        f = f.with_columns(pl.lit(pth.stem).alias("demo_key"))
        frames.append(f)
    frame = pl.concat(frames, how="vertical_relaxed")
    out = Path(args.out) if args.out else (_out_dir(args, settings) / "profile")

    dims = profile.compute_dims(frame)
    labels = [d[1] for d in profile.DIMS]
    vals = profile.radar_points(dims)
    out.mkdir(parents=True, exist_ok=True)
    plot.radar_chart(labels, vals, out / "radar.png", title=f"{args.player or '玩家'} 六维能力")

    # 短板诊断卡（按影响死亡数 × 与职业差排序）
    cards = profile.short_board_cards(dims, frame)

    lines = [f"# {args.player or '玩家'} · 你的短板诊断", "",
             f"（{frame.height} 次死亡 / {len(args.deaths)} 场）", ""]
    total = frame.height
    for c in cards:
        val = c["value"]
        val_s = f"{val:.1f}" if isinstance(val, float) else str(val)
        diff_str = f"，比职业差 {c['diff']:.0f}%" if c.get("diff") is not None else ""
        pro_str = f"（职业 {c['pro']}）" if c.get("pro") is not None else ""
        lines.append(f"### 第{c['rank']}短板：{c['name']}（你的值 {val_s}{pro_str}{diff_str}）")
        if c.get("top_zones"):
            lines.append(f"- **发生在**：{('、'.join(c['top_zones']))}")
        # 量化影响 = 这些 top_zones 涉及的死亡数
        tz = c.get("top_zones") or []
        n_zone = frame.filter(pl.col("zone").is_in(tz)).height if tz else c['n']
        pct = n_zone / total * 100 if total else 0
        lines.append(f"- **量化影响**：这些位置共 {n_zone} 次死亡，占 {pct:.0f}%")
        if c.get("advice"):
            lines.append(f"- **改善方向**：{c['advice']}")
        lines.append("")

    # 全局画像
    lines += ["## 整体情况", ""]
    if "aim_off" in dims:
        v = dims["aim_off"]["value"]
        if v:
            lines.append(f"- 准星偏差中位 {v:.1f}°（敌人冒出来时你偏这么多）")
    if "zone_norm" in dims and dims["zone_norm"]["value"] is not None:
        cold = frame.filter(pl.col("pro_cumulative_percentile") < 0.2).height if "pro_cumulative_percentile" in frame.columns else 0
        lines.append(f"- 站位常规度 {dims['zone_norm']['value']}，{cold}/{frame.height} 次死在冷门位")
    lines += ["", f"置信度：样本 {frame.height} 次死亡；少于 10 个样本的维度仅供参考。"]

    (out / "profile.md").write_text("\n".join(lines), encoding="utf-8")

    # profile.csv
    csv_rows = [{"短板": c["rank"], "维度": c["name"], "你的值": c["value"], "样本": c["n"],
                 "职业": c["pro"], "差%": c["diff"], "发生在": "、".join(c["top_zones"]),
                 "改善方向": c["advice"]} for c in cards]
    pl.DataFrame(csv_rows).write_csv(out / "profile.csv")

    print(f"短板诊断生成: {out}")
    for c in cards:
        zones = "、".join(c["top_zones"][:2]) if c["top_zones"] else "-"
        print(f"  第{c['rank']}短板 {c['name']}（{c['value']}）发生在 {zones}")
    return 0


def _cmd_findings(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    import json as _json

    frame = pl.read_csv(Path(args.deaths))
    out = Path(args.out) if args.out else _out_dir(args, settings)
    map_name = args.map or (frame["zone"].drop_nulls().head(1).to_list() or [""])[0]

    # 加载 geo（拿不到就降级，findings 不带处方）
    geo, geo_err = geometry.try_build_geo(map_name or "de_ancient", settings)
    if geo is None:
        print(f"[降级] 无几何（{geo_err}），findings 不带 verify 处方")

    zlist = zones.load_zones(settings.config_dir, map_name or "de_ancient")
    zmap = {z.name: z for z in zlist}
    fl = findings.build_findings(frame, geo, zmap)

    payload = {
        "player": args.player if hasattr(args, "player") else None,
        "n_deaths": frame.height,
        "map": map_name,
        "findings": fl,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "findings.json").write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"findings.json 生成: {out / 'findings.json'}（{len(fl)} 条）")
    if not fl:
        if not zlist:
            print(f"[提示] {map_name or '?'} 还没有 zones：先跑 `csc zones suggest --map {map_name or 'de_ancient'} --write`")
        elif "fault" not in frame.columns:
            print("[提示] deaths.csv 没有 fault/zone 列：用新版 `csc analyze` 重新生成")
        else:
            print("[提示] 0 条：样本太少，或都够不上 findings 阈值")
    for f in fl:
        adv = f.get("advice", {}).get("action", "(无处方)")
        print(f"  [{f['kind']}] {f['zone']} n={f['n_deaths']} — {adv}")
    return 0


def _cmd_selftest(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    result = synthetic.run_selftest(settings, _out_dir(args, settings) / "selftest")
    print("== selftest ==")
    print(f"死亡点: {result['total']}")
    print(f"标签  : {result['labels']}")
    print(f"被拉死先手差: {result['gap_peeked']} ms")
    if result["errors"]:
        print("FAILED:")
        for e in result["errors"]:
            print(f"  - {e}")
        return 1
    print("OK: 管线串联通过，输出在", result["out"])
    return 0


def _cmd_zones(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    map_name = args.map or "de_ancient"
    if args.action == "list":
        for m in settings.maps:
            zs = zones.load_zones(settings.config_dir, m)
            print(f"{m}: {len(zs)} zones")
        return 0

    if args.action == "check":
        zs = zones.load_zones(settings.config_dir, map_name)
        print(f"{map_name}: {len(zs)} zones")
        if len(zs) < 20:
            print(f"  [!] 偏少（目标 20-40）")
        # 重叠检查：两 zone 中心距 > min(r1,r2)
        bad = 0
        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                d = ((zs[i].center[0] - zs[j].center[0]) ** 2 + (zs[i].center[1] - zs[j].center[1]) ** 2) ** 0.5
                if d <= min(zs[i].radius, zs[j].radius):
                    print(f"  [!] 中心重叠: {zs[i].name} <-> {zs[j].name}")
                    bad += 1
        print(f"  重叠 {bad} 对，覆盖 {'OK' if len(zs) >= 20 and bad == 0 else '待调'}")
        return 0

    # suggest：用职业落位点聚类生成候选
    from . import benchmark_pro  # 局部导入

    bench = benchmark_pro.ProBenchmark(settings, map_name)
    if not bench.ready:
        print(f"职业落位点不可用（{map_name}），无法 suggest")
        return 1
    pts = bench._points or []
    cands = zones.suggest_zones(pts, n=args.n, eps=args.eps)
    print(f"{map_name}: 生成 {len(cands)} 个候选 zone（来自 {len(pts)} 职业落位点）")
    for z in cands:
        print(f"  {z.name}: center={z.center} radius={z.radius}")
    if args.write:
        path = zones.save_zones(settings.config_dir, map_name, cands, note="csc zones suggest 生成，待人工改名")
        print(f"已写入 {path}")
    else:
        print("（加 --write 写入 zones.yaml；之后人工改名）")
    return 0


def _cmd_maps(args: argparse.Namespace, settings: Any) -> int:
    bootstrap_env()
    action = args.action
    if action == "list":
        return _maps_list(settings)
    if args.map is None:
        print("需要 --map <map_name>")
        return 1
    if action == "check":
        return _maps_check(args, settings)
    if action == "show":
        return _maps_show(args, settings)
    if action == "fetch":
        return _maps_fetch(args, settings)
    if action == "apply":
        return _maps_apply(args, settings)
    return _maps_edit(args, settings)


def _maps_list(settings: Any) -> int:
    from . import radar

    version_dir = radar.find_version_dir(settings.paths.awpy_data)
    print("== 地图配置（maps.yaml + points.yaml 合并后）==")
    if version_dir is None:
        print(f"地图资源：未下载（`csc maps fetch --map de_mirage`；目录 {settings.paths.awpy_data}）")
    else:
        print(f"地图资源：{version_dir}（{len(radar.available_maps(settings.paths.awpy_data))} 张）")
    for name, spec in settings.maps.items():
        problems = spec.validate()
        entries = sum(len([ep for ep in eps if ep.usable]) for eps in spec.entry_points.values())
        covers = sum(len(cps) for cps in spec.cover_points.values())
        has_radar = bool(version_dir and radar.find_radar(version_dir, name))
        status = "[ok]" if not problems else f"[!] {len(problems)} 个问题"
        print(f"{name:<18} {status:<14} 进入点 {entries:>2}  掩体点 {covers:>2}  雷达 {'有' if has_radar else '无'}")
        for pr in problems:
            print(f"    - {pr}")
    return 0


def _maps_check(args: argparse.Namespace, settings: Any) -> int:
    spec = settings.map_spec(args.map)
    print(f"{args.map}: 区域 {list(spec.regions)}")
    for region in spec.regions:
        eps = [ep for ep in spec.entry_points.get(region, ()) if ep.usable]
        cps = spec.cover_points.get(region, ())
        print(f"  {region}: 进入点 {len(eps)} 个, 掩体点 {len(cps)} 个")
    problems = spec.validate()
    for pr in problems:
        print(f"  - {pr}")
    print("结论:", "可用" if not problems else "未配齐（相关指标会为 null）")
    return 0


def _maps_show(args: argparse.Namespace, settings: Any) -> int:
    from . import map_points, radar

    spec = settings.map_spec(args.map)
    print(f"== {args.map} ==")
    print(f"区域: {', '.join(spec.regions) or '（无）'}    默认区域: {spec.default_region}")
    for region in sorted(spec.regions):
        raw = spec.regions[region]
        print(f"  [{region}] center={raw.get('center')} radius={raw.get('radius')}")
        for ep in spec.entry_points.get(region, ()):
            print(f"      entry {ep.name:<16} {ep.pos}")
        for cp in spec.cover_points.get(region, ()):
            print(f"      cover {cp.name:<16} {cp.pos}")

    try:
        tf = radar.load_transform(
            str(args.map),
            settings.paths.awpy_data,
            version=args.version,
            lower=args.lower,
            fetch=False,
        )
        bounds = tf.world_bounds
        print(
            f"雷达: {tf.radar}  {tf.width}x{tf.height}  pos=({tf.pos_x}, {tf.pos_y})  "
            f"scale={tf.scale}  rot={tf.rotation_deg}"
        )
        print(f"世界范围: x∈[{bounds[0]:.0f}, {bounds[1]:.0f}]  y∈[{bounds[2]:.0f}, {bounds[3]:.0f}]")
    except radar.RadarError as exc:
        print(f"雷达/变换: 不可用（{exc}）")

    try:
        doc = map_points.load_points(settings.config_dir)
        rows = [r for r in map_points.summarize(doc) if r.map_name == args.map]
        print(f"points.yaml: {len(rows)} 条 · {map_points.points_path(settings.config_dir)}")
    except map_points.PointsError as exc:
        print(f"points.yaml: 读取失败（{exc}）")
    return 0


def _maps_fetch(args: argparse.Namespace, settings: Any) -> int:
    from . import radar

    tag = args.version or radar.latest_tag()
    print(f"拉取 awpy-data {tag} 的迷你资源（map_data.json + images.zip ≈ 3MB）...")
    try:
        version_dir = radar.fetch_assets(settings.paths.awpy_data, tag=tag, maps=[str(args.map)], log=print)
    except Exception as exc:  # noqa: BLE001 - 网络/磁盘错误都提示清楚
        print(f"下载失败：{exc}")
        print("离线做法：手动放 .awpy/<version>/map_data.json + .awpy/<version>/radars/<map>.png")
        return 1
    print(f"资源目录: {version_dir}")
    try:
        tf = radar.load_transform(
            str(args.map),
            settings.paths.awpy_data,
            version=tag,
            lower=args.lower,
            fetch=False,
            rotation_deg=args.rotate_deg,
        )
    except radar.RadarError as exc:
        print(f"变换读取失败：{exc}")
        return 1
    print(f"变换: pos=({tf.pos_x}, {tf.pos_y})  scale={tf.scale}  rot={tf.rotation_deg}  雷达={tf.radar}")
    print("碰撞网格（VisibilityChecker 用）仍需 `awpy get`；打点/坐标换算不需要它。")
    return 0


def _maps_edit(args: argparse.Namespace, settings: Any) -> int:
    from . import maps_edit, radar

    map_name = str(args.map)
    try:
        tf = radar.load_transform(
            map_name,
            settings.paths.awpy_data,
            version=args.version,
            lower=args.lower,
            fetch=not args.no_fetch,
            rotation_deg=args.rotate_deg,
            log=print,
        )
    except radar.RadarError as exc:
        print(f"地图资源不可用：{exc}")
        print(f"先跑：python -m csc maps fetch --map {map_name}")
        return 1

    payload = maps_edit.build_payload(map_name, settings, tf, serve=not args.no_serve, default_z=args.z)
    if args.no_serve:
        out = Path(args.html) if args.html else _out_dir(args, settings) / "maps" / f"edit-{map_name}.html"
        path = maps_edit.write_html(payload, out)
        print(f"静态打点页: {path}")
        print(f"打开它点位置 -> 导出 YAML -> python -m csc maps apply --map {map_name} --points <文件>")
        return 0

    print(f"打点器: {map_name}（浏览器里点位置，Ctrl+C 结束服务）")
    return maps_edit.serve(
        payload,
        settings.config_dir,
        port=args.port,
        open_browser=not args.no_open,
    )


def _maps_apply(args: argparse.Namespace, settings: Any) -> int:
    import json as _json

    import yaml as _yaml

    from . import map_points, maps_edit

    map_name = str(args.map)
    if not args.points:
        print("需要 --points <文件>（传 - 则从 stdin 读 YAML/JSON）")
        return 1

    try:
        if args.points == "-":
            text = sys.stdin.read()
            try:
                doc = _json.loads(text)
            except _json.JSONDecodeError:
                doc = _yaml.safe_load(text) or {}
            if not isinstance(doc, dict):
                print("stdin 内容不是有效 YAML/JSON")
                return 1
            if isinstance(doc.get("maps"), dict):
                payload = {"map": map_name, **(doc["maps"].get(map_name) or {})}
            else:
                payload = {"map": map_name, **doc}
        else:
            payload = maps_edit.load_map_file(Path(args.points), map_name)
        path, block = map_points.apply_payload(
            payload,
            settings.config_dir,
            map_name=map_name,
            source="csc maps apply",
        )
    except map_points.PointsError as exc:
        print(f"坐标不合法：{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - 读文件/格式错误
        print(f"导入失败：{exc}")
        return 1

    print(f"已写入 {path}")
    print(f"  进入点 {dict((r, len(v)) for r, v in (block.get('entry_points') or {}).items())}")
    print(f"  掩体点 {dict((r, len(v)) for r, v in (block.get('cover_points') or {}).items())}")

    fresh = load_settings(args.config_dir)
    spec = fresh.map_spec(map_name)
    problems = spec.validate()
    print("门禁:", "可用" if not problems else "未配齐（相关指标会为 null）")
    for pr in problems:
        print(f"  - {pr}")
    return 0


def _cmd_train(args: argparse.Namespace, settings: Any) -> int:
    """训练职业基准：解析 pro demo -> 缓存落位点 + 网格，供 analyze/report 对比。"""
    bootstrap_env()
    phases = tuple(p.strip() for p in (args.phases or "").split(",") if p.strip())
    if not phases:
        print("--phases 不能为空（early/mid/late）")
        return 1
    maps = [args.map] if args.map else None
    print(f"开始训练职业基准（阶段={','.join(phases)}，force={args.force}）...")
    results = benchmark_pro.train_all(settings, maps=maps, phases=phases, force=args.force, verbose=True)

    print("\n=== 训练结果 ===")
    for map_name, meta in results.items():
        if meta.get("error"):
            print(f"[!] {map_name}: 出错 {meta['error']}")
        elif meta.get("skipped"):
            print(f"[-] {map_name}: 跳过（{meta['skipped']}），demo={meta.get('n_demos', 0)}")
        else:
            tag = "缓存" if meta.get("cached") else "新建"
            print(f"[+] {map_name}: {meta['n_points']} 落位点 / {meta['n_grid_cells']} 网格 "
                  f"/ {meta['n_demos']} demo（{tag}，签名 {meta['signature'][:8]}）")
    ok = [m for m, meta in results.items() if meta.get("n_points")]
    print(f"\n共 {len(results)} 张图，{len(ok)} 张有落位数据：{', '.join(ok) if ok else '无'}")
    print("之后 analyze/report 的职业分位对比会直接读这份缓存，无需再解析 demo。")
    return 0


# --------------------------------------------------------------------------- 入口


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK(cp936)，中文能编码但 emoji 会崩、且 Web GUI 显示乱码。
    # 统一 UTF-8，一次性解决崩溃 + 乱码。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 某些环境不支持 reconfigure
            pass

    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(args.config_dir)

    dispatch = {
        "env": _cmd_env,
        "parse": _cmd_parse,
        "analyze": _cmd_analyze,
        "report": _cmd_report,
        "eval": _cmd_eval,
        "profile": _cmd_profile,
        "findings": _cmd_findings,
        "selftest": _cmd_selftest,
        "zones": _cmd_zones,
        "maps": _cmd_maps,
        "train": _cmd_train,
    }
    handler = dispatch[args.command]
    try:
        return int(handler(args, settings))
    except demo_io.DemoError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except geometry.GeoUnavailable as exc:
        print(f"几何不可用: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
