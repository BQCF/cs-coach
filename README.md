# cs-coach

CS2 demo 离线体检：**死亡归因 + 站位暴露面 + 职业基准 + 第一人称复盘**。

输入 `.dem`（官匹 / Faceit / 完美平台 / HLTV，awpy 能解析的都行），输出每个死亡点的
体检表（`deaths.csv`）、按位置讲故事的报告（`report.md` + 单文件 `report.html`）、
落位热图、六维画像、可执行的改善处方（`findings.json`）。

**不做**：实时分析、游戏内提示、自动瞄准。

## 功能

- **死亡归因 v2（8 类）**：被闪 / 背身侧身 / 信息缺失硬拉 / 被拉死 / 架死 / 正面遭遇 / 移动中对枪 / 其他对枪（空弹匣等状态进 `clip_state` 列，不做死因标签）
- **指标**：暴露面 `exposure`、交叉火力 `crossfire`、先手差 `first_seen_gap_ms`（FOV 版）、可预瞄 `predictable`、退路 `escape_time` + 预瞄偏差 / 对枪数据（谁先开火、命中、伤害）
- **职业基准（M4）**：`train` 训练落位网格，报告里对比「职业常站度」
- **位置体系（N2）**：`zones` 自动聚类位置，报告按 zone 讲故事 + 处方（`findings`）
- **画像（N4）**：六维能力雷达 + 短板排序（`profile`）
- **归因评估（M3）**：自动 vs 人工标注，准确率 / 精确率 / 召回率（`eval`）
- **第一人称复盘**：示意图兜底 + **真截图工作流**——把游戏截图放进 `shots/<tick>.png`，
  报告自动把红X（敌人死时位置）/绿+（敌人冒头处）/准星标注**投影叠加**到真截图上
- **半自动截真图（N7）**：CS2 netcon 远程驱动，一键报告（`scripts/make_report.py`）

## 安装

要求 Python 3.12+（开发环境为 3.12.10）。

```powershell
cd cs-coach
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"        # 国内建议 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m csc env                        # 环境自检

# 地图资源（雷达图 + 坐标变换，约 3MB）
python -m csc maps fetch --map de_mirage

# 碰撞网格（exposure/可见性/第一人称渲染要用，几百 MB）
awpy get
```

## 快速开始

```powershell
# 1. 归因 + 指标 -> deaths.csv（自动出 report.md / report.html / 热图 / 第一人称图）
python -m csc analyze --demo data/demos/me/xxx.dem --player "你的昵称"

# 2. 无 demo 自检（合成数据跑通全管线）
python -m csc selftest

# 3. 职业基准（先训一次，缓存后复用）
python -m csc train --map de_inferno

# 4. 人工标注回归（tests/fixtures/labels.csv）
python -m csc eval --labels tests/fixtures/labels.csv --deaths output/deaths.csv
```

## CLI 一览

| 命令 | 作用 |
| --- | --- |
| `env` | 环境自检（版本 / 资源 / demo 存量） |
| `parse` | 解析 demo + 列契约核对（awpy 版本漂移时先跑它） |
| `analyze` | 归因 + 指标 -> `deaths.csv` + 报告 + 图 |
| `report` | 对已有 `deaths.csv` 重出报告（`report.md` + `report.html`） |
| `eval` | 自动归因 vs 人工标注，算准确率（按 demo+tick 精确对齐） |
| `profile` | 跨场画像：六维雷达 + 短板排序 |
| `findings` | `findings.json`：zone 级问题 + verify 处方 |
| `selftest` | 合成数据跑通管线（门禁） |
| `zones` | 位置体系：`suggest / check / list` |
| `maps` | 地图配置 / 雷达打点器：`list / check / show / fetch / edit / apply` |
| `train` | 训练职业基准网格（解析 pro demo，缓存落位） |

`--output` / `--config-dir` 是全局参数，写在**子命令前面**：
`python -m csc --output out report ...`。`parse` / `analyze` 支持 `--dry-run`。

## 真截图工作流（第一人称复盘）

`.dem` 只记录位置 / 朝向 / 事件，**不含游戏画面**——真截图必须由游戏引擎渲染。
两条路：

1. **手动（每张 30 秒）**：`analyze` 会生成 `replay_guide.md`，列出每张卡片的
   tick 和现成命令；CS2 控制台 `playdemo <demo>` → `demo_gototick <tick>` →
   截图（F12 / `jpeg`）→ 放进 `<输出目录>/shots/<tick>.png` → 重跑 `csc report`，
   卡片自动换成「游戏内截图 + 标注」（相机参数来自 demo，标注精确投影）。
2. **半自动**：`scripts/make_report.py` 用 CS2 的 netcon（`cs2.exe -console -netconport 2121`）
   自动完成 跳 tick → 截图 → 收集 → 重出报告。netcon 只监听本地，**别把端口暴露公网**。

> 截图建议原生分辨率（4:3 拉伸会让标注投影偏位）。

## 目录约定

```
csc/                 # 包（python -m csc）
config/default.yaml  # 阈值 / 窗口 / 眼高（无魔数）
config/maps.yaml     # 地图区域意图 / 门禁（人写，工具不改）
config/points.yaml   # 进入点/掩体点坐标（打点器或 scripts/derive_entry_points.py 产出）
config/zones.yaml    # 位置体系（zones suggest 产出）
scripts/             # 独立工具（聚类坐标 / 职业热图 / 自动截真图等）
data/demos/{me,pro}/ # 自己的 / 职业 demo（不进 git）
data/cache/          # 解析后的 parquet 缓存
output/              # deaths.csv / report.md / report.html / 图 / shots/
tests/               # 单测（pytest，含 golden labels）
```

**缓存与产物不进 git**（`.gitignore` 已覆盖 `.venv/ .awpy/ .tmp/ data/ output/ *.dem`）。

## 归因规则（v2，按优先级命中即停）

| # | 标签 | 判定 |
| --- | --- | --- |
| 1 | 被闪 | 死亡前 2s 内被闪且 duration ≥ 1.2s |
| 2 | 背身/侧身 | 击杀者方向与死者朝向夹角 > 90° |
| 3 | 信息缺失硬拉 | 主动推进中被杀、看不到击杀者、无队友掩护 |
| 4 | 被拉死 | 击杀者从遮挡后出现且死者几乎静止 |
| 5 | 架死 | 双方提前 0.5s 已互相可见且死者站住 |
| 6 | 正面遭遇 | 双方几乎同时看见（先手差 <60ms）的站住对枪 |
| 7 | 移动中对枪 | 死前 0.5s 均速 > 60 u/s |
| 8 | 其他/对枪 | 兜底 |

所有阈值在 `config/default.yaml`，代码无魔数。指标带 sanity check，超出合理区间
（如先手差 ∉ [-300, +400]ms）会在 `metric_notes` 标「可疑」。

## 指标口径要点

`first_seen_gap_ms` 用**有效视野 FOV（≈90°）**而非 360° 几何通视：`is_visible` 是
360° 的，直接拿它算先手差会恒等于 0。正数 = 他先看见你（吃亏）。

## 已知边界

- `.dem` 不含画面：「游戏内截图」只能来自游戏引擎（见真截图工作流）；
- 无 mesh / 未配坐标时几何指标降级为 `null`，不报错（`metric_notes` 说明原因）；
- 职业基准按 地图 × 经济档 × 人数状态 控制混杂变量，样本不足不下结论。

## 测试

```powershell
python -m pytest tests      # 94 个单测
python -m csc selftest      # 8 场景各命中一条归因规则的管线门禁
```

## License

MIT（见 [LICENSE](LICENSE)）。
