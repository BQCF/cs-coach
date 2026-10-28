# 人工标注集（golden labels）

归因规则的回归基准。格式 `tests/fixtures/labels.csv`：

```
demo,death_tick,label,note
```

- `demo`：demo 文件名（对应 `data/demos/me/` 或 `pro/` 下的文件）
- `death_tick`：死亡 tick（从 `deaths.csv` 的 `death_tick` 列复制）
- `label`：人工判定标签（被闪 / 背身侧身 / 打空换弹 / 移动中对枪 / 被拉死 / 架死 / 信息缺失硬拉 / 其他对枪）
- `note`：可选的判定理由

## 怎么用

1. 跑 `python -m csc analyze --demo ...`，得到 `output/deaths.csv`；
2. 抽 10–50 个死亡点，人眼看 demo 回放（或看 `output/death_*.png`）标 `label`；
3. 每改一次归因规则，跑一遍对比脚本（待 M1.5 加 `csc eval --labels tests/fixtures/labels.csv`）看准确率涨跌。

没有这个基准，归因规则就是拍脑袋调参。
