---
schema: ref2dex.probe.v2
probe_id: P-20261016-shared-action-horizon
experiment_id: P-20261016-shared-action-horizon
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: 7ff34069
claim_id: C1
hypothesis_family: HF-shared-action-horizon
probe_index_in_family: 1
seed_pool: probe
seeds: [198, 192, 148]
decision_changed_if_positive: use current721 plus first-action supervision as the next matched shared route
decision_changed_if_negative: retain full-chunk supervision for current-input shared training and investigate input bottleneck separately
status: RUNNING
run_id: shared-action-horizon-20261016-r1
---

# 当前输入与动作监督跨度的四组对照

Class: Decision。这个 Probe 把两个仍未分开的因素放进同一 shared7k 起点：当前输入是否有用，
以及对完整 24 拍动作输出只监督前 1 拍还是监督前 8 拍。它服务于全局目标 C1，结果用于
决定后续 shared actor 是保留当前 721D、缩短动作监督跨度，还是先回到完整 chunk 目标。

四个标签分别为 `masked-8`、`delta0-8`、`masked-1`、`delta0-1`。四臂网络、forecast
监督、数据准入、batch、学习率、seed、动作表示和 checkpoint 选择保持一致。8 拍臂沿用
`P-20261015-shared-current-continuation` 的 1k 阶段，并从各自 `step1000.pt` 恢复 Adam
状态追加 6000 步；1 拍臂从同一 shared7k+weight-only robot parent 起步，optimizer 重置，
训练到 7000 步。单拍臂仍产生完整的 24×15D geometry-lift 动作，只把 action/anchor L1
的 query mask 设为第 0 拍；forecast hand/scene loss 始终覆盖完整 horizon。

已有 1k 8 拍阶段先给出工程和行为基线：delta0 action 0.3979、首控 PD XYZ 均值 60.9 mm，
masked action 0.4077、61.7 mm；三对象固定面板（airplane/spheresmall/mug，seed148）
已完成且独立 native audit 通过。它们不构成正式结论，最终比较以四臂相同阶段和相同固定面板
为准。1 拍臂的 action supervision 代码在 `7ff34069`，默认值仍为完整预测 horizon，
因此旧实验语义不变。

GPU 安排：GPU4/5 运行新增 `masked-1`/`delta0-1` 7000 步；GPU6/7 运行 `masked-8`/
`delta0-8` 从 step1000 恢复到 step7000；GPU1 扩大已有腕部 MLP 的 airplane、spheresmall、
mug 固定面板；GPU0 外部作业不动，GPU2/3 空闲。所有训练监督器绑定本臂进程，监测 GPU、
显存、磁盘和 stall；每臂上限 14000 秒、7GB，磁盘低于 20GiB 或出现外来 GPU 进程即停止
该臂并保留已有 checkpoint。预计单拍 7000 步约 3 小时，随后运行相同三对象 native 面板。

判定仍使用 Probe 标签：比较首控 PD XYZ 的 mean/p95、完整动作离线误差和固定面板结果；
单 seed 与短 episode 不能形成 SUPPORTED/REFUTED 结论。若单拍在 delta0 与 masked 之间
同时提高 teacher-positive held 序列数并不损害 forecast，则进入多 seed Validation；否则
记录为 UNCLEAR/UNPROMISING，继续拆解 721D 的分块贡献和 residual-policy 可行性。

## 当前执行记录

- 8 拍已完成 1k：`outputs/ptv3-shared-wam-v2/shared-current-continuation-20261015-r2/`。
- 8 拍追加运行目录：`outputs/ptv3-shared-wam-v2/shared-action-horizon-20261016-r1/continuation-*`。
- 1 拍训练目录：`outputs/ptv3-shared-wam-v2/shared-action-horizon-20261016-r1/single-*`。
- 固定面板和腕部 MLP 结果写入同一 run 目录的 `native/` 与 `wrist-panel/`。

2026-10-16 02:10 CST，原 `continuation-masked8`/`continuation-delta08` 在完整
step4000 checkpoint 后被监督器以预设 7GB artifact budget 停止；不是数值、梯度或
GPU 异常，step4000 权重和日志完整保留。随后从这两个 step4000 恢复 Adam/RNG，启动
`continuation-masked8-sparse`/`continuation-delta08-sparse` 到 step7000。该资源恢复臂
只改变 validation/checkpoint cadence 为 3000，模型、数据、优化器和学习率字段严格保持
一致；实际新增配置和 sparse 源码均保存在项目 `tmp/` 与 run 输出中。

## Implementation audit and corrective rerun

2026-10-16 04:55 CST 的结果审查发现，原 `continuation-masked8.json` 和
`continuation-delta08.json` 没有写入 `action_supervision_steps=8`。在
`robot_loss` 的默认合同中，缺省值是预测 horizon（24），所以这两条运行实际是
24-step action supervision；它们的 step7000 权重、训练曲线和 native panel 全部保留，
但对本 Probe 的“8拍”臂标为 `INVALID_IMPLEMENTATION`，不用于四组因果比较。single-
masked1/single-delta01 显式设为1，语义有效。

由于 shared-current 1k checkpoint 本身已经在错误的24拍语义下更新，直接从它续接会把
错误的前1k历史带入“8拍”臂。为恢复四组的共同起点，root 选择从同一个 shared7k+
weight-only robot parent fresh 重跑两条 8拍臂；这比继续污染的 optimizer state 多花约3小时，
但仍在本卡 deadline 和 GPU 预算内。新配置显式写入
`action_supervision_steps=8`，并使用稀疏 validation/checkpoint cadence 以满足每臂7GB
artifact budget；输出目录为 `fresh8-masked` 和 `fresh8-delta0`。原 continuation
输出保留作资源与错误实现证据，不参与四组比较。
