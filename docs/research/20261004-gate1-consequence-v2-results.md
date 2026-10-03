# Gate 1 consequence bridge v2: contract-corrected probe

这次 Probe 修正了旧 v2 的采集时序。旧 wrapper 在 `env_step` 之后保存
`object_root`、hand pose 和 force；因此 row `t` 的物理量对应 `x_{t+1}`，而
`return_to_go[t]` 仍从 `r_t` 开始。旧的 horizon 结果标记为历史实现产物，不能
用于拒绝或支持 Gate 1。

修正后的 collector 在 `env_step` 前保存物理量并写入
`physical_timing=pre_env_step`。assembler 对没有该字段的旧 shard 采用显式
`post_env_step_legacy` 一步回移，并从可对齐的窗口开始。future `E/I` 仍取
`t+1:t+H`，因此第一帧对应执行动作 `a_t` 后的状态。相对四元数沿时间轴做
q/−q 连续化；world-frame hand/object force 在每个 future object frame 中旋转；
contact mask 阈值与 live contact predicate 的 `0.1` 对齐。fit split 使用
`(source_run, episode_id)`，所有 ablation arm 共用同一初始化 seed。

fresh collection `tmp/fresh_pre_s86d` 有 6,577 条 transition、12 个 episode。
`object_root` 的位置和速度与 `state[:,36:49]` 精确一致，而与
`next_state` 不一致；四元数按 q/−q 等价后的最大误差为 `1.8e-7`。五个 horizon
均通过 v2 dataset audit，形状为 history `[N,10,509]`、effect `[N,H,13]`、
interaction `[N,H,80]`。

同一个 episode split（10 train / 2 test）、30 epochs、同 seed 的 `V_HEI` 相对
`V_H` 结果如下：

| horizon | V_H MAE | V_HEI MAE | relative change | episode bootstrap CI |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 6.724 | 7.815 | -16.2% | [-5.20, 3.25] |
| 5 | 5.452 | 5.158 | +5.4% | [-2.07, 2.79] |
| 10 | 5.659 | 6.727 | -18.9% | [-2.56, 0.50] |
| 16 | 5.055 | 9.188 | -81.7% | [-4.59, -3.65] |
| 32 | 6.817 | 7.824 | -14.8% | [-2.49, 0.57] |

这只是小样本 Probe：测试 episode 只有 2 个，不能升级为 `SUPPORTED` 或
`REFUTED`。当前最稳妥的状态是：实现契约已修复；在这个 fresh actor/data
样本上没有达到预设的 10% 且 CI 排除零的 Gate 条件，因此不启动 Cm 训练，
也不把旧 v1/v2 的负结果外推为“交互信息普遍无效”。

主要 artifacts 位于项目目录下的 `tmp/fresh_pre_s86d*`；可复算代码为
[`assemble_gate1_dataset_v2.py`](../../scripts/assemble_gate1_dataset_v2.py)、
[`assemble_gate1_dataset_v2_fast.py`](../../scripts/assemble_gate1_dataset_v2_fast.py)、
[`fit_gate1_value_bridge_v2.py`](../../scripts/fit_gate1_value_bridge_v2.py) 和
[`run_gate1_consequence_environment.py`](../../scripts/run_gate1_consequence_environment.py)。

## Larger follow-up and future-action control

为回应独立 episode 数过少的问题，又在同一修正后的 collector 上并行采集了
两组 fresh run：每组约 15.2k transitions、28 episodes，合计 56 episodes。
两组共包含 1 个 stable-success 和 1 个 drop-after-success。合并数据的
horizon sweep 仍使用同一个 GRU bridge 和 composite episode split。

使用 episode-balanced MAE 后，五个 split 上 h16 的 `V_HEI` 相对 `V_H`
改善为约 `16.6%`、`9.8%`、`29.8%`、`18.6%`、`24.0%`；其中两个 episode
bootstrap CI 排除零，其余仍跨零。h3、h5、h10、h32 没有同样稳定的改善，因此这只能算 h16 的
`PROMISING` 方向性 Probe，尚未达到 Validation 或 Gate closeout。

由于未来 `E/I` 仍来自 on-policy trajectory，额外加入 `future_action`
作为诊断控制。五个 h16 split 的 `V_HFEI` 相对 `V_HF` 变化为
`-9.2%`、`+10.8%`、`+27.5%`、`+15.9%`、`+23.3%`。这说明第一个 split
的增益可由未来策略行为解释，但其余 split 仍显示 E/I 的额外信号；这些
control CI 多数仍跨零。下一步若要形成正式结论，需要更多 success/drop
覆盖、预注册 split 和 matched future-action control；当前不启动在线 Cm
训练，也不把 h16 的方向性结果写成 `SUPPORTED`。
