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
两组共包含 1 个同时满足 stable-success 与 drop-after-success 的 episode；另一组
没有这两类 outcome。合并数据的
horizon sweep 仍使用同一个 GRU bridge 和 composite episode split。

使用 episode-balanced MAE 后，五个 split 上 h16 的 `V_HEI` 相对 `V_H`
改善为约 `16.6%`、`9.8%`、`29.8%`、`18.6%`、`24.0%`；其中两个 episode
bootstrap CI 排除零，其余仍跨零。h3、h5、h10、h32 没有同样稳定的改善，因此这只能算 h16 的
`PROMISING` 方向性 Probe，尚未达到 Validation 或 Gate closeout。

由于未来 `E/I` 仍来自 on-policy trajectory，额外加入 `future_action`
作为诊断控制。使用同一 episode-balanced MAE 重新拟合的五个 h16 split 如下：

| split | `V_HFEI` vs `V_HF` | episode bootstrap CI |
| ---: | ---: | ---: |
| 1 | -9.0% | [-1.69, 0.64] |
| 2 | +10.9% | [0.11, 13.45] |
| 3 | +28.9% | [0.76, 2.74] |
| 4 | +19.7% | [-1.07, 5.06] |
| 5 | +23.2% | [-0.49, 7.54] |

这说明未来策略动作可以解释 split 1 的全部增益；其余 split 仍有 E/I 的额外
方向性信号，但只有 split 2、3 的 CI 排除零。下一步若要形成正式结论，需要更多
success/drop 覆盖、预注册 split 和 matched future-action control；当前不启动在线
Cm 训练，也不把 h16 的方向性结果写成 `SUPPORTED`。

### Held-out episode influence audit

为检查 aggregate MAE 是否被少数 episode 主导，fit 报告现在保留每个 held-out
episode 的 MAE（`heldout_episode_error_table`）。在五个 e260 h16 split 中，`V_HEI`
相对 `V_H` 的 episode-level 正向 episode 数分别为 `6/11`、`7/11`、`10/11`、
`9/11`、`8/11`；但总增益中最大单 episode 的占比分别约为 `94%`、`96%`、`20%`、
`70%`、`67%`。移除该最大贡献 episode 后，前两个 split 的平均增益只剩约 `0.11`
和 `0.20` MAE，说明它们不能被当作普遍改善。该审计不改变预先定义的 gate 统计量，
但把 h16 的状态收紧为需要按 episode 类型和 success/drop 覆盖补样的
`PROMISING` 诊断方向。

## Independent e420 diagnostic actor

为检查 e260 actor 分布依赖，另外使用文档中已登记的 e420 `plain_off`
s286/s287 checkpoint 做了明确隔离的 diagnostic Probe。两组 pre-step run
合计 44 episodes、23.7k transitions，其中 s286 的同一个 episode 同时满足
stable-success 与 drop-after-success，s287 没有这两类 outcome。h16 的五个
composite split 中，`V_HEI` 相对 `V_H` 的
episode-balanced MAE 变化为 `+2.9%`、`+2.8%`、`-18.2%`、`+24.4%`、
`+3.7%`；只有一个 split 的 bootstrap CI 排除零。该 actor 上没有复现
e260 的稳定 h16 方向，因此这个 diagnostic 分支停止扩展，并继续与 pinned
e260 结果分开报告。

## Independent e260 coverage extension

为检验单个 success/drop episode 的影响，又从同一 pinned e260 checkpoint 采集了
两个独立 run（seed namespace 1864/1865），各 28 episodes，合计 29,111 个 h16
windows。1864 有 1 个同时满足 stable-success 与 drop-after-success 的 episode，
1865 没有这两类 outcome；dataset audit 通过。

在这组独立数据上，五个固定 split 的 `V_HEI` 相对 `V_H` 变化为
`+25.9%`、`+32.2%`、`+8.3%`、`-1.0%`、`-2.1%`，所有 episode bootstrap CI
均跨零。future-action control 的 `V_HFEI` 相对 `V_HF` 为
`+45.9%`、`+42.2%`、`+19.7%`、`+15.6%`、`+17.7%`；这只说明在该样本上 E/I
相对未来动作仍有诊断信号，不能抵消直接 `V_HEI`/`V_H` 的不稳定。

将四个 e260 run 合并为 112 episodes 做 exploratory fit，`V_HEI` 相对 `V_H`
为 `+12.1%`、`+15.0%`、`-25.8%`、`+29.8%`、`+5.4%`，只有 2/5 CI 排除零；
matched control 为 `+26.7%`、`+5.8%`、`-5.2%`、`+12.0%`、`+9.0%`，仅 1/5
CI 排除零。样本量扩展没有消除 split 敏感性，因此停止继续扩大该 actor 的
collector，仍不启动 Cm 训练。若进入正式 Validation，应预注册 actor/episode
cluster bootstrap，并补充多个独立 success/drop outcome。
