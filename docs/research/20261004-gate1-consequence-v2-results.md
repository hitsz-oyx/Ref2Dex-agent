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

## Interaction-delta representation diagnostic

为回应 ref1 对 interaction 事件信息不足的担忧，在不重新采集的前提下，对四个
e260 run 的 112-episode h16 dataset 做了离线 I+ 对照。I+ 保留原 80 维
contemporaneous object-frame interaction，并追加可从已有 tensor 重建的相对
object-frame relative-velocity increment、hand/object force increment 和 hand/object
contact increment，得到 119
维；target、split、GRU bridge 和 seed 不变。audit 与合成时序测试均通过。

e260 的 h16 `V_HEI`/`V_H` 变化为 `+29.1%`、`+20.2%`、`-18.5%`、`+25.8%`、
`+18.3%`，四个 split 的 CI 排除零；future-action control 为
`+34.6%`、`+13.9%`、`-6.4%`、`+12.9%`、`+10.9%`，control CI 全部跨零。
I+ horizon sweep 的正向点估计数为 h3 `3/5`、h5 `5/5`、h10 `4/5`、h16
`4/5`、h32 `4/5`；排除零的 CI 数分别为 `1/5`、`2/5`、`1/5`、`4/5`、`3/5`。
因此好转集中在 h16/h32，但仍有一个 split 跨 horizon 反向，不能形成 Gate
closeout。

最后在独立 e420 actor 上复用 I+ h16：直接变化为 `+19.4%`、`+3.5%`、`-8.5%`,
`+23.6%`、`+4.0%`，matched control 为 `+14.4%`、`+2.4%`、`-6.6%`、
`+21.2%`、`+4.8%`；两者均只有 2/5 CI 排除零。该跨 actor 结果支持继续设计
正式 Validation，但不支持继续堆叠 feature 或启动 Cm；下一步应预注册
actor/outcome cluster bootstrap，并增加多个独立 success/drop episode。

### Source-run cluster bootstrap audit

为估计 actor/run 内 episode 相关性对不确定性的影响，新增脚本
[`audit_gate1_actor_cluster_bootstrap.py`](../../scripts/audit_gate1_actor_cluster_bootstrap.py)
对每个 fit report 的 held-out episode 表按 `source_run` 聚类重采样 10,000 次。
这一步只审计不确定性，不改 primary fit。e260 四个 cluster 下，I+ h16 直接比较的
五个 split 区间为 `[0.935, 7.905]`、`[0.888, 2.039]`、`[-16.167, 2.018]`、
`[1.277, 3.803]`、`[0.646, 1.330]` MAE；future-action control 为
`[-1.440, 13.241]`、`[0.359, 1.835]`、`[-8.423, 2.978]`、`[0.435, 1.824]`、
`[-0.434, 1.466]`。e420 只有两个 cluster，直接比较在 split 1、3、4 跨方向，
control 在 split 3、4 跨方向。由于 cluster 数仍很少，这不是正式 actor-level
Validation，也不改变当前 `PROMISING` diagnostic 状态；正式 Gate 仍需预注册的
多 actor、success/drop 覆盖和 matched control。
该审计按 source-run 等权形成 cluster mean，而 primary fit 按 episode 等权报告
MAE；两者是不同 estimand，cluster 区间只用于揭示 run 内相关性带来的保守不确定性。

## History preceding-action contract repair

对照实验进一步显示，原始 assembler 在每个 history 窗口都把最老的
`previous_action` 清零；四个 e260 raw run 中约 `99.8%` 的窗口该 factual action
实际非零。这违反了 `H^-={s_{t-L+1:t},a_{t-L:t-1}}` 的合同，因此修复为保留
collector 的 preceding action，并重新组装、audit 和 augmentation。旧数据没有被
覆盖；修复后的 dataset 仍为 58,111 windows、112 episodes，audit 通过，除
`history_previous_action` 外其余 row keys 与旧数据完全一致。

在固定 split、固定 model seed 的 corrected h16 I+ fit 中，`V_HEI` 相对 `V_H`
为 `+37.8%`、`+20.6%`、`-28.2%`、`+21.2%`、`+6.2%`，episode bootstrap CI
分别为 `[1.326,15.160]`、`[0.467,2.455]`、`[-31.178,3.171]`、
`[0.095,3.338]`、`[-1.027,1.538]`。matched future-action control
`V_HFEI`/`V_HF` 为 `+38.4%`、`+18.0%`、`-20.7%`、`+18.1%`、`+14.3%`，
CI 为 `[0.182,19.629]`、`[0.247,2.203]`、`[-26.570,3.128]`、
`[0.076,3.026]`、`[0.172,1.495]`。

修复没有消除 split 3 的反向结果。独立 model-seed 对照也确认 split 1 始终正向、
split 3 始终反向；split 3 的高 return success/drop episode `(2,18640000018)`
仍是主要负向影响，说明近期瓶颈是 outcome coverage 和 episode influence，而不是
初始化随机性。该修复后的 I+ 仍只能标为 `PROMISING` diagnostic，不能升级为
Gate closeout 或 `SUPPORTED`，下一步应增加独立 success/drop 覆盖并预注册
actor/episode cluster Validation。

## Corrected cross-actor replication

因为 history preceding-action 修复也影响旧 e420 diagnostic，使用相同的 s286/s287
raw shards 重新组装，得到 44 episodes、22,644 windows；base dataset audit 通过，
[`audit_gate1_interaction_augmentation.py`](../../scripts/audit_gate1_interaction_augmentation.py)
验证 interaction dim 119、首槽增量为零、target/episode/row-key 不变。
corrected h16 direct I+ 的五个 split 为 `+3.0%`、`+7.4%`、`-19.8%`、`+3.1%`、
`-22.8%`，episode bootstrap CI 只有 split 3 排除零；future-action control
为 `+3.2%`、`+14.0%`、`-16.9%`、`+2.9%`、`-27.6%`，也没有稳定跨 split 的
方向。source-run cluster bootstrap（2 clusters）同样显示 direct split 2、4
跨方向，control split 3、4、5 跨方向。修复后 e420 仍不能复现 e260 的 I+ 方向，
因此当前证据继续限定为局部 `PROMISING` diagnostic，不支持 Gate closeout 或
`SUPPORTED`。

## Corrected e260 outcome-coverage extension

在 history contract 修复后，从同一 pinned e260 checkpoint 按相同 pre-step collector
并行补采四个 run（namespaces 1866--1869），新增 56 episodes、60,756 rows；其中
1 个 `stable_success`、0 个 `drop_after_success`。与前四个 corrected run 合并后，
得到 224 episodes、116,067 h16 windows；base dataset 与独立 interaction
augmentation audit 均通过。

五个 fixed split 的 direct I+ `V_HEI`/`V_H` 改善为
`+12.6%`、`+5.2%`、`+18.3%`、`+26.9%`、`+14.9%`，episode bootstrap CI 有
4/5 排除零。matched future-action control `V_HFEI`/`V_HF` 为
`+15.9%`、`+18.9%`、`+25.3%`、`+25.3%`、`+40.7%`，5/5 CI 排除零。用 8 个
source-run 做 cluster bootstrap 后，direct 只有 split 1 区间跨零，control 五个
区间均为正。
作为严格 action-inclusive 消融，`V_HAEI` 相对 `V_H` 为
`+13.7%`、`+9.8%`、`-3.2%`、`+24.8%`、`+18.2%`；split 3 仍跨零，说明
future-action control 的正向结果不能替代对 action/outcome 敏感性的正式检验。

这表明增加独立普通失败和一个 stable-success 后，旧 split 3 的极端反向影响不再
主导，I+ 的方向性显著更一致；同时 corrected e420 actor 仍未复现该方向。因此当前
状态是更强的 `PROMISING` coverage probe，而不是 `SUPPORTED`：扩展数据是在探索阶段
决定的，success/drop 仍不平衡，正式 Gate 需要预注册 actor/outcome cluster、固定
matched control 和独立验证集；在此之前不启动 Cm。

## Corrected horizon follow-up

在同一 8-run corrected 数据上，h3/h5/h10/h32 的单 seed screening direct I+
分别为 `+11.7%`、`+15.0%`、`+5.1%`、`+16.4%`；h10 的 action-inclusive
`V_HAEI` 为 `+21.6%`。按 decision memo 追加 h3、h5、h32 direct 多 seed：h3
为 `+11.7%,-5.2%,+9.2%,+7.9%`，h5 为 `+15.0%,+10.6%,-6.6%,+19.0%`，
均显示 seed 敏感；h32 五个 seed 为 `+16.4%`、`+22.6%`、`+35.3%`、`+47.2%`、
`+33.7%`，每个 episode CI 和 8-source-run cluster CI 均排除零。

h32 matched future-action control 五个 seed 为 `+34.8%`、`+25.4%`、`+38.7%`、
`+45.2%`、`+36.8%`，episode 与 cluster CI 均为正。这个结果把后续正式
Validation 的候选 horizon 固定为 h32，但仍不等同于正式 Gate：数据扩展和 horizon
选择发生在探索阶段，尚需独立、预注册的 actor/outcome cluster 验证；在此之前不启动 Cm。
