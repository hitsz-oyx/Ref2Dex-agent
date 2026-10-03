# Gate 1 v2 contract repair

> **Erratum (2026-10-04).** Any future-action control numbers later copied into this
> historical record used the pre-repair fast assembler and are `INVALID_IMPLEMENTATION`.
> Direct H/E/I results are unaffected; corrected controls are recorded in
> [D-20261004-gate1-future-action-index-repair](D-20261004-gate1-future-action-index-repair.md).

**当前需要决定的问题。** 旧 Gate 1 v2 结果能否作为历史窗口和未来后果的证据，还是必须先审计采集时序与坐标契约。

**关键证据。** 复核发现 collector 在 `env_step` 后写入物理张量，导致 row `t` 的物理量是 `x_{t+1}`，而 return 仍包含 `r_t`。旧 v2 因此漏掉了 `a_t` 的即时后果。复核还发现相对四元数存在 q/−q 跳变，force 分量仍是 world frame，以及多 run split 只使用 episode 编号。

**root 选择的行动及理由。** 修正 collector 为 pre-step capture；assembler 对旧 shard 做显式 legacy shift；对物理四元数做时间连续化；将 hand/object force 旋转到 contemporaneous object frame；以 `(source_run, episode_id)` 分组，并让各 ablation 使用相同 seed。重新采集一个小型 fresh Probe，不把旧 h32 结果用于科学结论。

**预计成本、成功/失败后的下一步和停止条件。** 使用一张 GPU、6 环境、约 6.6k transitions，随后在 5 个 horizon 上做固定 30 epoch 离线 fit。fresh Probe 若没有稳定的 held-out improvement，则维持 Gate 1 `UNCLEAR/UNPROMISING`，不启动 Cm 训练；只有独立 episode 规模和多 split 复现后才可升级为 Validation。

**结果。** fresh pre-step 数据通过 state/next-state 对齐审计，包含 12 个 episode。最终同 seed fit 的 `V_HEI` 相对 `V_H` MAE 变化为：h3 `-16.2%`（CI [-5.20, 3.25]），h5 `+5.4%`（CI [-2.07, 2.79]），h10 `-18.9%`（CI [-2.56, 0.50]），h16 `-81.7%`（CI [-4.59, -3.65]），h32 `-14.8%`（CI [-2.49, 0.57]）。测试集只有 2 个 episode，因此这些数字只是 Probe 证据，不能作为 Gate 级结论；当前不支持继续 Cm 路线。

**外部授权边界。** 只读取 baseline checkpoint 和 motion 数据；没有修改外部 worktree、覆盖 checkpoint 或启动在线 Cm/policy training。

**后续 Probe 记录。** 随后两组 fresh run 合计 56 episodes、约 30k 可用窗口；以
episode-balanced MAE 计算，h16 在五个 split 都保持正向，改善约 `9.8%`–`29.8%`，
但只有部分 CI 排除零。
早期诊断输出（尚未统一 episode-balanced 主指标）中，加入未来 on-policy action
control 后五个 split 的 E/I 相对 `V_HF` 变化为
`-9.2%`、`+10.8%`、`+27.5%`、`+15.9%`、`+23.3%`。因此 h16 方向值得继续
验证，但仍不能升级为 Gate closeout 或 Cm 训练授权。

随后用修正后的 episode-balanced 统计重跑 matched control，结果为
`-9.0%`、`+10.9%`、`+28.9%`、`+19.7%`、`+23.2%`；只有 split 2、3 的
episode bootstrap CI 排除零。held-out episode 审计还显示 split 1、2 的总增益
分别约 94%、96% 来自单个 episode，移除该 episode 后平均增益仅约 0.11、0.20
MAE。因此 root 维持“不启动 Cm、补充按 episode 类型和 success/drop 覆盖的
Probe”这一行动，h16 仍是 `PROMISING` 诊断方向而不是 Gate closeout。
