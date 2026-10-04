# Gate 1 data-coverage diagnostic

**当前需要决定的问题。** Gate 1 的四次 grouped split 没有达到预设
`>=10%` MAE 改善且 bootstrap CI 排除零的门槛；是否应把它直接解释为
`(E,I) -> G` 无效，还是先检查 P0 actor 是否提供了足够的 task-value 变化。

**关键证据。** 大样本数据有 21,004 个窗口和 41 个 episode，但稳定成功数为
0，`stable_success` 和 `drop_after_success` 辅助标签均为常数。四个 split 的
`V_HEI` 相对 `V_H` 变化为 `-41.6% / +9.5% / -4.9% / -4.0%`，方向随
episode split 改变。Gate 1 数据契约、episode 隔离和模型拟合本身均通过。

**root 选择的行动及理由。** 保留当前 Gate 1 为“此 actor/data/bridge recipe
UNPROMISING”，不训练 Cm、不降低门槛；另外用已有自训练 `plain_off` e420
checkpoint 做一次独立小 smoke，只回答它能否产生非恒定 outcome 标签。该
checkpoint 和数据分布不回写当前 Gate 1，也不把 smoke 当科学结果。

**成本、成功/失败后的下一步和停止条件。** 先使用一张空闲 GPU、6 环境、
约 1200 条 transition 做 smoke；若出现历史保持，再以同一协议扩充到约
12k rows，并用并行 GPU 做四个固定 split。若仍没有稳定且可复制的 Gate 1
改善，关闭这条 data-coverage 分支并把 Gate 1 结果作为当前路线边界；只有
出现达到预设门槛的独立结果时，才新建后续 Probe。

**结果。** e420 smoke 有 2/6 历史五步保持但 0/6 stable success；扩充采集为
14,081 rows、26 episodes、1 stable success、1 drop-after-success。四个 split
的 `V_HEI` 相对 `V_H` MAE 变化为 `-25.5% / -6.9% / -11.3% / +1.6%`，均未
通过 10% 加 bootstrap-CI 门槛。因此关闭该 data-coverage 分支，冻结当前
Gate 1 为本 actor/data/bridge recipe 的 `UNPROMISING`。

**外部授权边界。** 不修改外部 baseline worktree，不覆盖 checkpoint，不启动
在线 Cm 或 policy training；当前用户已授权继续推进和并行使用空闲 GPU。
