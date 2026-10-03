# Gate 1 e420 coverage follow-up

**当前需要决定的问题。** e260 fresh 数据的 h16 `V_HEI` 方向在五个 split
都为正，但只有 1/56 episodes 是 stable-success，future-action control 也
对 split 敏感；是否需要更有 outcome variation 的独立 Probe。

**关键证据。** e260 合并数据有约 30k windows、56 episodes，其中 1 个 episode
同时满足 stable-success 和 drop-after-success；其余 episode 没有这两类 outcome。
h16 episode-balanced improvement 为 `9.8%`–`29.8%`，但 CI 并非都
排除零。仓库已有文档登记的 e420 `plain_off` s286/s287 checkpoint，历史
diagnostic collection 显示 success/drop 非零；它们不是 e260 pinned actor，
不能混入主结论。

**root 选择的行动及理由。** 用修正后的 pre-step collector，分别从 s286 和
s287 e420 checkpoint 采集独立 diagnostic runs；保留 source/checkpoint namespace，
单独 assemble、audit、horizon sweep 和 h16 future-action control。这样只回答
“修正后的表示在较有 outcome variation 的 actor 分布上是否仍有信号”，不改变
当前 Gate 的 claim，也不启动 Cm 训练。

**成本、成功/失败后的下一步和停止条件。** 两张空闲 GPU、每个约 14k rows、
约 15 分钟 wall budget。若 e420 也只在少数 split 有方向性改善，则停止扩展
collector，保留 h16 为 `PROMISING` Probe；若独立 split 与 e260 一致且 control
后仍有增益，才准备正式 Validation 设计。任何结果都不把 diagnostic actor
与 pinned e260 结果合并成单一科学结论。

**外部授权边界。** 只读取两个 checkpoint 和 motion 数据；不覆盖 checkpoint，
不修改 baseline worktree，不启动在线 Cm/policy training。用户已授权并行使用
空闲 GPU。

**结果。** 两个 run 合计 44 episodes、23.7k transitions；s286 的同一个 episode
同时满足 stable-success 与 drop-after-success，s287 没有这两类 outcome。五个
h16 split 的相对变化为 `+2.9%`、`+2.8%`、
`-18.2%`、`+24.4%`、`+3.7%`，只有一个 CI 排除零。该 actor 未复现 e260 的
稳定 h16 方向，因此停止该 diagnostic 分支，不把它与 e260 主 Probe 合并。
