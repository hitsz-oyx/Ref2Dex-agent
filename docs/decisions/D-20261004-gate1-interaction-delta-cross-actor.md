# Gate 1 interaction-delta cross-actor probe

**当前需要决定的问题。** I+ 在 e260 112-episode exploratory fit 的 h16/h32
方向明显好于基础 I，但 split 3 被一个 success/drop episode 主导；是否能在已
采集的独立 e420 actor 上复现。

**关键证据。** e260 I+ h16 的五个直接点估计为 `+29.1%`、`+20.2%`、`-18.5%`,
`+25.8%`、`+18.3%`；future-action control 的点估计为 `+34.6%`、`+13.9%`,
`-6.4%`、`+12.9%`、`+10.9%`，control CI 仍全部跨零。I+ 的 h3/5/10/32 sweep
没有同样一致，因此需要独立 actor 检查而不是继续调 feature。

**root 选择的行动及理由。** 对现有 e420 s286/s287 pre-step raw shards 做同样
的离线 I+ augmentation，固定 h16、composite episode split、GRU bridge、五个
seed；不重新采集、不改变 e420 与 e260 的 namespace。结果只回答表示改进是否跨
actor 复现，不替代正式 Validation。

**预计成本、成功/失败后的下一步和停止条件。** 一个 dataset transform、audit、
五个 GPU fit，约数分钟。如果 e420 I+ 仍无稳定方向，停止继续堆 interaction
feature，转向 actor/outcome cluster Validation；如果复现，只记录为 `PROMISING`
表示改进并设计正式 matched validation，仍不启动 Cm。

**外部授权边界。** 只读取已存在 e420 shards；不修改 raw 数据、checkpoint 或
baseline，不启动在线 policy/Cm training。
