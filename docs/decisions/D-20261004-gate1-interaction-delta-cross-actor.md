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

**结果。** e420 I+ audit 通过（44 episodes、22,644 windows、interaction dim
119）。五个 h16 `V_HEI`/`V_H` 点估计为 `+19.4%`、`+3.5%`、`-8.5%`、`+23.6%`、
`+4.0%`，其中 2/5 CI 排除零；matched `V_HFEI`/`V_HF` 为 `+14.4%`、`+2.4%`,
`-6.6%`、`+21.2%`、`+4.8%`，其中 2/5 CI 排除零。I+ 在独立 actor 上保留
方向性，但没有稳定 Gate 级复现，因此停止继续堆叠该 feature，转入更严格的
actor/outcome cluster Validation 设计，不启动 Cm。

**聚类不确定性审计。** 对每个 h16 fit report 的 held-out episode 表按
`source_run` 聚类重采样（e260 4 clusters，e420 2 clusters，10,000 次），不改变
原 episode-balanced MAE。e260 直接 I+ 的五个 split 的 cluster-bootstrap
delta CI 分别为 `[0.935, 7.905]`、`[0.888, 2.039]`、`[-16.167, 2.018]`、
`[1.277, 3.803]`、`[0.646, 1.330]` MAE；matched future-action control
分别为 `[-1.440, 13.241]`、`[0.359, 1.835]`、`[-8.423, 2.978]`、
`[0.435, 1.824]`、`[-0.434, 1.466]`。e420 只有两个 source-run cluster，
直接 I+ 的 split 1、3、4 区间跨越方向，control 的 split 3、4 也跨越方向。
这是保守的诊断审计，不是正式 actor-level Validation；它进一步说明需要预注册
多个独立 actor 和 success/drop outcome 后才能判断 Gate。
