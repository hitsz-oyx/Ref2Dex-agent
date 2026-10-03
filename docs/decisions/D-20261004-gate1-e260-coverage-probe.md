# Gate 1 e260 outcome-coverage probe

**当前需要决定的问题。** e260 h16 的方向性增益在五个 split 中为正，但前两个
split 的大部分收益来自同一个同时满足 stable-success 与 drop-after-success 的
episode；是否需要独立 episode 才能判断它是否可复现。

**关键证据。** 56 个 e260 episode 中只有 1 个 episode 有这两类 outcome。去掉该
episode 的 held-out 敏感性分析使前两个 split 的平均增益降到约 0.11、0.20 MAE；
e420 diagnostic actor 也没有稳定复现 h16 方向。现有证据不足以启动 Cm，但仍值得
用同一 pinned e260 actor 获取更多独立 episode。

**root 选择的行动及理由。** 从相同 e260 checkpoint、相同 pre-step collector
和相同 6-environment 配置启动两个新的独立 run（seed namespace 1864、1865），
分别采集约 14k transitions；独立 assemble h16，沿用 composite episode split、
episode-balanced MAE、future-action control 和逐 episode influence audit。两组
数据与现有 e260 数据保持分离，避免事后改变原有 split。

**预计成本、成功/失败后的下一步和停止条件。** 两张空闲 GPU、每组最多 900 秒，
只做离线 fit，不训练 policy/Cm。如果新增 run 仍没有 outcome variation，或 h16
增益仍由单个 episode 主导，则停止扩展该 actor，保留 `PROMISING` 诊断状态并把
actor/episode cluster uncertainty 记入后续 Validation 设计；若有多个独立
success/drop episode 且方向保持，才考虑正式 Validation 方案。

**外部授权边界。** 只读取 pinned checkpoint 和 motion 数据；不覆盖已有 shard
或 checkpoint，不修改 baseline worktree，不启动在线 Cm/policy training。
