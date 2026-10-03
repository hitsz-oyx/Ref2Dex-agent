# Gate 1 interaction-delta representation probe

**当前需要决定的问题。** `I` 的现有 contemporaneous object-frame pose/force/mask
序列已经用 GRU 编码，但 ref1 指出它没有显式表达 slip、force change 和
contact establish/release。加入这些可从现有 assembled tensor 重建的变化量，
是否能改善 h16 的 held-out return bridge。

**关键证据。** 四个 e260 run 合并为 112 episodes 后，基础 `V_HEI`/`V_H` 在五个
split 上为 `+12.1%`、`+15.0%`、`-25.8%`、`+29.8%`、`+5.4%`，只有 2/5 CI
排除零；同一数据的 future-action control 也只有 1/5 CI 排除零。当前不能把
不稳定归因于某一个缺失 interaction statistic。

**root 选择的行动及理由。** 离线复制 112-episode h16 dataset，保留原 80 维
interaction，并追加相对加速度、hand/object force increment、hand/object
contact increment；第一 future slot 显式置零。target、episode key、split、
GRU bridge 和训练 seed 全部不变。这个 A/B 只测试表示信息，不改变 Gate claim。

**预计成本、成功/失败后的下一步和停止条件。** 只需 CPU tensor transform、一次
dataset audit 和最多五个 GPU fit。若 `I+` 在同一五个 split 上没有一致的增益，
停止继续堆叠 interaction feature，转入 actor-level Validation 设计；若出现
稳定方向，也只能记为 `PROMISING` Probe，不启动 Cm，直到独立 actor/outcome
覆盖和预注册 cluster bootstrap 完成。

**外部授权边界。** 不重新采集、不修改 raw shard、checkpoint 或 baseline，
不启动在线 policy/Cm training。
