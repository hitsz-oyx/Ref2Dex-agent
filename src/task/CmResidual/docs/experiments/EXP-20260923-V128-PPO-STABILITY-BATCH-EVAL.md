# PPO 稳定化与 CmLite 在线候选替换

- experiment_id: `EXP-20260923-V128-PPO-STABILITY-BATCH-EVAL`
- run_status: `COMPLETED`
- conclusion: `REFUTED`

从自训练 CmLite epoch 140 恢复后，adaptive-KL 和 mini-epochs 2 都在两个 epoch 内把严格成功率
从 `54.69%` 降至约 `40%`。在无官方 actor 的 DAgger/BC 策略上，CmLite 候选替换两 seed
聚合为 Cm-off `79/128`、Cm-on `77/128`，配对修复/破坏为 `8/10`。这些证据反驳当前两种
PPO 稳定化和当前在线 selector 能推进 90% 稳定抓取的假设。
