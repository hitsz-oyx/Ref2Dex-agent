# V1.28：PPO 稳定性 smoke 与批量严格策略评估

- timestamp: `2026-09-23`
- branch: `agent/ppo-stability-v128`
- run_status: `COMPLETED`
- official actor used for training/evaluation: `no`
- conclusion: `REFUTED`（两种 PPO 稳定化没有保持 epoch-140 性能；当前 CmLite 候选器没有跨 seed 净增益）

## PPO 稳定性

提交 `63e0bb2` 为 CmLite scratch-policy launcher 增加可审计的 `lr_schedule`、
`schedule_type`、`kl_threshold`、`mini_epochs` 和 PPO clip 覆盖入口。所有恢复都固定来自本项目
epoch 140 checkpoint，SHA256 为
`5d1f50a21409df5a09f0a471d115d2f06eedce6bc53855a81acf3cc268e6e1a7`。

| run | 变更 | epoch 142，seed 49 严格成功 |
| --- | --- | ---: |
| parent epoch 140 | 无 | 35/64（54.69%） |
| `agent_v128_norm_cmlite_adaptivekl_smoke_s45_e142` | adaptive LR，target KL 0.016 | 25/64（39.06%） |
| `agent_v128_norm_cmlite_miniepoch2_smoke_s45_e142` | constant 2e-5，mini-epochs 6→2 | 26/64（40.62%） |

adaptive run 第一轮 KL 约 `0.0837`、LR 降至约 `3.95e-6`，第二轮 KL 估计为负并把
LR 升回 `2e-5`；说明入口生效，但旧 PPO 的 sample KL 噪声不适合直接驱动该 scheduler。
仅减少重复更新仍在两轮内退化，不能把问题归结为 mini-epochs 数量。

## DAgger/BC 批量严格评估

[eval_cmlite_bc_policy.py](../../tools/eval_cmlite_bc_policy.py) 现支持每个并行环境只记录第一个
完整 episode，并沿用 `object dz >= 0.03 m` 且手物接触连续至少 5 步的严格成功定义。被评估
BC checkpoint 明确记录 `official_policy_checkpoint: null`，但它是参考动作 DAgger 学生，不是
PPO 主线策略。

| eval seed | Cm-off | Cm-on（适配 CmLite） | 配对修复/破坏 |
| --- | ---: | ---: | ---: |
| 49 | 39/64（60.94%） | 42/64（65.62%） | 7 / 4 |
| 50 | 40/64（62.50%） | 35/64（54.69%） | 1 / 6 |
| pooled | **79/128（61.72%）** | **77/128（60.16%）** | **8 / 10** |

因此 seed 49 的局部正结果不能外推；当前 CmLite 五候选在线替换器跨 seed 净效应为负。
批量 reference teacher 的 8-env smoke 也仅为 4/8，说明把单环境参考教师扩充为更多训练样本
本身不足以达到 90%。

## 产物入口

- PPO smoke：`outputs/Dexplore/agent_v128_norm_cmlite_adaptivekl_smoke_s45_e142`、
  `outputs/Dexplore/agent_v128_norm_cmlite_miniepoch2_smoke_s45_e142`。
- BC 配对：`outputs/CmResidual/agent_v128_bc_batch_{off,on}_s{49,50}_n64`。
- reference teacher smoke：`outputs/CmResidual/agent_v128_reference_teacher_s49_n8`。
- 第一次 BC smoke 因错误 cwd 在配置加载前失败：
  `outputs/CmResidual/agent_v128_bc_batch_off_s49_n8`；成功重试为 `_r2`。

下一轮不应继续阈值 sweep 或无保护地从 epoch 140 PPO 更新。更有价值的是验证互补策略路由、
策略保持约束，或使用已有 V1.21c 反事实排名基础设施学习真实物理候选排序。
