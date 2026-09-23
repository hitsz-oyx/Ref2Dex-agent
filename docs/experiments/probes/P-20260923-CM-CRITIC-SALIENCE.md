# Probe: Cm 一步效应只引导 PPO critic 的状态显著性

probe_id: `P-20260923-CM-CRITIC-SALIENCE`

date: 2026-09-23

branch: `agent/cm-critic-salience`

status: COMPLETED

code commit: `b8b5cfcfe623b014625cfc97a57e3c3db7935e71`

## Question and decision

当前 actor 样本加权族跨 seed 不稳定。测试不同机制：冻结 CmLite
预测**策略均值动作**的一步接触概率×绝对竖直效应，将其用于 PPO
critic loss 的样本显著性；actor loss、advantage、真实奖励和推理策略
都不使用 Cm。若该方法在困难训练 seed 的最小 Probe 有明显增益，
再设计对 Cm-off 和置乱显著性的正式验证；否则放弃这个 critic 接法，
考虑状态表征/辅助目标。

## Minimal protocol

源为相同自训练 s3 e260 checkpoint，训练 seed74 至 e300，CmLite
V1.37、corrected_r2 运动输入、PPO/奖励/课程与当前 Cm-off 一致。
每 rollout 状态用确定性的 actor mean action 计算 raw weight
`1 + p_contact * clamp(|pred_dz|/3mm,0,1)`，只乘到 critic 的逐样本
平方误差；每个 minibatch 除以平均 raw weight，使平均 critic loss
系数与 off 相同。模型冻结，不在推理期加载。actor PPO mask 与 loss
完全不改。严格评估 seed133/134，每次 64env 完整首回合。

已知同训练 seed74 Cm-off 为 32/64、36/64。预定推进门：新方法
两次合计至少 79/128（比 off 68/128 高≥11/128，即≥8pp），
且每个评估 seed 都高于 off。若没过，不扩大同接法。
此困难 seed 与评估对照已知，只是 Probe，不产生论文 claim。

## Budget and stop

先做单元测试、dry-run 和 1-epoch 接线 smoke，再固定代码 commit。
正式 Probe 1 GPU、预计 <30min、新产物 <2GB。源/Cm/input SHA
和唯一 run_id 固定；GPU 占用、非有限梯度、合同失败或输入漂移即停止。

## Result

run_status: `COMPLETED`（1-epoch smoke、训练 1/1、评估 2/2）

conclusion: `PROMISING`（仅此已知困难 seed 的 Probe）

单元测试 11 passed；e261 接线 smoke 完成、checkpoint 存在，Cm
critic 权重进入日志，无非有限训练。固定 e300 训练正常完成。

| 评估 seed | critic-salience | Cm-off | joint actor 权重参考 |
| ---: | ---: | ---: | ---: |
| 133 | 52/64 | 32/64 | 43/64 |
| 134 | 46/64 | 36/64 | 41/64 |
| 合计 | **98/128** | **68/128** | **84/128** |

critic−off = +30/128（+23.44pp），两 seed 均正，过预定继续门。
这不能证明 Cm 对 critic 有独立作用：当前无同结构但置乱/常数显著性
控制，且困难训练 seed、评估对照已知。下一步先用另一个训练 seed73
做最小复制；若仍过门，再设计新 seed 的正式 Cm-on/off/placebo 验证。

训练 manifest：`outputs/Dexplore/cm_critic_salience_probe_t74/run_manifest.json`；
严格评估：该目录下 `eval_s133_e300_full_cmcritic/`、
`eval_s134_e300_full_cmcritic/`。接线 smoke：
`outputs/Dexplore/cm_critic_salience_smoke_t74_e261/run_manifest.json`。
