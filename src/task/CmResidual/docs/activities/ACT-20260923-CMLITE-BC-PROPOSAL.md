# DAgger policy 上的 CmLite 局部动作选择验证

- timestamp: `2026-09-23`
- work_version: `V1.27`
- run_status: `COMPLETED`
- official_policy_checkpoint: `null`
- conclusion: `INCONCLUSIVE`（世界模型预测改善，在线候选动作尚未改善抓取）

## 目的

`Ref2Dex-review` 的两轮 DAgger 策略已经在 5909/5910/5929 三个 seed 抬升约 13.5 cm，但 Cm 还没有参与该策略。这里固定同一个本地训练 MLP，给 Cm-on 和 Cm-off 相同的五个局部动作候选，测试 CmLite 是否能在真实接触后选择更好的动作。

## 实现

- [dexplore_bc_policy.py](../../dexplore_bc_policy.py) 加入本地 BC/DAgger policy 的严格归一化加载。
- [cmlite_policy_select.py](../../cmlite_policy_select.py) 生成固定五候选；策略动作是候选 0，CmLite 只有在真实手物接触、预测接触可信且预测进展为正时才能覆盖候选 0。
- [eval_cmlite_bc_policy.py](../../tools/eval_cmlite_bc_policy.py) 在单环境 DExplore 物理 rollout 中记录动作、物体状态和 Cm 选择，并明确拒绝官方策略 checkpoint。

## 证据

先用 5910/5929 的 DAgger 策略真实转移训练 CmLite，再用未参与训练的 5909 评估：

| arm | Cm checkpoint | max lift | frames > 5 cm | proposal selection |
| --- | --- | ---: | ---: | ---: |
| Cm-off | DAgger policy | 13.71 cm | 21 | 0% |
| Cm-on，旧 CmLite，预测接触直接门控 | GRAB seed42/43 model | 0.50 cm | 0 | 99.5% |
| Cm-on，旧 CmLite，真实接触门控 | GRAB seed42/43 model | 3.93 cm | 0 | 18.8% |
| Cm-on，正进展门控 | GRAB seed42/43 model | 8.03 cm | 24 | 12.3% |
| Cm-on，DAgger 转移适配、稳定接触门控 | `outputs/CmLite/V1.27/train_dagger_s5910_5929_val_s5909/best.pt` | 12.99 cm | 21 | 4.4% |

适配模型在 5909 留出转移上的 moving EPE 为 `1.79 mm`，零位移基线为 `3.31 mm`；物理复评固定使用 GPU 5（Cm-off）和 GPU 6（Cm-on），并在导入 Isaac Gym 前锁定可见设备。单步世界模型本身学到了可验证的运动信号，但把预测直接用于在线动作替换仍有分布偏移和夹持破坏风险。该实验不支持 Cm 已提高 DAgger policy 的抓取成功率。

## 后续使用边界

该 selector 默认保守门控，适合继续做候选排序/蒸馏实验，不应覆盖已经验证的策略。Cm 的主线结论仍采用 V1.25/V1.26 的从零 PPO 配对结果；下一轮若要让 Cm 改善 DAgger，应先收集带候选动作的真实反事实转移，再训练排序目标，而不是继续调在线阈值。
