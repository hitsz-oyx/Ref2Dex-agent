# V1.51: Cm 一步效应预测参与 PPO actor 样本权重

- experiment_id: `EXP-20260923-V151-CM-PPO-ACTOR-WEIGHT`
- branch: `agent/v151-cm-ppo-weight`
- status: `PREREGISTERED`
- official actor checkpoint: **never**

## 问题与假设

V1.46 已否定当前 CmLite 的“预测一步目标进展作为奖励”接法；V1.49–50
发现原版 Cmv2 的名义 hand flow 与真实一步流不匹配，且冻结权重存在跨域偏差。
本轮不以这些结果声称 Cm 无用，也不再单独调无 Cm baseline。检验另一种明确
的 PPO 接法：冻结低延迟、在真实接触转移上优于零位移的 V1.37 CmLite，
只在训练时为**记录动作**预测接触概率 `p` 和世界系竖直物体位移 `dz`，
令 actor PPO surrogate 的样本权重为
`w=1+1.0*p*clip(abs(dz)/0.003,0,1)`，范围 `[1,2]`。
真实环境的 advantage 决定梯度方向；Cm 不能更改奖励、价值目标或推理动作。
Cm-off 为 `w=1`。预期该接法强调物手相互作用状态，却不直接奖励模型幻觉。
效应绝对值而非正向目标进展是有意的：负 advantage 仍会抑制不佳接触动作。

## 冻结对照与停止规则

两臂都从自己训练的 V1.39 s3 e260 actor 继续到固定 e300：源 checkpoint
SHA256 `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`，
s3 corrected_r2 manifest SHA256
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`。
训练 seed70、64env、horizon32、minibatch256、学习率1e-5、既有相同几何/持物/
抬升奖励和 reset 课程。唯一处理差异是 Cm-on 的 actor 权重。原计划有条件地
复用 V1.46 同源同预算 Cm-off e300，但检查到 V1.48 后 approach agent 源码变化；
虽然新增 grasp-link 路径系数为0，仍不把旧权重作为因果对照，改用当前同版本源码
从 e260 重新训练 V1.51 Cm-off 到 e300。这个改动在观察任何 heldout 结果前冻结。
CmLite checkpoint SHA256
`396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`。

先接线测试：静态测试、e262 两 epoch smoke、checkpoint 正常恢复、权重日志
有限且均值在 `(1,2)`，确认 zero-coef 精确退化到原 actor mask。工程门失败
先修复，不解释为科研负结果。正式到 e300，不依赖训练中途指标选 epoch。
全新 heldout seed119–123、每 seed 每臂两次64环境、严格完整首 episode、
关闭提前终止；不逐 env_id 配对。固定20次评估，不根据中途结果改系数。

预注册 Cm 增益门：总成功率相对 Cm-off ≥8pp；5个 seed 的两次均值
均高于 off；以 seed 为簇的10000次 bootstrap 两侧95% CI 下界>0。
稳定目标另需 Cm-on 十次运行各≥58/64。若未过门，明确否定**这个权重接法**，
而非整个 Cm 架构。即使过门，也需跨种子重复与替代/打乱模型权重消融，
才可将增益归因到学到的 Cm 表征而非只是重加权。

最多同时 GPU5/6 两张卡，先检查占用；单训练预计几分钟，评估约20次，
产物预算<20GB、总上限300GB。失去空闲 GPU、输入合同或权重有限性则停止。
