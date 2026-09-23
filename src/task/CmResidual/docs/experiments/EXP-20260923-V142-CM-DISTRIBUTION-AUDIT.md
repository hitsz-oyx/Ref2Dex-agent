# V1.42: CmLite 在后期 s3 专家上的分布审计

- experiment_id: `EXP-20260923-V142-CM-DISTRIBUTION-AUDIT`
- branch: `agent/v142-cm-distribution-audit`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 冻结诊断与门槛

V1.41 的 V1.37 CmLite 抓握门控在四个 heldout seed
相对原路由仅净增2/256，seed93还损失5/64；必须
先判明一步模型在路由的**更晚自训练专家**上是否仍然
准确，不能直接归因于打分目标。冻结模型 SHA256
`396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`，
以及自训练 s3 `back240`、`back260` 专家 SHA256
分别为 `a88924a590966c964b029e067985fea41465f230d459454b8048899cbe8670c2`
和 `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`。
`source` e180 已在 V1.37 seed79 做过独立离线验证；
此处只补两枚后期专家。

对每专家新 seeds95、96 各采64环境首个完整 episode
非终止转移，关闭提前终止、保持原评估动作且不训练
策略。用冻结 CmLite 计算一步物体位移 EPE、零位移
基线、固定 seed 打乱动作后的 EPE 和接触 precision。
四份均需满足：真实动作 EPE 至少比零位移低20%，
打乱动作 EPE 至少比真实动作高10%，接触 precision
≥0.5，才称为通过当前专家分布准确性门槛。任一
失败，则 V1.37 CmLite 对当前动作分布不可靠，下一
轮可预注册新数据的因果化训练；若四份均通过，则
优先检验一步目标函数与长期抓取是否错位，不再仅
追加同类数据。

该离线诊断不能证明 Cm 改善抓取，也不选择新模型。
每次最多GPU5/6两卡，评测输出远低于300GB；输入
轨迹、checkpoint、运行代码 commit 和转移 SHA
写入 run manifest 与审计 JSON。
