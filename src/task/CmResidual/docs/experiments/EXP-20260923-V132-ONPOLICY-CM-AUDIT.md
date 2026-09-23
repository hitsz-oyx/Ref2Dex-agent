# V1.32: s3 微调策略分布上的 CmLite 一步预测审计

- experiment_id: `EXP-20260923-V132-ONPOLICY-CM-AUDIT`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 问题与固定方案

V1.30 中相同源策略与训练预算下，CmLite 奖励在 s3 seeds 71–73
只得到 30/192，而 Cm-off 为 83/192。V1.31 排除了 s1 源 run
的另外八枚 checkpoint 直接迁移优于 e140 的假设。现在区分两个
可能原因：模型在高成功率策略的数据分布上失准，或模型仍能预测
但其目标进展奖励与最终抓取指标不对齐。

冻结 V1.30 的 Cm-on/Cm-off e160 checkpoint、相同 s3 重建输入、
未见 seed 74、64 env、完整轨迹且提前终止关闭。每臂收集评估时
真实 `(q, action, object_state, next_object_state, contact, done)`；
仅取每个环境首次 episode 中非终止转移，排除 reset 污染与重复
episode。两臂都只审计**同一** V1.29 相对手腕 CmLite checkpoint
`b7aa7630e31c820802cb95d81c490d4f27be8e74c1c9e1e400b20fcd849a9a38`。
模型参数与策略均不更新。

预注册指标：运动转移（真实平移 >0.5 mm）的一步 EPE 对零位移
基线的比值；真实手物接触的 precision/recall/F1；固定状态、
seed 129 打乱动作后 EPE 的变化与预测位移变化。若 Cm-off 高成功
分布仍达到运动 EPE 至少比零位移低 20%、接触 precision ≥0.5，
且打乱动作使运动 EPE 上升至少 10%，则认为该分布上**仍有
预测信号**，奖励设计更值得优先检查；否则优先处理分布外校准。
这一离线审计不能证明在线 Cm 有收益。

GPU 5/6 各一臂，同时最多两卡；每臂一轮评估及小于 2 GB 的转移
tensor，预计几分钟。发现占卡、源 SHA/输入漂移或转移 schema
不符就停。结果、checkpoint SHA、代码提交写入独立 manifest/JSON。
