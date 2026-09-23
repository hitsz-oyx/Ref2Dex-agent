# V1.32: s3 微调策略分布上的 CmLite 一步预测审计

- experiment_id: `EXP-20260923-V132-ONPOLICY-CM-AUDIT`
- branch: `agent/multitrajectory-v129`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（当前 CmLite 在高抬升策略分布上的可靠运动预测假设）
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

## 冻结审计结果

seed 74 的 64 个完整 episode：Cm-on 严格抓取 11/64，Cm-off
26/64，继续重现 V1.30 的方向。每臂的转移 tensor 为 11.8 MB；
筛除重复 episode 和终止 reset 行后，每臂均剩 33,189 条首 episode
非终止转移。冻结相对手腕 CmLite 的指标：

| 策略数据 | 运动样本 | 真实动作 EPE / 零位移 | 打乱动作 EPE | 接触 precision / recall |
| --- | ---: | ---: | ---: | ---: |
| Cm-off e160 | 11,027 | 10.99 / 10.52 mm | 11.18 mm | 0.760 / 0.536 |
| Cm-on e160 | 7,498 | 9.30 / 8.65 mm | 9.57 mm | 0.616 / 0.738 |

两臂的运动 EPE 均**高于**零位移基线；打乱动作虽使 EPE 上升，
但 Cm-off 仅约 1.7%、Cm-on 约 3.0%，低于预注册的 10% 门槛。
因此模型接触分类仍有信号，但其运动预测没有通过高成功率策略
分布的门禁。V1.29 seed 69 上的 1.739 / 4.428 mm 预测增益不能
外推至当前 e160 高抬升分布。这支持“分布外预测失准可能导致
Cm 奖励误导”的机制解释，但不能单凭相关性证明其为唯一因果原因。

两臂 `eval_s74_e160_full/run_manifest.json` 记录了评估 checkpoint、
输入与转移 SHA；`cmlite_audit.json` 记录筛选数量、模型 SHA 和
完整离线指标。下一步优先用成功转移重新校准 Cm，再以未见 seed
做离线门禁；未通过之前不启动新的在线 Cm 奖励训练。
