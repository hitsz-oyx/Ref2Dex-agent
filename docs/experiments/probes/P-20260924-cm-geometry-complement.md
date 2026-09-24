# P-20260924-cm-geometry-complement

date: 2026-09-24
branch: agent/cm-contact-aware-effect
classification: Decision

## Question

六区域几何 Cm 在短期接触预测上略弱于 raw state+action MLP。
它是否携带互补信息，能改善动作接触效应排序？

## Hypothesis and decision

固定已训练 seed151/152 的 geometric、raw checkpoint，不再更新。
事前规定 ensemble 为两个模型的 `contact_fraction` 概率算术
平均，权重各 .5，不使用新测试标签调权重或阈值。新 seed155/156
的 ±0.1 随机接触干预五步随访仅测试。

若 ensemble 在两 seed 的接触预测 RMSE 均低于 raw，且合并
随机处理效应排序高−低四分位差比 raw 高 ≥3pp，paired
environment-cluster bootstrap 的差值 95% CI 下界 >0，
才认为几何有足够互补信号可继续用于策略；否则停止当前
六区域几何架构微调，转向训练期 Cm 用法或重新定义表示。

## Minimal protocol

相同自训练 e260 actor、s3 Inspire 轨迹，64 env，global step
50..150、stride10，在实际接触时随机 ±0.1。存处理前状态、
动作和五步接触；分组依据固定模型在处理前对 ± 两候选
预测的接触差，实际组内效果由随机分配估计。

## Budget and stop

每个采集 1 张空闲 GPU、<30min；离线 CPU 2 threads、<30min；
新增产物 <200MB。输入/模型 SHA 漂移、非有限值、GPU 冲突
或预算超限停止。

## Result

Status: UNPROMISING（当前固定六区域几何的互补性门未过）

新 seed155/156 的随机随访分别 616/646 条接触干预，均
`COMPLETED`；冻结 checkpoint 未重训。接触比例预测 RMSE：

| seed | geometric | raw | 固定 .5/.5 ensemble |
| --- | ---: | ---: | ---: |
| 155 | .1367 | .1337 | .1178 |
| 156 | .1213 | .1022 | .1035 |

合并未见 seed 的随机处理效应排序高−低四分位差：
geometric +26.58pp、raw +27.85pp、ensemble +28.72pp。
ensemble−raw 仅 +0.87pp；同一批 environment 聚类配对
bootstrap 95% CI [−2.26,+3.98]pp。未达预设两 seed
RMSE 都降低、排序差 ≥3pp 且 CI 下界 >0 的联合门槛。

产物：`outputs/CmResidual/agent_randomized_followup_s{155,156}_d01_h5_n64/`
的随机化数据与 `outputs/CmResidual/agent_cm_geometry_complement_s155156/`
的 manifest/report。所有数据与冻结模型 SHA 记录于 manifest。

## Decision update

停止对当前六区域几何架构做局部小修。这个负结果不否定
动作条件的短期接触模型；raw 模型仍显示明显随机效应排序。
下一步可将 raw + 五步接触/物体效应视为重新设计的 Cm，
做一次成本受控、预先固定规则的在线动作选择 Probe；若
仍无 policy utility，再转向训练期表示/辅助目标，不继续
搜索局部抬腕阈值。
