# P-20260924-contact-aware-cm

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

六区域几何 Cm 能否从执行前状态与候选动作同时预测一步物体
效应和随后五步的接触保持，并在未见随机化 seed 上识别动作
对接触的异质影响？这决定是否值得再做在线 Cm 候选评分。

## Hypothesis and decision

用 seed151/152 的随机 ±0.1 真实执行随访训练，seed153/154
只测试。几何 Cm、相同特征但动作手流置零的 state-only 对照、
小型 raw state+action MLP 共享批次与更新预算。

若几何 Cm 在两个未见 seed 上均能预测接触处理效应为负，且
预测接触效应最高−最低四分位的真实 RCT 接触效应差合并 ≥5pp、
environment-cluster 95% CI 不含 0，并且至少不劣于 raw MLP，
则设计一次安全的在线 matched Cm-on/off Probe；否则不做
接触排序在线搜索，转向训练期 representation/auxiliary route。
若处理效应几乎均匀，即使平均方向正确也不够形成 action ranking。

## Minimal protocol

每种模型三头：一步物体局部位移、五步物体局部位移、五步手物
接触比例。线上可用输入仅为当前 `q,dof_vel,object_state,action`
与固定运动学/执行模型；不使用未来 `next_q`。训练不接触测试
seed、测试标签不用于调结构或阈值。

比较标准预测误差与随机处理效应排序。分组由模型在处理前
为 ±0.1 两个候选的接触预测差决定，真实分组效果由随机分配
估计；这不是逐状态物理反事实。

## Budget and stop

采集各 1 张空闲 GPU、64 env、单 run <30min；训练/分析 CPU
2 threads、≤1000 更新、总 <60min，新增产物 <200MB。
输入漂移、非有限训练、接触标签损坏或预算超限停止。

## Result

Status: UNCLEAR（动作信息强，但几何 Cm 独立优势未成立；不升级在线）

固定 500 更新、CPU 2 threads，训练 seed151/152 的 1268 条处理，
未见 seed153/154 共 1272 条测试。三个模型均完成；几何 Cm
13,031 参数，state-only 同规模，raw state+action MLP 8,967 参数。

| seed | 模型 | 一步物体 EPE | 五步物体 EPE | 五步接触 RMSE |
| --- | --- | ---: | ---: | ---: |
| 153 | geometric | 6.31mm | 20.20mm | .1206 |
| 153 | state-only | 8.55mm | 21.69mm | .1318 |
| 153 | raw-action | 6.86mm | 18.50mm | .1149 |
| 154 | geometric | 5.91mm | 18.82mm | .1428 |
| 154 | state-only | 7.86mm | 20.50mm | .1581 |
| 154 | raw-action | 6.45mm | 17.46mm | .1306 |

几何模型优于同结构零手流，对接触维度不如 raw。用处理前
候选 ±0.1 的模型接触预测差分组，两个未见 seed 合并的真实
随机处理效应：几何 Cm 最高分组 −0.84pp，最低分组 −23.24pp，
高−低 +22.40pp，environment-ID 聚类 95% CI [+15.87,+29.29]pp；
raw 模型高−低 +25.11pp，CI [+18.90,+31.69]pp。
因此“效应异质性可预测”有 Probe 信号，但几何并未达到
“至少不劣于 raw”门槛；也未完成每个 seed 平均处理效应
预测方向的预注册检查，不能把部分通过写成整个门通过。

产物：`outputs/CmResidual/agent_contact_aware_cm_s151152_train_s153154_test/`
的 `run_manifest.json`、`report.json`、三模型 checkpoint。
测试输入 SHA 固定于 manifest；没有官方 actor 权重。

## Decision update

不进行此几何 Cm 的在线候选搜索，也不根据 seed153/154
调结构刷分。下一条实现路线应转向训练期 representation /
auxiliary objective 的最小 matched Probe；先评估是否有
足够低成本的 PPO 接点。raw 模型是必要对照；不能声称
当前几何 token 结构对策略不可替代。
