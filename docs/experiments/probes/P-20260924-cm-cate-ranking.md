# P-20260924-cm-cate-ranking

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

几何 Cm 在未见随机干预状态中预测的“+z 比 -z 更有利”的差异，
能否排序真实随机化实验中的**处理效应异质性**？若只能预测
总体平均效应，接到策略上做候选动作评分的价值很弱。

## Hypothesis and decision

H1: 以固定训练好的几何 Cm，对 seed146 ±0.3 与 seed148 ±0.1
每个干预前状态分别预测 +δ/-δ 的物体 z 位移差。按预测差异的
最低/最高四分位分组后，真实随机分配的高组加减臂处理效应
比低组更大：大动作至少 +5mm，小动作至少 +2mm，且两个
seed 方向一致。按环境 ID 聚类 bootstrap，至少一个 seed 的
高低差 95% 区间下界 >0。与固定 raw state+action MLP 同法对照。

H1 通过：可进入少环境、低延迟的在线候选动作评分 Probe，
之后仍需要 matched Cm-on/off 策略结果。

H1 不通过：不投入当前 Cm 的在线动作评分；重新设计效应监督
或表示，而不是继续只优化平均 EPE。

## Minimal protocol

模型与输入 SHA 固定；不再训练/选择 checkpoint。两个未见 seed
各自只用干预前 `q,dof_vel,object_state,base_action` 算模型的
同状态**预测** +δ/-δ 差，再利用已随机执行的单臂结果估计
每个分组的真实平均处理效应。分组阈值由模型预测分位数决定，
不使用结果；真实高低效应按全局步分层，bootstrap 按环境 ID
聚类。不能把模型两次预测当作物理逐样本真值。

## Budget and stop

CPU 2 threads、GPU 0；wall <=20 min；输出 <10MB。
checkpoint/输入漂移、非有限预测、分组样本不足或预算超限停止。

## Result

Status: PROMISING

Evidence: fixed geometric Cm and raw MLP checkpoints; held-out randomized
seed146 ±0.3 and seed148 ±0.1, no retraining. `agent_cm_cate_ranking_s146148`
COMPLETED. Seed146 geometric score lowest/highest quartile (136 each):
actual treatment effect −0.79/+71.58mm; high−low +72.38mm, environment
cluster-bootstrap 95% interval +61.27 to +81.82mm. Seed148 (156 each):
+0.31/+25.81mm; high−low +25.51mm, interval +22.18 to +28.18mm.
Thus both prespecified thresholds and the interval gate pass. Raw
state+action MLP also ranks: high−low +65.54/+24.32mm, slightly below
geometric Cm. This comparator prevents claiming geometry is uniquely
necessary from this Probe.

The score is computed only from pre-intervention state and candidate actions;
random assignment supplies the observed subgroup treatment effect. Because
the test trajectory/object are still s3/airplane, this is exploratory
evidence of within-task heterogeneity, not cross-trajectory or policy gain.

## Decision update

Proceed to a latency-bounded, few-environment online candidate-action
Probe with matched actor-only control; stop if geometry inference is too
slow or grasp stability degrades. Do not claim `Cm-on > Cm-off` from
offline CATE ranking alone. Formal multi-training-seed policy Validation
remains necessary for the paper.

## Artifacts

`src/task/CmResidual/tools/probe_cm_cate_ranking.py`
`outputs/CmResidual/agent_cm_cate_ranking_s146148/report.json`
