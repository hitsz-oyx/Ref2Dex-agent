# P-20260925-cm-history-grasp-value

date: 2026-09-25
branch: agent/cm-conditional-grasp-value
classification: Decision
status: PROMISING for conditional action-effect ranking; UNCLEAR for history-specific gain

## Question

在已有真实随机化 `+/-z` 干预上，带短历史的动作条件 Cm 能否预测
后续五步的多目标结果，并把真实处理效应异质性排序？这里把 Cm
视为条件概率模型，不把一条观测状态当成可以复制的确定性仿真器。

## Hypothesis and decision

H1：输入最近五个干预时刻的观测/已执行动作历史，以及当前候选动作
的 `q/dof_vel/object_state`，比当前时刻动作模型、当前动作屏蔽模型和
当前动作置乱模型更能预测未见 seed 的：

* 一步物体 z 位移；
* 五步物体 z 位移；
* 五步接触比例；
* 末步仍接触。

H1 的排序门是：在未见 `s155/s156` 的真实随机 A/B 结果中，历史 Cm
按预测 `+z - -z` 差异分出的最高四分位，比最低四分位的实际处理效应
更大；至少一个连续承重目标（五步 z 位移或接触支持 z 位移）满足
高低差 >= 5 mm 且 environment-cluster 95% CI 下界 > 0，同时接触比例
目标的方向与该物理结果不矛盾。历史 Cm 的高低差还必须超过动作屏蔽和
动作置乱控制的高低差，或其区间与控制重叠但自身区间不跨零。

通过：只进入低环境数、延迟受限的在线候选动作 Probe；仍不能声称
Cm-on 优于 Cm-off，也不直接接 PPO。

不通过：停止当前“短历史 + 五步随机干预价值”路线，把结果保留为
Cm 的研究证据，转向能观测最终 held-lift 的信用分配数据；不在同一
seed 上继续调网络或阈值。

## Data and split

复用已完成的 `agent_randomized_followup_s151...s156`，每次 64 环境、
11 个干预时刻、`delta_z=0.1`、follow-up horizon=5。训练固定为
`s151/s152/s153/s154`，测试固定为未见 `s155/s156`；测试标签在模型
冻结后才读取。只保留随机分配且干预前接触的行，历史只使用当前时刻
及其之前的观测和已执行动作，不读取任何 follow-up 字段。

## Models and controls

* `history_action`: 五时刻历史 + 当前已执行/候选动作；输出三个连续
  均值/方差头和一个末接触 Bernoulli 头。
* `current_action`: 仅当前时刻的同类输入。
* `history_blind`: 历史保留，但当前动作替换为未干预 base action。
* `history_shuffled`: 在每个干预时刻内置乱动作扰动，保留状态和边际
  动作分布，打断动作与结果的对应关系。

所有模型使用相同随机种子、训练步数和目标标准化。高低分组阈值只由
冻结模型在测试输入上的预测决定；真实结果按干预时刻分层，bootstrap
按环境 ID 聚类。报告保留每个目标，禁止合成为手工单一 `J` 分数。

## Budget and stop

CPU 两线程、GPU 0、wall <= 30 min、输出 < 20 MB。输入 SHA 漂移、
训练/测试泄漏、非有限值、分层缺少两臂或超预算即停止。

## Limits

这是顺序随机化下到达状态的平均条件效应 Probe，不是同状态逐样本
反事实；五步结果不是长期策略效用，现有数据没有最终 held-lift 标签。
即使通过，也必须另做 matched Cm-on/off 验证。

## Result

Run `agent_cm_history_value_probe_20260925_v2` completed on CPU with
800 updates and 1000 environment-cluster bootstrap resamples. The model
was fit on 2,540 selected rows from `s151/s152/s153/s154` and evaluated on
1,262 untouched rows from `s155/s156`; all six transition SHA256 values were
checked before fitting.

The `history_action` model's predicted `+z - -z` score produced the following
pooled held-out randomized effects (high-score quartile minus low-score
quartile):

| outcome | high-minus-low | 95% cluster CI |
| --- | ---: | ---: |
| one-step object z displacement | 32.42 mm | [31.16, 33.71] |
| five-step object z displacement | 32.96 mm | [27.98, 37.85] |
| contact-supported z displacement | 33.70 mm | [28.78, 38.67] |
| five-step contact fraction | +0.271 | [+0.203, +0.340] |

The same five-step z ranking was positive in each untouched seed separately:
31.91 mm [24.25, 39.77] for `s155` and 35.42 mm [26.80, 44.85] for `s156`.
The action-shuffled control ranked in the opposite direction for five-step z
(`−8.73` and `−16.30` mm by seed; pooled `−13.67` mm [−20.40, −6.67]).
The predeclared physical, control-separation and contact-direction gates
therefore passed (`prespecified_continue_gate_passed: true`).

The current-action model was also strong (pooled five-step z gap 31.78 mm
[26.40, 37.28]), so the Probe does not isolate a benefit from five-step
history. The result is `PROMISING` for conditional action-effect ranking on
this trajectory/object and `UNCLEAR` for the added history representation.

An independent fit seed (`241951`, 500 bootstrap resamples) reproduced the
direction: history-action five-step z gap 34.72 mm [28.91, 41.09], while the
shuffled control was −1.95 mm [−9.12, 4.10]. This is a robustness repeat of
the frozen split, not a new simulator seed or a validation claim.

## Decision update

Do not connect this model to PPO or call it a grasp-value model. Repeating the
frozen representation on the held-out cross-object apple split
(`P-20260925-cm-crossobject-history-value`) failed the transfer gate, so an
online selector based on this representation is stopped. A new route should
either learn a cross-object representation or attach randomized interventions
to an actual final held-lift label; any policy claim still requires matched
Cm-on/off.

Artifacts:

* `src/task/CmResidual/tools/probe_history_value_cm.py`
* `src/task/CmResidual/tests/test_probe_history_value_cm.py`
* `outputs/CmResidual/agent_cm_history_value_probe_20260925_v2/report.json`
