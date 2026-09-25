# P-20260925-cm-postcontact-heldlift-value

date: 2026-09-25
branch: `agent/cm-postcontact-heldlift-value`
classification: Decision Probe
status: **UNPROMISING for this post-contact wrist-z value/policy route**

## Question and pre-registered decision

在固定的 self-trained airplane 任务分布内，把接触后的随机动作干预和
完整 episode 的 held-lift 结果连接起来，能否学到一个比 state-only 和
action-shuffled 更有用的 Cm value/ranking 表示？冻结该表示后，在线
选择候选动作是否提高 held-lift？

进入在线 Probe 的最低条件是：完整标签通过审计；留出 simulator seed
上的 Cm-aware policy value 同时超过 state-only 与 action-shuffled，并且
达到至少 5 个百分点的合并增益。在线比较固定为同 seed、同 motion/start
frame 协议的 Cm-on/Cm-off；这一步仍属于 Probe，不能单独形成正式因果
结论。

## Data collection

`evaluate_randomized_action.py` 新增了 `--record-final-outcome`。每个环境
在第 50 步接触后只做一次实际执行的 `+/-0.1` wrist-z 干预，继续到完整
episode 结束，再按 `env_id` 回填：

* `final_lift_success`：物体上升至少 3 cm 且手物接触连续 5 步；
* `final_max_contact_lift_m`；
* `final_contact_fraction`；
* `final_episode_steps`。

烟雾运行 `agent_postcontact_value_smoke3_s246_n2` 验证了标签对齐。正式
随机数据使用固定 self-trained checkpoint、airplane motion source、64
环境和 `followup_horizon=5`：

| seed | 有效随机行 | plus / minus | 有效行 held-lift |
| ---: | ---: | ---: | ---: |
| 246 | 63 | 32 / 31 | 41 / 63 |
| 247 | 60 | 30 / 30 | 37 / 60 |
| 248 | 60 | 30 / 30 | 39 / 60 |
| 249 | 62 | 31 / 31 | 46 / 62 |

训练固定为 246/247，留出测试固定为 248/249。输入文件 SHA、assignment
与实际 action 的 `0.1` 差值、pre-contact 条件和有限性均由工具检查。

## Offline value probe

工具 `src/task/CmResidual/tools/probe_postcontact_heldlift_value.py` 只用
干预前状态（相对关节位置、关节速度、物体状态、base action、progress、
start frame），先标准化和 24 维 PCA，再拟合二元 held-lift head；Cm-aware
增加 treatment 与状态的交互项。比较：

* `state_only`：不输入候选动作；
* `action_main`：只输入动作主效应；
* `cm_aware`：动作和状态交互；
* `action_shuffled`：训练集内置乱 treatment 对应关系。

在 122 条留出随机行上的结果：

| 模型 | held-lift factual AUROC | IPW policy value | 选择的 −z / +z |
| --- | ---: | ---: | ---: |
| state-only | 0.806 | 0.672 | 122 / 0 |
| action-main | 0.801 | 0.672 | 122 / 0 |
| Cm-aware | 0.764 | **0.702** | 80 / 42 |
| action-shuffled | 0.809 | 0.678 | 44 / 78 |

Cm-aware 比两个对照高，但只高约 2.4–3.0 个百分点，未达到 5 个百分点
门；其预测 action contrast 的高低四分位实际差为 0.059，95% environment
cluster 区间为 `[-0.455, 0.474]`，没有超过置乱控制的 0.213。预注册
gate 为 `false`，因此没有把离线拟合称为 Cm policy utility。

## Frozen online Cm-on/Cm-off probe

将训练 seed 246/247 冻结的 `cm_aware` head 接入采集器，Cm-on 在第 50 步
选择 `+/-z`；同 seed 重新运行 `policy-mode=off` 作为 Cm-off。模型在两
个留出 seed 都选择了 61/64 个有效环境的 `-z`：

| held-out seed | Cm-off | Cm-on | on − off |
| ---: | ---: | ---: | ---: |
| 248 | 46/64 | 42/64 | −6.25 pp |
| 249 | 42/64 | 35/64 | −10.94 pp |
| **合计** | **88/128** | **77/128** | **−8.59 pp** |

同 seed 的 motion ID 和 start frame 按环境一致；但 GPU 仿真重放的 pre-action
`q/object_state` 并未逐元素复现（因此这是 matched protocol Probe，不是
严格的 same-state causal pair）。在线结果与离线小幅正向差异不一致，说明
当前 value head 对 simulator replay/state drift 不稳，且强行施加 `-z`
会伤害最终 held-lift。

## Decision update

停止本轮“第 50 步单次 wrist-z、二元 value head、强制动作替换”路线，结果
记为该 action family 的 `UNPROMISING`。保留采集器的完整 episode 标签能力，
后续若继续应先做同一仿真运行内的三臂（`base/+z/-z`）随机化或带 abstain
的候选选择，再谈 Cm-on/off；不再调当前阈值、正则或 seed 来追逐这次差异。

这不是对 Cm 总体或跨物体问题的结论。当前证据只覆盖一个 airplane
分布、一个接触时刻和一个局部动作族；跨物体泛化仍是后续 Research Debt。

## Artifacts and checks

* `third_party/DExplore/dexplore/evaluate_randomized_action.py`
* `src/task/CmResidual/tools/probe_postcontact_heldlift_value.py`
* `src/task/CmResidual/tests/test_probe_postcontact_heldlift_value.py`
* `outputs/CmResidual/agent_postcontact_heldlift_value_probe_v2/report.json`
* `outputs/CmResidual/agent_postcontact_cmonoff_report_v1/report.json`

`PYTHONPATH=. python -m pytest -q` 的相关随机干预、follow-up 和新分析
测试共 **12 passed**；采集器和分析工具均通过 `py_compile`。
