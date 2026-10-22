# S2c：阶段判据、dense reward 与固定预算复评

范围：仅 `src/task/ptv3-shared-wam-v6/**`。没有修改 v3 executor、native host、物理参数、
checkpoint 或数据；没有训练、模型推理或新物理交互。以下是工程合同与档案诊断，
不认证新策略成功率或 dense reward 的学习收益。代码提交以本报告所在的 S2c commit 为准；
JSON 中 `git_commit` 是运行时父提交，`implementation_state` 明示当时含本轮工作区 diff。

## A. 阶段判据 diff 与复现

新增 `src/ptv3_shared_wam_v6/staged_progress.py`；修改
`tools/run/samples_efficiency_env.py` 的 `_refresh`、`step` 和 indexed-reset 日志：

```diff
- reward = shared MeasuredRewardTracker; parts discarded
+ observed feedback -> StagedProgress -> StagedDenseReward
+ info.stages / reward_parts / stage_shaping / reward_contract
+ episode_results.stages / reward_parts_sum
```

`stages` 同时给 `current`（当前满足）、`achieved`（历史首次通过）及
`first_time_s`（episode elapsed 秒，未通过为 null）。因此“曾 hold、现在已脱手”可以明确表达；
每次 indexed reset 清空本环境的阶段历史，terminal 的快照在 reset 前保存。

| 阶段 | 判据（均为实测、无 WAM 预测输入） |
| --- | --- |
| approach | palm gap ≤100mm 且 mean tip gap ≤50mm；无物理 abort |
| contact | approach、`force_valid`、至少 2 个 tip force norm ≥0.1N，且原 `feedback.contact_observed` 为真（至少 2 指距表面≤18mm） |
| lift | contact 后的更晚一帧，当前 contact/有效力持续满足、`grasp_established && !grasp_lost`、明确 unsupported；`z-max(z_initial,z_first_contact) ≥0.05m` |
| hold | 连续满足上述 lift ≥1s；失接触、无效力、失握、落回、支撑或 abort 任一出现即清零当前计时；相邻观测间隔超过 1/30s+1e-8 也清零 |
| release | 先有 hold，随后 scheduler 已授权 RELEASE，`force_valid` 且全部 tip force norm <0.1N、`feedback.released && !grasp_established`、hand clearance ≥25mm 连续≥0.12s |

原 feedback 的 `released` 使用 **最小** tip gap ≥25mm 和自身 dwell，不能仅靠均值间隙或
“force 数组全零”证明脱手；无效 force mask 永远不表示零力。新 release 还额外要求上述
全部条件连续≥0.12s。它表示物理脱手阶段，完整终点/有序路径/稳定支撑仍由原 scheduler
单独判断，`task_success` 不由阶段历史拼凑。

lift 基线取首次有效 contact 的高度，因此先无接触抬高、再在高处伪造接触不能追认抬升。
5cm 阶段是独立诊断，不把原 synthetic 精确 8cm 终点降成 5cm。

默认 task 仍为 `synthetic_hold`，其 release 应为 false。
新增显式 `SampleEfficiencyEnv(..., task_mode='source_declared')` 和 launcher
`--task-mode source_declared`：完整使用原 immutable TaskSpec，供原 RELEASE 任务输出所有阶段。
其 `source_full_task_success` 为真实 scheduler 结果；synthetic 模式此字段仍是 null。
source 模式仍限 20s/600 controls，超预算不能重标成成功。

复现 A 与 D（仓库根目录运行，CPU 是纯合同测试，不涉及模型/仿真）：

```bash
T=src/task/ptv3-shared-wam-v6
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider --basetemp="$T/tmp/s2c-pytest" "$T/tests/test_staged_progress.py" "$T/tests/test_samples_execution_signals.py" "$T/tests/test_samples_efficiency_history.py" "$T/tests/test_samples_efficiency_summary.py" "$T/tests/test_negative_result_invariants.py" "$T/tests/test_v6_executor_wiring.py"
```

期望：`44 passed`。涵盖真实 v6 环境方法、CPU replay transport、reward 求和、源 TaskSpec
不变、强制 reset 的失败罚分与 terminal/reset episode 所有权；没有把 transport stub 当新物理。

## B. dense reward diff、公式、量纲、上限

只使用已有 `RewardFeatures` 的 `palm_gap_m`、`mean_tip_gap_m`、`hand_clearance_m`，
已有初始/当前 object z、contact force/有效掩码、反馈、action、时钟。没有新增传感器或物理量。
原外部 reward 分项名继续为 `shaping,event,sparse,time,effort`，返回值严格为这五项之和。
`stage_shaping` 是 shaping 的五项分解，不能再加一次。旧 phase potential 被本实现替换，
不是叠加两套 reward；这会改变学习目标，不声称保持旧最优策略不变。

令 `clip(x)=min(1,max(0,x))`，`a_t` 是实际 canonical clipped action12：

| 项 | 公式 | 输入量纲 / reward 量纲 | 每回合正分上限 |
| --- | --- | --- | --- |
| approach progress | `q_A=clip(1-(palm_gap/.10 + mean_tip_gap/.05)/2)`，仅安全时 | m/m，无量纲 | 0.10 |
| contact progress | `q_C=1[current.contact]` | N 与有效掩码用于门控，无量纲 | 0.15 |
| lift progress | `q_L=clip((z-max(z_initial,z_first_contact))/.05)`，仅有效接触后、coupled、unsupported 时 | m/m，无量纲 | 0.25 |
| hold progress | `q_H=min(continuous_hold_s/1s,1)`，断续清零 | s/s，无量纲 | 0.25 |
| release progress | `q_R=1[current.release]`，全门控完成才有分 | 无量纲 | 0.25 |
| event | `0.1 * 本 tick 首次通过且未付过的阶段数` | 无量纲 | 0.50 |
| sparse | 原 scheduler 完整成功 `+1`；失败/物理 abort/工程 reset `-1`，失败优先 | 无量纲 | 1.00 |
| time | `-0.01 * dt_s` | 1/s × s，无量纲 | 0（30Hz 每步 -1/3000） |
| effort | `-0.01 * mean(a_t**2)` | canonical action 无量纲 | 0（每步最低 -0.01） |

五个 progress 的权重为 `w=[.10,.15,.25,.25,.25]`。
每回合维护 `h_t=max(h_(t-1),q_t)`；
`shaping_t=sum(w*(h_t-h_(t-1)))`。
初始化时 `h_0=q_0` 且初态已满足的阶段加入 paid 集合，不给初态免费分。
总 shaping≤1、event≤0.5、唯一 terminal sparse≤1，故忽略负成本总正分≤2.5。
这是有界进展增量，不是旧 `gamma*Phi_next-Phi_prev`，也不再需要终止 potential 修正。

防刷规则：反复靠近/接触/抬落只能刷新一次进展纪录；hold 片段不相加，历史最长片段
最多领一次对应 shaping 增量，未满连续 1s 绝不给 hold 阶段或其 event；release 未满足
授权、有效零力和真实分离则整项为 0。停在部分阶段不会每步获正分，还持续付 time/effort；
失败仍付 -1。此机制抑制重复刷分，不等于证明 PPO 一定越过局部最优。

环境逐步输出五项 reward，结束输出累计和；测试检查逐步总和与返回 reward 一致。
`train_samples_efficiency.py` 的 resume identity 新绑定 `stage_contract` 和 `reward_contract`，
且 source task 有独立 task contract：旧 reward checkpoint 不能静默续跑。没有启动此入口训练。

## C. 固定初态、固定预算与耗时分布

新增 `configs/staged-eval-panel-v1.json` 固定 6 序列和历史实测初态：cylindersmall、
cubesmall、cup、airplane、cubemedium、cylindermedium；seed185，30Hz，
每例最多600 controls，总上限3600，无重试/追加 seed。评价工具是
`tools/audit/evaluate_staged_progress.py`。

档案入口检查 601×6 状态及完整连续 ticks，并把 q/dq/physical/object/hand/table/contact/valid
初态逐数组与固定来源比较（atol1e-7，rtol0）；排序、初态或覆盖不足直接报错。
每例在第一次 scheduler terminal 停评分，剩余档案后缀不偷偷增加预算。
按与环境相同 20s timeout 评分，report 保留终止原因、删失与忽略后缀数。
初态只认证保存的可见数组，未保存的 PhysX solver/caches 不冒充配对通过。

每阶段报告 `successes/total/not_observed`，`first_passage_s` 和
`from_previous_stage_s` 的 n/min/p25/median/p75/max；只对通过该阶段的样本算时间，
失败/删失不能填0秒或混入耗时均值。JSON 逐例保留全部布尔/首次时间/最长连续 hold。

可执行 demo（输出路径必须全新；下面两条不是训练命令）：

```bash
T=src/task/ptv3-shared-wam-v6
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python "$T/tools/audit/evaluate_staged_progress.py" --demo --output "$T/tmp/s2c-demo-reproduce.json"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python "$T/tools/audit/evaluate_staged_progress.py" --panel "$T/configs/staged-eval-panel-v1.json" --output "$T/tmp/s2c-archive-reproduce.json"
```

第一条期望 `PASS, synthetic_injections, successes={approach:6, contact:4, lift:3, hold:2, release:1}`，
6例各150 controls/5s，统计900个合成控制间隔；新物理0。
第二条期望 `PASS, archived_native_trace, successes={approach:2, contact:2, lift:1, hold:1, release:0}`，
完整任务仍0/6。`PASS` 表示评估程序及输入合同通过，不表示 release 任务通过。
真实时间分布见 [档案 receipt](s2c-stage-archive-20261022.json)，
合成正/反例分布见 [demo receipt](s2c-stage-demo-20261022.json)。

档案首达时间：approach median2.9s（1.0–4.8），contact median2.9167s（1.0333–4.8），
lift1.8s（n1），hold3.7333s（n1），release 无通过样本，时间为 null。
旧窄任务5/6与本协议1/6不可直接比较：本次需 coupling、接触后的增量高度、
连续有效观测，且按第一 task terminal 截断；旧档案还早于生产 reset 修复。
不以这份 replay 诊断宣称新执行退化或新正控制通过。
档案未带同 tick canonical command，故没有用假零 action 伪报 reward/effort 评估。

要比较新 frozen executor：先固定生产 reset 的新初态面板并记录同样字段；每臂同序列/同初态/
同最大600控制/同终止协议，先检查实际可见与可得 hidden reset 状态。不得把不匹配初态
塞进当前面板。只做冻结推理/采集，按当轮授权使用空闲物理4/5/6并记录环境和资源。
本轮没有执行该新物理对照，也未重训 v6。

## D. 故意伪造部分进展的反例

新增 `tests/test_staged_progress.py`，并扩展原环境 reset 测试。A 的 pytest 命令直接复现。

| 注入 | 期望 |
| --- | --- |
| z+10cm，力为零/无 contact，但伪称 coupled | contact/lift/hold/release 均 false，后四项 shaping 为0 |
| 持续有效 contact，z不变 | contact 可 true；lift/hold/release 均 false、对应 shaping 为0 |
| 抬升后每15tick注入无效力，累计很长 | 最长有效片段<0.5s，hold false、无 hold event，片段不得拼接 |
| 先无接触抬升，再在高处补 contact | contact 可 true；额外抬升不足5cm，lift false |
| 伪称 released，但仍有多指实测接触力 | release false，release shaping/event 为0 |
| 无效力掩码/未授权/物理 abort 下伪称释放 | release false |
| 两个高处样本相距>1s，中间未观测 | hold 连续计时重置，不能以稀疏样本跨越1s |
| 接触、抬落反复150tick | shaping≤1、event一次付，未连续1s不给 hold event；terminal 后再付分直接拒绝 |

合成正例必须通过全部五阶段，确保不是“把所有输入都拒绝”的假防护。
44项回归 PASS；执行环境 `/home2/wyy/miniconda3/envs/graspenv/bin/python`，Python3.8.20、
NumPy1.24.4；GPU未分配、optimizer updates0、新仿真 controls0。

完整文件 diff 可在本 S2c 提交查看：

```bash
git log -1 --format='%H %s' --grep='feat(v6): add staged evidence and bounded dense reward'
S2C_COMMIT=$(git log -1 --format=%H --grep='feat(v6): add staged evidence and bounded dense reward')
git show "$S2C_COMMIT" -- src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py src/task/ptv3-shared-wam-v6/tools/run/samples_efficiency_env.py src/task/ptv3-shared-wam-v6/tools/run/train_samples_efficiency.py src/task/ptv3-shared-wam-v6/tools/audit/evaluate_staged_progress.py src/task/ptv3-shared-wam-v6/tests/test_staged_progress.py
```
