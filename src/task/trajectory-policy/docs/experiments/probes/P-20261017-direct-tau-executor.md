---
schema: ref2dex.probe.v2
probe_id: P-20261017-direct-tau-executor
claim_id: C3
hypothesis_family: HF-trajectory-policy-executor-transition
probe_index_in_family: 3
seed_pool: probe
experiment_id: P-20261017-direct-tau-executor
task: trajectory-policy
branch: main
git_commit: e8ea6019
date: 2026-10-17
classification: Decision
status: UNPROMISING
hypothesis: A closed-loop policy can use a future GT tau plan as a high-level plan and emit direct active Inspire actions without the old tau-to-R-to-PD inverse route.
question: Does direct pi_exec(state, GT tau future8, task goal) preserve grasp/lift/hold on an unseen corrected s1 sequence, and does an oracle interaction context C recover it if not?
decision_changed_if_positive: advance from GT tau to WAM-predicted tau; treat tau as a usable high-level plan.
decision_changed_if_negative: do not start PPO or WAM-C training from this probe; use the oracle-C comparison to decide whether interaction context is the missing input.
protocol: BC on official applied native actions from the audited s3 teacher rollout, then frozen direct-action execution with 8 official teacher calibration environments and 8 direct environments on corrected unseen s1; oracle-C arm adds current 10D contact/support context.
train_source: outputs/trajectory-policy/competent-pd-teacher-20261011-r2/trajectory.npz
unseen_source: outputs/trajectory-policy/gt-tau-new-sequence-20261017-r4/teacher/trajectory.npz
unseen_sequence: s1_airplane_lift
action_contract: policy emits 12 active native coordinates; passive coordinates are zero and Inspire coupling is applied once by DExplore.
input_contract: measured state87 + future hand tau8 in the live object frame + task goal pose12 + previous applied active action; oracle-C arm appends contact pair/gap, support gap, object velocity6 and table footprint.
gate: direct arm requires hold45 >= 6/8, success >= 6/8 and clipping < 1%; teacher is calibration only.
runtime: /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train, Torch 2.0.1+cu118, GPU 0 with pre-existing 0.4 GiB context.
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：撤销方法否决；降级为故障观测。** 撤销由这个受污染合同产生的 τ/C 方法否决；0/8 降级为旧实现局部故障，保留原始轨迹与数字。
- **能支持：** 旧固定实现的 learner 0/8 hold45 与 teacher 8/8、同序列诊断及离线拟合数值，仍可作为历史实现/分布故障观测。
- **不能支持：** 不能支持独立参考 τ、v6 或交互信息 C 无效：旧实现混用 reference future 与 live-current 手，mean-pool 丢失时间顺序，训练是单 motion 多环境复制；10D oracle-C 不代表一般 C。本波 teacher placement 4/8 或 5/8 也未过完整校准。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


## Motivation and decision note

The prior physical probe showed that the old `GT tau -> R -> PD` route does not
hold a coordinate-corrected unseen sequence. This probe tests the proposed
replacement directly: the policy reads the live state every control step and
outputs the native action, while GT tau is only a plan. The cheapest useful
discriminator is one BC fit, one unseen closed-loop run, and one oracle-C arm;
there is no PPO or WAM-v4 training before that discriminator.

## Offline BC

The fit uses `applied` native teacher commands and selects active indices
`(0,1,2,3,4,5,6,8,10,12,14,15)`. The training rollout contains 32 parallel
copies of one s3 sequence, so these are not 32 independent demonstrations.
The direct arm reached active-action L1 0.0155 on the held-out environment
split and 0.1121 on the unseen s1 source replay. The oracle-C arm reached
0.0156 and 0.0796 respectively. These are offline diagnostics, not task
success evidence.

## Closed-loop results

On unseen corrected `s1_airplane_lift`, the teacher calibration was 8/8 hold45
and 4/8 success. The direct GT-tau BC arm was 0/8 hold45, 0/8 success, median
held 0, clipping 0. The oracle-C arm was also 0/8 hold45 and 0/8 success,
median held 0, clipping 0. Therefore the registered decision result is
`UNPROMISING` for this fixed BC contract. The oracle-C arm did not recover the
failure, so this particular 10D contact proxy is not evidence that C restores
the missing behavior.

As an implementation diagnostic only, the same-sequence s3 run produced 6/8
hold45 and 4/8 success for direct BC, while its teacher was 8/8 hold45 and
8/8 success. This separates a working closed-loop action path from the much
larger sequence-generalization failure on s1; it is not a same-motion claim.

Artifacts:

- BC: `outputs/trajectory-policy/direct-tau-bc-20261017-r2/`
- unseen direct: `outputs/trajectory-policy/direct-tau-execution-20261017-r2/`
- same-sequence diagnostic: `outputs/trajectory-policy/direct-tau-execution-s3-20261017-r1/`
- oracle-C BC: `outputs/trajectory-policy/direct-tau-oracle-c-bc-20261017-r1/`
- oracle-C unseen execution: `outputs/trajectory-policy/direct-tau-oracle-c-execution-20261017-r1/`

## Limits and next action

This is a single-seed Probe. The teacher source duplicates one s3 motion, and
the oracle C is a compact current contact proxy rather than a learned or
full-force interaction representation. The result does not refute tau as a
plan in general, but it does not justify starting PPO from this BC checkpoint
or training WAM-C yet. The next useful route is to inspect sequence-normalized
tau/state representation and, if retained, define a richer interaction oracle
before any PPO budget is spent.

## 2026-10-22 负结果 CPU 审计：结论范围修订

原0/8是真实轨迹，但旧实现混用reference future与live-current手、mean-pool时间，且训练是单motion的多环境复制；不能否决独立参考τ语义或交互信息C。Oracle-C只是当前10维代理，0/8不证明C无用。实际manifest为5fb0c65b；完整gate还要求teacher placement≥6/8，而本波只有4/8或5/8，因此完整placement结论受校准不足限制，hold45的learner0/8对teacher8/8现象仍保留。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
