---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-finger-residual
experiment_id: P-20261011-goal-conditioned-finger-residual
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 81b59f4
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 9
seed_pool: probe
seeds: [316]
decision_changed_if_positive: retain analytic-wrist/finger-residual decomposition and design a separately bounded continuation
decision_changed_if_negative: close this low-level student decomposition and proceed only with fresh teacher provenance or Stage 2 offline forecasting
status: UNPROMISING
run_id: goal-conditioned-finger-residual-20261011-r2
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：加注。** 加注保留该分解局部门槛失败及部分持握，排除 r1 工程中断与方法层面外推。
- **能支持：** 有效 r2 hold45 7/16、long-held 0/16、clipping 13.5148%，说明该 finger-only 分解未恢复组合门槛。
- **不能支持：** 不能支持完全没有持握、v6/τ 无效或所有 wrist/finger 分解无效。r1 resource guard 中断无轨迹不能作负例；仍是单 motion/seed 和特权 nominal。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


# Analytic-wrist, finger-only residual student

## Motivation and decision note

The full 12-D goal-conditioned residual student reached partial holds but failed
its physical gate, while the retained tau executor has a nominal wrist
feedforward and the user-specified ref2 route explicitly allows a learned
low-level controller to focus on contact/preload. This Probe tests whether the
student's learned wrist correction is the failure source. It keeps the same
24-step goals, live measured-state feature contract, retained teacher source,
executor and physical gate, but assigns all six wrist coordinates to the
analytic nominal controller and trains only the six active finger residuals.

The experiment distinguishes two hypotheses: (A) full residual output damages
tracking through an under-conditioned wrist inverse, in which case finger-only
student execution should recover the gate; (B) the retained teacher labels or
goal-conditioned student distribution are insufficient even with analytic wrist,
in which case the gate remains failed. It does not claim a deployable H-to-tau
planner or a REGRIND reproduction.

## Contract and protocol

The input is the existing 1170-D residual observation. Previous residual history
has six zero wrist slots and six bounded finger slots. The network emits six raw
finger latent coordinates; the shared executor inserts zero wrist residual,
applies tanh/limits, coupling, native conversion and clipping. No future q,
phase, clock, tactile input, contact label, native action or Cm enters the
policy. Teacher labels are exactly the retained `role==tracker` raw latent
coordinates 6:12; the engineering-only packet supplies goals only.

Fit one environment-held-out BC run (seed316, at most 3,000 updates/600s), then
execute one fresh 16-env/542-control physical run. The predeclared physical gate
is at least 12/16 long-held terminal episodes and clipping `<1%`; otherwise the
Probe is `UNPROMISING` and no PPO is launched. Independent audit must replay
features, masked latent, bounded residual, executor, PD targets and outcomes.

Outputs:
`outputs/trajectory-policy/goal-conditioned-finger-residual-bc-20261011-r1/`,
`outputs/trajectory-policy/goal-conditioned-finger-residual-20261011-r2/`, and
the matching audit directory.

## Results

The offline fit was `PROMISING` for a finite validation loss: 6,504 train rows,
2,168 held-out rows, and best raw-finger latent L1 `0.0208398` after 1,300
updates on GPU0. The valid physical rerun is
`outputs/trajectory-policy/goal-conditioned-finger-residual-20261011-r2/`.
The first physical attempt (`...-r1/`) stopped at control 384 because a
transient GPU process guard fired before any trajectory was written; it is a
setup failure and is not counted.

The completed r2 run used 16 environments and 542 controls. It reached `0/16`
long-held terminal episodes and `7/16` hold45 episodes, with median held `5`
frames and clipping `13.5148%`. All controlled-placement success flags were
false, so the physical gate failed and the Probe is `UNPROMISING`.

The independent replay audit is `PASS`: feature error `2.86e-6`, latent error
`5.96e-7`, bounded-residual error `5.36e-7`, executor target error `3.58e-7`,
command error `4.77e-7`, exact clipping flags and exact PD reconstruction. The
finger-only intervention therefore does not rescue the current residual
student; no PPO or further wrist/finger mask sweep is authorized by this card.

## Limitations and stopping rule

This is a single-motion, single-seed privileged nominal-reference diagnostic.
Even a positive result only retains the decomposition for a new card; it does
not prove generalization, H-to-tau forecasting, Cm utility, or full REGRIND
reproduction. A negative result closes this decomposition at the current budget.

## 2026-10-22 负结果 CPU 审计：结论范围修订

有效物理运行是r2（manifest b1f1883d），r1资源guard中断且无可审计轨迹应排除。保留hold45 7/16、long-held0/16、clipping13.5148%的混合现象；finger-only未恢复注册门槛不等于完全无持握或所有wrist/finger拆分无效。仍依赖特权nominal，单motion单seed。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
