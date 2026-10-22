---
schema: ref2dex.probe.v2
probe_id: P-20261018-absolute-plan-executor
claim_id: C3
hypothesis_family: HF-trajectory-policy-executor-transition
probe_index_in_family: 7
seed_pool: probe
experiment_id: P-20261018-absolute-plan-executor
task: trajectory-policy
branch: main
git_commit: d066138a
date: 2026-10-18
classification: Decision
status: UNCLEAR
hypothesis: Adding the reference hand's current object-frame anchor will make direct tau execution identifiable after learner drift; a control-space PD objective will expose physical errors that raw absolute-PD normalization hides.
question: Does an absolute reference-hand anchor plus ordered future tau improve offline teacher-PD prediction on the existing DAgger queries sufficiently to justify a new recovery-basin roll-in?
decision_changed_if_positive: run one bounded-intervention DAgger collection with beta 0.25 using the repaired absolute-plan contract.
decision_changed_if_negative: do not spend another autonomous DAgger round; treat the anchor-plus-loss probe as insufficient and inspect interaction context or richer plan geometry.
protocol: Reuse the existing direct DAgger query packet. Add A_ref_t=R_ref_t(h_ref_t-o_ref_t), a 33-D current reference hand anchor in the reference object frame, to the live state87, ordered tau8x33, and previous active action12 inputs. Warm-start from the selected r3 checkpoint. Train with equal original/query batches using wrist control targets qdes-q and finger control targets qdes, normalized by train control-space dispersion, plus 0.1 action Huber. Query environments 0-4 are fit and 5-7 are held out. Report xyz mm, rotation rad, finger rad, held-out query metrics, and query_clipped rates.
query_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/dagger_queries.npz
rollin_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/trajectory.npz
base_checkpoint: outputs/trajectory-policy/direct-tau-bc-r3-20261018-r2/best.pt
runtime: /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train, Torch 2.0.1+cu118, GPU 0
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：降级。** UNCLEAR 仅保留为离线/收集诊断；降级历史物理资格及失败机制的外推，撤销把混合 roll-in 当自主成功的引用。
- **能支持：** 绝对参考锚点/控制空间损失下的离线误差、query 分布和 normalized Huber 筛选未过；teacher warm-up 改变了收集分布。
- **不能支持：** 不能支持自主抓持成功、v6/τ 无效或绝对锚点无用。本卡没有最终自主物理评估；120 控制 teacher warm-up 加 beta=.25 的 8/8 hold 含实际约 46.11% 教师干预。原生单位误差没有独立预注册物理门槛，不能把离线筛选叫物理资格。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


## Motivation and decision note

The prior direct executor supplied only the reference future displacement
`tau_ref`. On the teacher manifold this implicitly identifies the current
reference location, but after learner drift it leaves the recovery target
underdetermined. This probe adds only the missing current reference anchor; it
does not add oracle interaction context or retrain the WAM planner.

The same query packet is reused so the probe separates input-contract and loss
changes from a new physics distribution. The offline gate is deliberately
physical: normalized loss is reported for optimization, while xyz, rotation,
and finger errors are reported in their native control units. `query_clipped` is
audited rather than silently filtered.

## Results

The first offline fit reused the old autonomous query packet. The absolute
anchor and control-space loss were wired correctly, and `query_clipped` was 0,
but 99.30% of rows were catastrophic. Held-out query error after fitting was
152 mm wrist xyz RMSE, 0.482 rad rotation RMSE, and normalized Huber 0.329.
This is insufficient to select a physical checkpoint, but it is confounded by
the collection distribution.

A bounded-intervention roll-in with beta 0.25 was then collected. Its actual
intervention fraction was 0.25, but the state audit remained 0.46% near,
1.86% moderate, and 97.68% catastrophic; the direct arm achieved 0/8 hold45.
This confirmed that beta 0.25 is too weak when the starting executor is already
outside the recovery basin.

For a recovery-basin collection, teacher actions were forced for the first 120
controls (through the stable grasp phase), then beta 0.25 was used. The direct
arm held in all 8 environments during this collection, but placement was not
evaluated as autonomous behavior. The resulting query distribution was 22.19%
near, 10.61% moderate, and 67.20% catastrophic; actual intervention fraction
was 0.461 because of the warm-up. Query clipping remained 0.

Fitting the same absolute-plan policy and control-space PD objective on this
recovery-basin packet reduced held-out control RMSE from 0.345 to 0.273 and
normalized Huber from 0.784 to 0.493. Native errors were still 211 mm xyz,
0.433 rad rotation, and 0.182 rad finger RMSE, so the offline gate did not
pass. The source-sequence diagnostic was much better (41.7 mm xyz, 0.078 rad
rotation, 0.050 rad finger), showing that the remaining problem is the
learner-state distribution rather than checkpoint loading or a nonfinite
policy.

Artifacts:

- absolute contract fit on the old autonomous packet:
  `outputs/trajectory-policy/absolute-plan-pd-fit-20261018-r1/`
- beta 0.25 roll-in (insufficient recovery basin):
  `outputs/trajectory-policy/absolute-plan-dagger-rollin-20261018-r1/`
- beta 0.25 with 120-control teacher warm-up:
  `outputs/trajectory-policy/absolute-plan-dagger-rollin-20261018-r2/`
- recovery-basin absolute-plan fit:
  `outputs/trajectory-policy/absolute-plan-pd-fit-20261018-r3/`

The probe is `UNCLEAR`: the anchor contract and bounded collection repair the
data coverage, but the held-out live-state PD error remains too large to justify
an autonomous physical replay. Do not interpret the 8/8 hold during the warm-up
roll-in as executor success; teacher actions were part of that trajectory.

## Next decision

Stop adding DAgger rounds to this contract for now. The next informative branch
is an executor with live replanning or explicit interaction context `C`, tested
on a small matched probe; a longer PPO run is not justified by the current
offline gate.

## 2026-10-22 负结果 CPU 审计：结论范围修订

没有自主物理执行结果。120步teacher warmup后β=.25的rollin实际干预比例46.11%，其8/8 hold45不能作为executor成功。最终fit实际manifest4d21a1fb；脚本状态门槛实际是normalized Huber<.2，原生xyz/rot/finger有数值但没有独立预注册阈值，因此“physical/native offline gate”应限定为原normalized筛选，不能包装成物理资格。保留离轨query误差仍大和该筛选未过，不否决绝对锚点或τ路线。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
