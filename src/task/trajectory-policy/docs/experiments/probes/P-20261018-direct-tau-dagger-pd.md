---
schema: ref2dex.probe.v2
probe_id: P-20261018-direct-tau-dagger-pd
claim_id: C3
hypothesis_family: HF-trajectory-policy-executor-transition
probe_index_in_family: 6
seed_pool: probe
experiment_id: P-20261018-direct-tau-dagger-pd
task: trajectory-policy
branch: main
git_commit: 357e4a6d
date: 2026-10-18
classification: Decision
status: UNPROMISING
hypothesis: The previous DAgger negative was partly caused by optimizing raw active action instead of live-state PD targets; a PD-space loss should improve teacher correction without changing the tau/state contract.
question: Does warm-start DAgger with normalized wrist/finger PD Huber loss materially reduce learner-state teacher-PD error, and is the improvement sufficient to justify one frozen physical replay?
decision_changed_if_positive: run one frozen physical replay of the PD-loss checkpoint; only continue to recovery-basin collection if the physical gate improves.
decision_changed_if_negative: stop this objective correction and treat the remaining issue as roll-in/recovery distribution or executor-state representation, not as evidence against tau semantics.
protocol: Reuse the existing 3448 learner-state queries without new physics collection. Decode policy active action through the native current-q controller into PD targets; optimize equal original/query batches with normalized wrist/finger PD Huber plus 0.1 action Huber. Keep the r3 standardizer and temporal executor fixed. Query environments 0-4 are training, 5-7 are validation. Also stratify rows by measured student/reference hand and object divergence.
query_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/dagger_queries.npz
rollin_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/trajectory.npz
base_checkpoint: outputs/trajectory-policy/direct-tau-bc-r3-20261018-r2/best.pt
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：降级。** 保留离线改善与本执行器 hold 失败；完整任务负证据降级，未恢复抓持不等于 τ 方法否决。
- **能支持：** 复用 query 下 PD loss 降低离线 PD 误差，无干预 learner hold45 仍为 0/8；99.30% catastrophic query 的分布诊断有效。
- **不能支持：** 不能支持 v6/τ 无效或 PD 监督无用；teacher placement 4/8 未达到完整任务校准，终点误差 0.0402 m 不能替代抓持成功。修 loss 与修恢复分布是不同问题。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


## Motivation and decision note

The previous DAgger run stored current-q teacher PD targets but trained only on
`query_action[..., ACTIVE]`. Since wrist and finger action coordinates have
不同 physical scales, action L1 was not the intended PD correction objective. This
probe changes only that objective and reuses the same query packet, so it isolates
loss-space mismatch from new roll-in data.

The query rows are also stratified rather than silently filtered. A row is `near`
when hand and object position errors are both at most 10 mm, `moderate` when both
are at most 50 mm, and `catastrophic` otherwise. These thresholds are diagnostic,
not a task success gate.

## Offline results

The frozen base policy had query-validation PD RMSE `0.6475` and normalized PD
Huber `0.6283`. PD-loss DAgger reduced them to `0.2533` and `0.0721`; wrist/finger
PD RMSE became `0.3388/0.1164`. Reference-sequence validation remained finite,
with PD RMSE `0.1359`; the fit is therefore marked `PROMISING` as an offline
probe only.

The divergence audit found 16 near rows (`0.46%`), 8 moderate rows (`0.23%`), and
3424 catastrophic rows (`99.30%`). Catastrophic rows averaged 1.399 m hand error,
7.8 cm object-center error, and 1.645 rad q RMS. Their PD RMSE fell from `0.6507`
(base) to `0.2541` (PD-loss DAgger), but they dominate the packet. This confirms
that the original autonomous roll-in was mostly outside the demonstrated recovery
basin; it does not establish that the corrected policy can recover physically.

Artifacts:

- PD-loss fit: `outputs/trajectory-policy/direct-tau-dagger-pd-fit-20261018-r4/`
- row divergence audit: `outputs/trajectory-policy/direct-tau-dagger-pd-fit-20261018-r4/divergence.json`

## Physical follow-up

The offline gate passed, so run one frozen execution with no teacher intervention
on the same unseen corrected `s1_airplane_lift` panel. This is a Probe, not a new
DAgger round or a validation claim.

The frozen PD-loss checkpoint still achieved `0/8 hold45` and `0/8 success`, with
median held frames 0 and clipping rate 0. The teacher calibration in this launch
achieved `8/8 hold45` and `4/8 success`. The direct endpoint position error was
0.0402 m, but the object was never stably grasped, so endpoint error is not a
success proxy. The physical arm therefore fails the gate and the combined Probe
is `UNPROMISING`; the offline fit remains `PROMISING` only as a label-space
diagnostic.

Artifacts:

- frozen PD-loss execution: `outputs/trajectory-policy/direct-tau-dagger-pd-execution-20261018-r1/`

The result narrows the diagnosis. Raw-action loss was genuinely misaligned with
PD space, and catastrophic rows dominate the autonomous roll-in, but fixing the
loss alone does not recover the task. Do not spend PPO budget or interpret this
as a tau-plan refutation. The next unresolved intervention is recovery-basin
collection with bounded teacher intervention (for example beta 0.1--0.25), with
phase/divergence masks retained for analysis.

## 2026-10-22 负结果 CPU 审计：结论范围修订

保留PD空间误差改善以及无干预learner0/8 hold45；40mm终点误差不等于抓持。fit实际manifest0dc48974、执行357e4a6d分别记录。本波teacher hold45=8/8但placement=4/8，实际共用runner要求teacher placement≥6/8，因而完整placement gate未得到合格校准；不能把UNPROMISING全部当作方法否决，更不能据此否定τ或证明只缺C。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
