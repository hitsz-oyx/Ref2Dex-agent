---
schema: ref2dex.probe.v2
experiment_id: P-20261018-direct-tau-dagger-pd
task: trajectory-policy
branch: main
git_commit: pending
date: 2026-10-18
classification: Decision
status: PROMISING
hypothesis: The previous DAgger negative was partly caused by optimizing raw active action instead of live-state PD targets; a PD-space loss should improve teacher correction without changing the tau/state contract.
question: Does warm-start DAgger with normalized wrist/finger PD Huber loss materially reduce learner-state teacher-PD error, and is the improvement sufficient to justify one frozen physical replay?
decision_changed_if_positive: run one frozen physical replay of the PD-loss checkpoint; only continue to recovery-basin collection if the physical gate improves.
decision_changed_if_negative: stop this objective correction and treat the remaining issue as roll-in/recovery distribution or executor-state representation, not as evidence against tau semantics.
protocol: Reuse the existing 3448 learner-state queries without new physics collection. Decode policy active action through the native current-q controller into PD targets; optimize equal original/query batches with normalized wrist/finger PD Huber plus 0.1 action Huber. Keep the r3 standardizer and temporal executor fixed. Query environments 0-4 are training, 5-7 are validation. Also stratify rows by measured student/reference hand and object divergence.
query_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/dagger_queries.npz
rollin_source: outputs/trajectory-policy/direct-tau-dagger-rollin-20261018-r2/trajectory.npz
base_checkpoint: outputs/trajectory-policy/direct-tau-bc-r3-20261018-r2/best.pt
---

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
