---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-panel-trajectory-utility
experiment_id: P-20261010-history-panel-trajectory-utility
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: pending-clean-commit
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 4
seed_pool: probe
seeds: [412, 413, 414, 415]
decision_changed_if_positive: retain the trajectory evaluator as a useful offline ranking component and audit its matched controls
decision_changed_if_negative: keep evaluator/selector/PW online integration frozen and return to candidate-bank design
status: PLANNED
run_id: history-panel-trajectory-utility-20261010-r1
---

# Does an episode-split approximate-H panel let tau retain U32 ranking signal?

## Motivation and decision note

The history-preserving candidate Probe found full actor-observation near pairs
with non-tied frozen U32 labels in every split. That is enough to test the next
offline link in `完整链路.md`: a separate C0/C1 trajectory utility, without
PointWorld, online selection, or native R execution. The question is whether
the observed candidate tau branch improves ranking over H alone under a matched
initialization and a tau-shuffle control.

This is a Probe because the panel is only approximate-H: every candidate is
within a recorded H RMS tolerance of its anchor, not bitwise identical after a
fork. A positive result changes the priority of offline evaluator analysis; it
does not support a deployable selector or any native control claim.

## Frozen data construction

The source is the history-preserving structured episode split: 96 train, 64
validation, and 64 test episodes (seeds 412/413/414). At ticks 16/24/32,
candidate panels are built using only query-time distances

```
H RMS <= 0.03, hand RMS <= 3 mm, q RMS <= 0.03
```

Each panel has seven candidates. The six non-anchor candidates are selected by
farthest-point diversity in the 24-step current-object-frame hand trajectory;
U32 labels are not used for composition. The panel packet retains H, tau,
measured 24-step object effect, labels, panel IDs, and all match distances.

The fit has two matched-initialization arms:

* C0: `H + tau`;
* C1: `H + tau + E_GT`.

Validation selects checkpoints by episode MSE. The held test panel reports
strict pair accuracy, regret, C1 gain over C0, and a candidate-tau shuffle.
The exploratory screen is C1 >= .70, gain >= 3pp, shuffle drop >= 3pp, and
no worse regret; it is not a formal support gate because H is approximate.

## Stop conditions and boundary

Stop on panel completeness, split leakage, nonfinite values, or a failed
matched-initialization/hash contract. Do not fit C2/PW, add selector deadzones,
run online planner/MPC, or connect the result to native R in this Probe.

## Artifacts

Panel builder:
`src/task/consequence-evaluator/tools/audit/build_history_candidate_panel.py`.
Panel audit:
`src/task/consequence-evaluator/tools/audit/audit_history_candidate_panel.py`.
Fit entry point:
`src/task/consequence-evaluator/tools/run/train_history_candidate_utility.py`.
Planned panel and fit outputs are
`outputs/consequence-evaluator/history-candidate-panel-20261010-r1/` and
`outputs/consequence-evaluator/history-panel-trajectory-utility-20261010-r1/`.
