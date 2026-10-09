---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-panel-trajectory-utility
experiment_id: P-20261010-history-panel-trajectory-utility
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 309d18d
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 4
seed_pool: probe
seeds: [412, 413, 414, 415]
decision_changed_if_positive: retain the trajectory evaluator as a useful offline ranking component and audit its matched controls
decision_changed_if_negative: keep evaluator/selector/PW online integration frozen and return to candidate-bank design
status: UNCLEAR
run_id: history-panel-trajectory-utility-20261010-r4
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

The fit has four matched-initialization arms:

* B: `H` only (tau input zeroed);
* T: `tau` only (H input zeroed);
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
`outputs/consequence-evaluator/history-candidate-panel-20261010-r2/` and
`outputs/consequence-evaluator/history-panel-trajectory-utility-20261010-r3/`.

## Result

The clean replay panel packet is
`history-candidate-panel-20261010-r2/`: 25/12/18 train/val/test panels
(385 rows), with all candidates within the recorded H/state tolerances. The
panel audit is
`history-candidate-panel-audit-20261010-r2/`; composition used no labels.

The matched-initialization B/T/C0/C1 fit completed 1200 GPU2 steps in about 40 s
with finite values. On the 18 held test panels (137 strict pairs across only 7
informative panels):

| arm | strict pair accuracy | tau-shuffle accuracy | mean regret |
| --- | ---: | ---: | ---: |
| B H-only | .7226 | .7226 | .07234 |
| T tau-only | .7007 | .4599 | .00637 |
| C0 H+tau | .6569 | .4234 | .00463 |
| C1 H+tau+E_GT | .5839 | .5693 | .09433 |

C0's shuffle drop is 23.36pp, but matched H-only B reaches .7226 while C0 is
only .6569 (`-6.57pp`). T tau-only reaches .7007 and its shuffle drop is
24.09pp, so tau has an exploratory signal but combining it with this
approximate H is not useful in the current architecture. C1 is .5839
(`-7.30pp` versus C0) and its shuffle drop is only 1.46pp; the exploratory
screen is false. The fit is therefore `UNCLEAR`, with the C0/C1 combined arms
`UNPROMISING` on this panel, not support for GT-effect information or a
deployable selector. C2/PointWorld, deadzones, online planning, and native R
execution remain closed. The panel's approximate-H tolerance and seven
informative test panels are evidence limitations; do not treat any single-seed
arm as a formal trajectory-ranking claim.
