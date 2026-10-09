---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-to-tau
experiment_id: P-20261010-history-to-tau
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: pending-clean-commit
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 5
seed_pool: probe
seeds: [412, 413, 414, 416]
decision_changed_if_positive: retain an offline H-to-tau baseline for later integration audit
decision_changed_if_negative: keep H-to-tau, PointWorld, selector, and native execution frozen
status: PLANNED
run_id: history-to-tau-20261010-r1
---

# Can history predict a short current-object-frame hand trajectory?

## Motivation and decision note

The history-preserving candidate bank established that full actor observations
can be retained while constructing episode-separated candidate panels. The
next later-chain link in `完整链路.md` is an H-to-trajectory policy. This
bounded offline Probe asks whether the current actor observation plus the
current 11-point hand pose can predict the next 24 measured hand poses in the
current object frame. A positive result keeps a simple H-to-tau baseline for a
future integration audit; a negative or unclear result leaves the H-to-tau,
PointWorld, selector, and native routes frozen.

This is not a control or ranking experiment. It does not use R, PointWorld,
candidate labels, contact force, or online execution.

## Protocol

Inputs are `actor_observation[t]` (1442 values) and the measured hand pose at
`t`, transformed into the object frame at `t`. The target is the measured hand
trajectory at `t+1:t+24`, all transformed using the object frame at `t`.
Windows start at tick 8 and use stride 8; train/validation/test remain the
96/64/64 episode split from the history-preserving rollouts (seeds 412/413/414).
The probe fits a width-256 MLP for at most 1200 steps or 600 seconds on GPU 2,
selecting only on validation L1. Baselines are current-hand persistence and
the train-target mean. The test screen is exploratory: predicted point RMSE
must be at most 90% of persistence and predicted 24-step RMSE must be lower
than persistence. Even if that screen passes, the result remains an offline
prediction observation and cannot reopen R/PW/selector/native work.

## Stop conditions and boundary

Stop on missing history-preserving fields, split leakage, nonfinite values,
actor hash mismatch, or a failed bounded-fit contract. Do not add action
inputs, fit on test data, run a native behavior screen, or connect the model
to the online planner in this Probe.

## Artifacts

Model:
`src/task/consequence-evaluator/src/consequence_evaluator/history_to_tau.py`.
Fit entry point:
`src/task/consequence-evaluator/tools/run/train_history_to_tau.py`.
Planned output:
`outputs/consequence-evaluator/history-to-tau-20261010-r1/`.

## Result

Pending bounded GPU fit.
