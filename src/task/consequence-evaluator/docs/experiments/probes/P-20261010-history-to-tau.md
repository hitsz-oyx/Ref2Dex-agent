---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-to-tau
experiment_id: P-20261010-history-to-tau
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 1aa4d1b
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 5
seed_pool: probe
seeds: [412, 413, 414, 416]
decision_changed_if_positive: retain an offline H-to-tau baseline for later integration audit
decision_changed_if_negative: keep H-to-tau, PointWorld, selector, and native execution frozen
status: UNCLEAR
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
Output:
`outputs/consequence-evaluator/history-to-tau-20261010-r1/`.

## Result

The bounded fit completed on GPU2 in 7.29 s at the 1200-step cap. The
history, current-hand, and target arrays were finite; the actor hash was
identical across the three episode splits, and the test split was not used for
checkpoint selection. The packet contains 6144/4096/4096 train/validation/test
windows.

| split | predicted point RMSE | persistence point RMSE | predicted H24 RMSE | persistence H24 RMSE |
| --- | ---: | ---: | ---: | ---: |
| train | .04328 m | .20065 m | .07022 m | .31118 m |
| validation | .06698 m | .19587 m | .10205 m | .30542 m |
| test | .30280 m | .26214 m | .47161 m | .40719 m |

The exploratory test screen is false: on the held seed-414 split, the model is
15.5% worse than persistence in point RMSE and 15.8% worse at H24. Training
and validation improve over persistence, but that does not transfer to the
held episode split. The input/episode/hash contracts and finite-value checks
passed, so this is retained as an `UNCLEAR` single-Probe result rather than a
formal negative claim about all H-to-tau policies. The offline baseline is not
carried into PointWorld, selector, online control, or native R execution.
