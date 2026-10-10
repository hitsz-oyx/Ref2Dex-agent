---
schema: ref2dex.probe.v2
probe_id: P-20261011-current-history-tau-forecast
experiment_id: P-20261011-current-history-tau-forecast
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: PENDING
claim_id: C3
hypothesis_family: HF-trajectory-policy-history-to-tau
probe_index_in_family: 10
seed_pool: probe
seeds: [317]
decision_changed_if_positive: retain a current-source pure-H displacement forecaster as the Stage 2 proposal component
decision_changed_if_negative: do not connect Stage 2 to projection/physics; obtain fresh coverage or revise the history target contract
status: PLANNED
run_id: current-history-tau-forecast-20261011-r1
---

# Current-source pure measured-history H to tau forecast

## Motivation and decision note

Stage 1 low-level student Gates failed, so ref2 permits advancing to the later
PointWAM-style proposal layer without claiming a controller. An earlier clean
measured-history displacement forecaster was positive on a different
history-preserving source, but cross-source replay on the current execution wave
was worse than persistence. This Probe tests the cheapest discriminating question:
does the same H300 contract predict the current retained tracker wave when trained
and evaluated on its own environment-held-out rows?

The decision is forecast-only. A positive result permits a new card for
H-to-tau-to-projection; it does not authorize URDF projection, native execution,
PPO or Cm conclusions. A negative result leaves Stage 2 disconnected until
fresh teacher/reference coverage is available.

## Contract and protocol

Use only the 16 `role==tracker` environments from the retained tau-finger final
evaluation. The input is `proposal_history.condition`: four measured states
`t-3:t` containing hand/object geometry, finger q/dq and object velocity in the
current object frame. At ticks 0--2, frame 0 is repeated for the missing history.
The target is measured hand `t+1:t+24` transformed by the current object pose;
the matched displacement arm predicts target minus current hand. No future q,
future object, native action, phase, clock, contact or Cm enters H.

Fixed environment split is the first 10 tracker envs for train, the next 3 for
validation, and the final 3 for test. Fit matched absolute/displacement MLPs
(width 256, identical initialization, AdamW, 1,200 updates, seed317) and choose
each arm by validation L1 only. The predeclared offline screen is displacement
test point RMSE at most 90% of current-hand persistence and strictly lower H24
RMSE. No physical run is attached to this card.

Output: `outputs/trajectory-policy/current-history-tau-forecast-20261011-r1/`.

## Results

Pending execution.

## Limitations and stopping rule

This is one motion, one retained tracker wave and an environment-held-out
offline forecast. A positive result does not establish feasible tau projection,
closed-loop execution, generalization, self-trained grasping or Cm utility. Keep
the existing cross-source result and do not select a checkpoint using future
labels or physical outcomes.
