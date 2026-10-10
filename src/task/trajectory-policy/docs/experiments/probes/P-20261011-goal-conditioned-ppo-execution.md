---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-ppo-execution
experiment_id: P-20261011-goal-conditioned-ppo-execution
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 6
seed_pool: probe
seeds: [314]
decision_changed_if_positive: retain the direct-action Stage 1 PPO variant and compare it against a residual/nominal adapter
decision_changed_if_negative: stop the direct-action Stage 1 route and obtain an explicitly training-eligible teacher or introduce nominal-plus-residual control before further training
status: PLANNED
run_id: goal-conditioned-ppo-execution-20261011-r1
---

# Frozen evaluation of the Stage 1 PPO actor

## Motivation and decision note

The preceding PPO continuation completed its bounded training budget but training
sampling had no long hold. This separate frozen execution Probe prevents the
training trace from being treated as a success claim. It reuses the same 16-env
native setup and goal-only hand/object packet as the BC evaluation, loads only the
final PPO actor, and records all measured geometry and dispatch values.

The actor receives the unchanged live-state plus eight-step hand/object-goal
contract. It has no future q, teacher command, phase, clock, force label or Cm
input. The comparison is to the earlier self-owned tracker calibration, not to
the engineering-only packet's forbidden actions.

## Fixed inputs and gate

* native input manifest:
  `outputs/consequence-evaluator/hand-execution-inputs-contact-capture-20261010-r1/inputs.json`;
* goal-only packet:
  `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`;
* geometry/reset contract:
  `outputs/consequence-evaluator/ref7_4-tau-geometry-20261010-r1/`;
* frozen PPO actor:
  `outputs/trajectory-policy/goal-conditioned-ppo-20261011-r1/final.pt`;
* one idle GPU, 16 environments and 542 controls, with a 300-second timeout.

The gate is at least 12/16 long-held terminal episodes and native clipping below
1%, matching the BC execution Probe. This remains a single-motion Probe and does
not establish generalization, REGRIND reproduction or Cm utility.

## Results

Pending frozen evaluation.

## Limitations and follow-up

If this actor also fails before a short hold, direct native-action BC/PPO is
closed for this motion. The next meaningful test is a nominal-goal plus bounded
residual adapter or a fresh rollout marked `training_allowed=true`; high-level
PointWAM/ACT forecasting remains deferred.

