---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-ppo
experiment_id: P-20261011-goal-conditioned-ppo
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 5
seed_pool: probe
seeds: [313]
decision_changed_if_positive: continue Stage 1 with a frozen PPO low-level actor and run a separate frozen evaluation
decision_changed_if_negative: stop this direct-action continuation and obtain a fresh training-eligible teacher or add an explicit nominal/residual adapter
status: PLANNED
run_id: goal-conditioned-ppo-20261011-r1
---

# Stage 1 goal-conditioned PPO continuation

## Motivation and decision note

The goal-conditioned BC fit has low validation action error but its frozen physical
execution loses hold in the first few controls. The independent audit shows that
the native executor is wired correctly and that the student leaves the training
feature manifold after its first action. Ref2 explicitly places PPO after BC, so
this bounded continuation tests whether measured-state on-policy updates can
repair that covariate shift while preserving the same goal-conditioned contract.

The actor starts from the BC network. It samples a bounded native action through
an active-coordinate tanh mask. The measured reward contains current hand/object
goal tracking, physical proximity/lift/hold/loss and action-rate terms. It does not
read future q, teacher commands, phase, clock, contact labels or Cm values. This
is the direct-action Stage 1 variant; it does not silently claim REGRIND's
joint-reference gains or residual architecture.

## Fixed inputs and budget

* native input manifest:
  `outputs/consequence-evaluator/hand-execution-inputs-contact-capture-20261010-r1/inputs.json`;
* goal-only reference packet:
  `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`;
* BC initialization:
  `outputs/trajectory-policy/goal-conditioned-bc-20261011-r3/best.pt`;
* geometry/reset contract:
  `outputs/consequence-evaluator/ref7_4-tau-geometry-20261010-r1/`;
* one idle GPU, 16 environments, 24 updates of 128 controls, and a 480-second
  timeout. The run writes fresh outputs and logs GPU utilization, memory and ETA.

The PPO run is engineering/training evidence only. It is not a frozen evaluation
and cannot establish grasp success. A positive training result requires a separate
frozen execution Probe with the same observation contract; a negative result stops
this direct-action variant and motivates a nominal/residual adapter or a fresh
training-eligible teacher rollout.

## Results

Pending PPO continuation.

## Limitations and follow-up

This single-motion, single-seed Probe does not test high-level forecasting,
generalization, failure-bin sampling, matched Cm-on/off training, or the full
REGRIND reference-tracking configuration. It only decides whether on-policy
adaptation is worth a separate frozen evaluation.

