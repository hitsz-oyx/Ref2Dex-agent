---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-bc
experiment_id: P-20261011-goal-conditioned-bc
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 5699a2c6f919cf1561c7c9a2b06882f3559f6b68
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 3
seed_pool: probe
seeds: [311]
decision_changed_if_positive: freeze the goal-conditioned low-level BC policy and run a bounded physical hold Probe before PPO
decision_changed_if_negative: keep the executor contract, reject this label/interface combination, and obtain an explicitly training-eligible fresh teacher rollout
status: RUNNING
run_id: goal-conditioned-bc-20261011-r1
---

# Stage 1 goal-conditioned low-level BC

## Motivation and decision note

`ref2.md` makes the first learnable boundary a REGRIND-style low-level controller:
measured robot/interaction state plus a short future hand/object goal should produce
the native action. This Probe tests that interface before adding a high-level
`H -> tau_hat` model or PPO. The portable REGRIND idea is the closed loop—nominal goal,
bounded correction, measured-state feedback, native command—not its asset-specific
gains, reset distribution, phase variables, or 4096-environment configuration.

The old reactive teacher packet is explicitly `engineering_only=true` and
`training_allowed=false`; its actions are not used here. Labels come from the
self-owned successful tracker execution wave, whose manifest records an on-policy
PPO tracker and no imitation labels. The original retargeted motion supplies only
future hand keypoints and object poses as a fixed goal. This is a derived
distillation Probe, not a claim that the source packet is a training corpus. If the
provenance audit or offline result is insufficient, the next action is a fresh
rollout with an explicit `training_allowed=true` annotation.

## Student contract

The policy sees `q_t`, `dq_t`, measured hand keypoints, measured object pose and
velocity, the previous native command, and eight future hand/object goals. Goals are
expressed in the current object frame. No future robot q, teacher command, phase,
clock, contact force, or Cm value enters `goal_features`. The MLP emits the twelve
native active coordinates in `[-1, 1]`; dependent Inspire coordinates are masked and
the existing native adapter remains responsible for coupling and final clipping.

## Fixed inputs and budget

* trajectory labels: `outputs/consequence-evaluator/ref7_3-tracker-feedforward-final-eval-20261010-r1/trajectory.npz`, tracker rows only;
* trajectory manifest: the matching `manifest.json`, including its source hashes and
  the `training_contract: on-policy PPO only` declaration;
* goal source: the read-only hand/object arrays in
  `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`;
  its `actions` and q arrays are rejected and never enter the dataset. The packet
  remains explicitly `engineering_only=true, training_allowed=false` and is used
  only as a fixed privileged goal sequence, matching the Stage 1 GT-goal contract;
* native target semantics: the existing 18-DOF DExplore Inspire adapter and
  `trajectory_policy.goal_executor` contract;
* one deterministic seed, train/validation split by environment id, at most 3,000
  optimizer updates and 600 seconds of GPU time, with no output overwrite.

The offline gate is finite loss and a validation active-coordinate L1 no worse than
the fixed teacher-action scale threshold declared in the run result. Offline success
only permits a separate physical short-hold Probe; it cannot be reported as grasp
success. GPU state and elapsed time must be recorded during the fit.

## Results

Pending implementation and offline fit.

## Limitations and follow-up

This single-motion distillation set cannot establish generalization, contact preload
transfer, or a deployable policy. A positive offline result must be followed by a
fresh physical hold comparison against the existing frozen tracker. PPO and the
PointWAM-style high-level forecaster remain deferred until that gate passes.
