---
schema: ref2dex.probe.v2
experiment_id: P-20261018-direct-tau-dagger
task: trajectory-policy
branch: main
git_commit: pending
date: 2026-10-18
classification: Decision
status: RUNNING
hypothesis: Querying the official teacher on learner-visited unseen states and fine-tuning the repaired direct tau executor with the teacher's current-q PD target can repair the remaining closed-loop failure.
question: Does a DAgger-style learner-state correction improve direct tau execution on the same unseen s1_airplane_lift sequence?
decision_changed_if_positive: retain the reference tau plus live-state teacher-PD DAgger route as the next executor candidate before PPO.
decision_changed_if_negative: do not conclude that tau is unusable; distinguish action/executor contract limits from the need for a stronger interaction-conditioned policy.
protocol: Autonomous roll-in of the repaired r3 direct executor on 8 unseen direct environments. At each tick, query the official teacher on the same live direct observation, record its native action and current-q PD target, pair them with the independent reference tau for that source tick, then warm-start the r3 policy and train on equal original/query batches. Evaluation is a fresh frozen-policy run on the same unseen sequence with no teacher intervention.
query_semantics: reference tau remains tau_k = R_ref_t(h_ref[t+k] - h_ref[t]); teacher PD is computed from the official teacher native action and the live student q at the visited state. No future student hand is used as a tau label.
source: outputs/trajectory-policy/gt-tau-new-sequence-20261017-r4/teacher/trajectory.npz
base_checkpoint: outputs/trajectory-policy/direct-tau-bc-r3-20261018-r2/best.pt
gate: direct arm requires hold45 >= 6/8, success >= 6/8 and clipping < 1%; teacher is calibration only.
runtime: /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train, Torch 2.0.1+cu118, GPU 0 with pre-existing context only.
---

## Motivation and decision note

The repaired r3 executor has correct reference-frame tau wiring and temporal order,
but its BC policy achieved 0/8 hold45 on the unseen physical sequence. This probe
asks the narrower question suggested by DAgger: can the teacher supply the correct
current-state PD target after the learner leaves the teacher state distribution?
The cheapest test is one autonomous learner roll-in, one teacher query per visited
state, one warm-start correction, and one frozen re-evaluation.

The query is intentionally explicit. The source future hand supplies the independent
reference plan tau. The official teacher is queried on the live direct observation;
its native action is converted using the live q into a PD target and both values are
stored. This is a valid current-state PD label. It is not a claim that the teacher
has generated a new future hand trajectory from the off-track state.

## Planned audit

The roll-in must preserve the 16-row native layout: eight teacher calibration rows
and eight autonomous direct rows. The teacher action queried on direct rows must be
computed before the learner action is applied. The recorded PD target must exactly
reconstruct from the queried action, current q, and native controller offset/scale.
The query packet is split by environment, with five direct environments for fit and
three for selection. The base standardizer remains frozen. The corrected policy is
warm-started from the selected r3 checkpoint and receives equal original/query
batches; no PPO or C input is used.

## Results

To be filled after the roll-in, DAgger fit, and independent frozen execution.
