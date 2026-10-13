---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-student-pd
experiment_id: P-20261013-goal-point-wam-student-pd
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: ea2575aa
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 45
seed_pool: probe
seeds: [292, 295]
decision_changed_if_positive: keep this reference contract and collect the next student round from the updated actor
decision_changed_if_negative: stop adding epochs to this trajectory-R pair and reconsider the intermediate representation or the executor
status: UNPROMISING
run_id: wam-ref2-execution-student-pd-eval-20261013-r2
---

# One student-state action round on the successful reference contract

Keep the three-domain initialization, the hand-trajectory representation, and
the competent PD retargeter that already grasps. Reference input is the
demonstration hand and object pose at t+1 and t+16, not a sparse wrist point
and not the robot's future state. Effect loss is off. The pass condition is
an autonomous hold of at least 45 frames, not a millimeter fit.

## Decision Note

The current closed loop does not grasp. The reference-conditioned PD DAgger
baseline does, and its useful correction was an expert command queried on the
states the learner actually visited. Post-failure hand motion was not treated
as the correct trajectory. This round labels the current airplane actor the
same way, then adapts only the robot adapter and trajectory head through the
frozen competent R. Nominal and human windows may supervise a real hand
trajectory. Student windows supervise only the expert's current PD. One
collection, one fixed training budget, one autonomous rollout. No R swap
under the old actor, and no effect-on/off comparison until a grasp repeats.

Collection uses the admitted realign actor and visited R so the states are
the ones that policy visits. Training does not start from that pair.

## Completed result

Collection `wam-ref2-execution-student-pd-20261013-r1` recorded expert PD on
all 542 actor states and stored no future-hand label. The actor of that
collection held 0. Training `wam-student-pd-20261013-r1` ran 1000 updates on
GPU0 in 127 s, peak 478 MiB. Effect loss stayed off, R stayed frozen, and
scene/World weights stayed frozen. The selected checkpoint is step 250 by the
lowest late-student wrist error, 615.5 mm to 611.7 mm. That split is ticks
400–530 of the same failed episode. On earlier visited states the encoded
wrist command also stayed large: tick 0 moved 311 mm to 304 mm, tick 16 moved
85 mm to 57 mm.

Autonomous rollout `wam-ref2-execution-student-pd-eval-20261013-r2` used that
checkpoint, the 408-D demonstration reference at t+1/t+16, the object goal at
reference t+16, and one command from the frozen competent R. GPU0, 125 s,
about 18.6 GB. The execution audit passed with maximum contract difference 0.
Teacher held 435 and placed. GT tau through the same R held 438 and placed.
The adapted actor held 0, clipped 9.6%, minimum gap 2.4 mm at tick 107, and
peak object height 19 mm. The object stayed table-supported for the whole
episode.

Local label UNPROMISING for this round. It is one airplane reset, seed 292.
The student labels come from the previous actor, so they do not cover states
this updated policy would visit later. The same update is not extended by
more epochs or by swapping R. Effect loss remains unmeasured.
