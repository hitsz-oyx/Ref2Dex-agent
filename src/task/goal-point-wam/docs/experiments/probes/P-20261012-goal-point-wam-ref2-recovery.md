---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-recovery
experiment_id: P-20261012-goal-point-wam-ref2-recovery
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 20
seed_pool: probe
seeds: [292]
decision_changed_if_positive: admit live-state expert PD labels for one bounded frozen-R actor correction
decision_changed_if_negative: inspect expert recovery and goal observability before fitting learner-state PD targets
status: UNCLEAR
run_id: wam-ref2-execution-recovery-20261012-r1
---

# Expert recovery from the measured WAM approach state

## Decision Note

This serves the self-trained trajectory-policy baseline, not a Cm causal claim.
The startup head improved initial actual PD error225.27->11.71mm, but the
unassisted542-control closure still had zero held frames. Wrist deviation
crossed10mm at state9 and reached53.68mm at state16, before contact. Teacher
and GT-R each held441 frames and placed the object. Exact input/PD audits
passed. The next question is whether a live-state expert can recover this
early deviation, rather than treating nominal future trajectories as recovery
labels or repeating nominal offline fitting.

Root chooses one expert-takeover episode and concurrent live expert queries:
this is cheaper than training an actor against potentially unusable labels.
Use idle GPU2, one group<=600s, total existing/new ref2 artifacts<=100MB;
current shared storage296.51GB, below proactive298GB/hard300GB. No external
identity, source modification, real robot, video or new branch is needed.
If takeover recovers, root may define a separate PD-only correction protocol;
otherwise diagnose this takeover timing/teacher before training. Stop on
source/frame/reset/finite/PD/peer mismatch, resource or foreign GPU conflict.

## Fixed protocol

Four simulated roles: teacher, privileged GT-tau-R calibration, unassisted
WAM-R, and WAM-R until controls0:16 then official expert from control16.
The two WAM rows must share exactly equal reset and pre-takeover measured
state/action. One queried WAM tau is reused only during this equal prefix.
The recovery row thereafter uses its own live expert observation/action.
Keep the authenticated startup head2b7eff74991a8ef306b192747d716210d4348d3259f628fa73c78dfc7b617fc2,
human base, R, native physics/controller, object reference and actor period1
unchanged. Deliberately reuse physics seed292 for the same frame0 condition;
this is a paired diagnostic, not a fresh statistical replicate.

Query the official expert before each action on every row's actual native
observation. Save observation, raw expert action, clamped expert action,
expert PD reconstructed with that row's current measured q, plus actually
applied action/PD and measured q/dq/hand/object/velocity, learner H4/goal/tau.
Authenticity, native normalization and planned-reference expert semantics
must be checked. Future measured hand/q/object/contact never enters WAM.
Expert reference conditioning is training-time privilege, not autonomous
actor input. The official checkpoint is a query teacher, not a deployed
student weight initialization.

Any training admission applies only to finite, audited live-state expert
PD query rows. Mixed-policy actual future hand is not an expert tau target;
source nominal teacher future/PD is not the recovery label. Persist query
training eligibility explicitly after audit, independently of inference
evaluation roles. No training/checkpoint selection in this experiment.

## Predeclared decision

All roles complete542 controls without resets. Teacher and GT-R each require
held>=45 frames and controlled placement; otherwise UNCLEAR. With calibration
passed, takeover held>=45 plus controlled placement and clipping<1% is
PROMISING for recovery/PD-label collection. Zero takeover held is UNPROMISING
for this exact tick16 takeover; other outcomes UNCLEAR. Unassisted WAM is a
concurrent failure diagnostic. Teacher-assisted recovery is never counted
as autonomous WAM success, broader recovery or retargeter correctness.

## Results

Pending implementation and authenticated execution. No new training started.
