---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-recovery
experiment_id: P-20261012-goal-point-wam-ref2-recovery
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 7f516d59
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 20
seed_pool: probe
seeds: [292]
decision_changed_if_positive: admit live-state expert PD labels for one bounded frozen-R actor correction
decision_changed_if_negative: inspect expert recovery and goal observability before fitting learner-state PD targets
status: UNCLEAR
run_id: wam-ref2-execution-recovery-20261012-r2
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
WAM-R, and WAM command replay until controls0:16 then official expert from
control16. Reset is equal within the original1e-5 gate. The recovery row
replays the exact normalized native command from the unassisted row; its
PD target is independently reconstructed with its own current measured q.
It thereafter uses its own live expert observation/action. Replay is not
WAM inference on that row's H4; its tau placeholder is zero/invalid throughout.
Expert queries use its real state, not the other row or a forced state fork.
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

First r1 fixed5354774d FAILED after1 control because independent PhysX rows
are not bit-identical despite equal reset and commands: hand max7.75e-7m,
q4.77e-7, object matrix2.17e-5, velocity.001378m/s. Durable query/actual PD
audit passed with zero error. Preserve logs/partial; this is an overstrict
comparison implementation constraint, not evidence against expert recovery.

Root Decision: permit one r2 after command-replay semantics and physical
comparability checks. For measured states0:17 require qXYZ norm<=1e-4m,
other q max<=1e-3rad, hand point norm<=1e-4m, object XYZ norm<=1e-4m and
object rotation-matrix Frobenius/sqrt2<=1e-3 (an approximate angular quantity).
All dq/velocity differences are reported. Applied native command equality
remains exact, while both rows' PD must match their own current-q inverse.
These numerical comparison bounds are far below the measured54mm failure;
they do not make the independent rows exact hidden-state forks. No state
copy/reset, actor/R/source/physics/clipping change. Cost<=600s/100MB unchanged.
Only audited PD labels admitted; no new training has started.

R2 completed542 controls/543 states in139.13s. Teacher held438/placePASS,
GT-R251/placePASS, WAM0/placeFAIL/clipping4.059%; calibrationPASS. Expert
takeover held435, first stable tick99, clipping0, but intermediate loss1 and
terminal height+.310m/settled0: placementFAIL. Predeclared recovery remains
UNCLEAR and complete-recovery admission false. No full-phase correction fit.

Independent audits authenticate134 source paths and542 durable expert query
files. Root recomputed9768 array checks, all exact0: actual/expert native PD
inversion, preclamp/clamp, own q/current R state, own H4/planned goal, replay
commands and expert takeover. Pair hand max1.67e-6m, object XYZ2.71e-7m,
qXYZ1.33e-7m, rotation proxy2.21e-5, all declared bounds pass; velocity max
.002420m/s is reported rather than claimed equal. GPU2 released; outputs
42.13MB. Full source/durable query arrays are finite. This supports a local
early-recovery decision only, not autonomous policy success or full-task
teacher recovery.

Artifacts under `outputs/trajectory-policy/wam-ref2-execution-recovery-20261012-r2`:
`native/trajectory.npz`, `native/root_recovery_audit.json`, `collection_audit.json`.
