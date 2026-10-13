---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref31-own-R
experiment_id: P-20261013-goal-point-wam-ref31-own-R
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 9cfb13bd
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 40
seed_pool: probe
seeds: [292, 295, 296]
decision_changed_if_positive: qualify repaired R physically before freezing it for actor work
decision_changed_if_negative: preserve old R and diagnose own-state inverse-control coverage
status: RUNNING
run_id: wam-ref31-own-R-20261013-r1
---

# Chunk-R repair on actual C states

Result: Pending bounded fit and privileged GT execution qualification.
Decision: Freeze the existing actor; prioritize its inverse-control contract.

Decision Probe serving C3 self-trained manipulation: can current R represent
actual expert commands given own measured C states and their own actual expert
future hand trajectories? A positive fit triggers GT physical qualification;
a negative fit changes the inverse-control coverage/representation decision.
Cheapest test reuses three audited128-control C takeovers at32/40/48. These
provide exactly3 initial C states and192 expert continuation states, not195
independent autonomous states. Exact states and first16 panels reported separately.

Warm-start qualified wam-v2-control-fit-20261012-r1/chunk.pt, preserve model,
normalization and control-aware objective. Batch256:64 nominal startup,64 later
nominal,64 exact takeovers,64 continuation. Adam1e-4,3000 updates, evaluate250.
Nominal TRAIN env0..15,DEV16..23,REPORT24..31 unchanged. Select only on DEV and
fitted recovery panels; REPORT is descriptive once after selection.
Gate: each exact and contact16 XYZ p95 <=.9 baseline; nominal/startup DEV
<=1.1 baseline+.1mm; their rotation/finger <=1.2 baseline+.01rad; clipping
<=baseline+.01 and recovery <=.01. Select lowest mean of6 relative recovery
errors among passing candidates, else lowest relative error for diagnosis.
No unseen recovery holdout exists. Fit success alone is not generalization.

One idle GPU2, fit <=900s and <=1GB output. GT qualification <=600s, existing
128-control teacher/old-R/new-R contract, >=45 held frames and >=80% control,
clipping<1%. Full frozen-actor rollout only after nominal qualification,
with explicit new executor provenance. Total new artifacts <=2GB and global
outputs+tmp<400GB. Stop for nonfinite tensors, data/units drift, foreign GPU
process or deadline. No new branch, no overwrite of previous results.

Inputs: outputs/trajectory-policy/competent-pd-teacher-20261011-r2;
outputs/trajectory-policy/wam-ref2-execution-C-t{32,40,48}-20261013-r1/native.
Outputs: outputs/trajectory-policy/wam-ref31-own-R-20261013-r1 and
outputs/trajectory-policy/wam-ref31-own-R-GT-20261013-r1.

Limitations/future evidence: only3 fitted C states; nominal replay cannot prove
new autonomous coverage. Actor may have compensated old R, so replacement
can change closed-loop behavior even with better GT-tau fit. New own-state
collection and matched causal input/Cm tests remain deferred.
