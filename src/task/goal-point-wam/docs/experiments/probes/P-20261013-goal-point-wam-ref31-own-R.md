---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref31-own-R
experiment_id: P-20261013-goal-point-wam-ref31-own-R
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 81956abb
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 40
seed_pool: probe
seeds: [292, 295, 296]
decision_changed_if_positive: qualify repaired R physically before freezing it for actor work
decision_changed_if_negative: preserve old R and diagnose own-state inverse-control coverage
status: UNCLEAR
run_id: wam-ref31-own-R-20261013-r1
---

# Chunk-R repair on actual C states

Result: Fitted GT-to-command contract PROMISING; direct frozen-actor R swap
UNPROMISING, held0 with late50.37% clipping. Overall deployment UNCLEAR.
Decision: Preserve repaired R as an inverse-control candidate, retain original
R default; do not scale robot adapter or silently swap the executor.

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

## Completed results and Decision Note

Fit81956abb,3000 updates/55.88s,GPU2 ~529MiB/30–40% utilization; selected3000.
Same old normalization/controller, no actor weights trained. Exact C takeovers
32/40/48 PD XYZ mm:32.820/58.948/159.934 ->1.736/1.840/1.367.
Contact first16 p95:33.997/171.628/223.146 ->8.322/16.478/27.275.
Nominal DEV18.011->18.683mm, REPORT19.512->19.386mm; startup DEV5.318->1.790.
All20 offline checks pass. These are3 fitted C states and192 continuations,
not evidence of unseen autonomous-state generalization.

First GT run r1 aborted after1 control when another session entered GPU2;
no foreign process stopped. Qualification re-run on idleGPU4 using configurable
GPU entrypoint at a0adcec2, output wam-ref31-own-R-GT-20261013-r2.
Teacher/old PD-R/repaired Chunk-R held73/75/75, zero clipping, partial128 gate
passes. Independent source/first-only/PD/outcome audit passes (metric rounding
max6.4e-6mm); qualification receipt records the new artifact explicitly.

Decision: The contract is repaired on retained samples and nominal GT is
preserved, so spend one <=600s full frozen-actor rollout to test compatibility.
Use explicit --allow-repaired-chunk and verified same training-R parent;
keep C/adapter training identity untouched and record new execution-R identity.
No implied permission to retrain or enlarge actor. Failure stops direct swap.

Full542 at18ad8d49, GPU4 ~18.8GiB,133.01s, native/server exit0. Output:
outputs/trajectory-policy/wam-ref2-execution-R31-20261013-r1.
Teacher/GT held440/438, both controlled placement true; GT clipping1.845%
means full-run strict <1% calibration fails despite nominal128 gate passing.
Actor held0, highest lift2.715mm, minimum gap.244mm; no clipping first128,
first clipping tick252, full50.369%. Thus late clipping is not by itself the
cause of initial grasp failure. Original frozen actor+oldR:held0,276.6mm transient
object flight,1.476% clipping. These separate physics rollouts are exploratory,
not exact paired hidden-state clones.13 native contracts pass,max8.84e-7.

Same-state cross diagnostic holds actor tau and observed state fixed:
oldR -> repairedR actor first PD at32/40/48:2.04->29.36,83.49->54.05,
33.71->74.14mm; actor contact16 p954.36->33.51,106.97->162.62,
111.95->161.59mm. GT-tau improves at all three. This directly demonstrates
that improving the GT contract need not improve the already adapted actor's
commands. It supports an actor/executor compatibility issue, without isolating
all causes of the closed-loop failure or proving one unique compensation story.
Slight differences from previous panels are TF32 evaluation batching/kernel
rounding; cross arms use identical16-query batches.

Preserve both R checkpoints. Next useful robot experiment is bounded semantic
re-alignment/actual visited-state coverage with the repaired R fixed, before any
large adapter expansion. User's current request did not require another actor
training round; data weighting remains a separate Probe41.

Evidence: fit/result.json, root_repair_audit.json, frozen_actor_query_cross.json;
GT r2/result.json; rollout/native/root_effect_rollout_audit.json and
rollout/comparison/old_vs_repaired_R.png. Original raw runs remain intact.
