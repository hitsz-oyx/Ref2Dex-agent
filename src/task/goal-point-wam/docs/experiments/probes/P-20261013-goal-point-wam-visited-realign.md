---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-visited-realign
experiment_id: P-20261013-goal-point-wam-visited-realign
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: f3b5d771
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 42
seed_pool: probe
seeds: [292, 295, 296]
decision_changed_if_positive: retain fixed repaired R and aligned actor for expanded closed-loop coverage
decision_changed_if_negative: localize failure between recovery, inverse control, and actor trajectory before scaling
status: RUNNING
run_id: wam-visited-realign-20261013-r1
---

# Current visited-state coverage and fixed-R actor realignment

Result: Collection and bounded realignment pending.
Decision: Complete one collect/qualify/align/rollout loop, without larger models.

Decision Probe for C3 manipulation: is inadequate coverage of the current
actor/executor contact states blocking stable grasp, and can fitting the real
expert trajectories together with fixed-R commands improve closed-loop hold?
Prior R repair fits3 old C states but hurts current actor when swapped. Sparse
reference-wrist A/B is completed UNPROMISING, so keep ref2_8 actor4500 inputs.
Use this fixed actor with ref31 repairedR for collection and initial comparison.

Collection: controls128, takeovers32/38/44/50/56/62, four prefix families.
Family0 no perturbation; families1/2/3 add respectively +.5mm x/-.5mm x/+.5mm y
wrist PD target each control24..31 before clipping, after which the own-state
actor resumes. These are bounded exploratory actions, not new observations or
expert assistance. Expert takes over the paired command-replay row using its
own actual state; all future24/action8 labels come from that same row.
Family3 is held out wholesale, including its expert continuations.24 distinct
queries are temporally correlated within4 prefixes; only6 are unperturbed.
No unseen task/large-reset generalization claim. Record physical pair tolerance
failures and expert inability separately; never substitute another row's future.

Cheapest first wave is no-perturbation takeovers to locate recoverability; stop
adding repetitive data if the expert cannot recover most contact states or
prefix replay loses physical comparability. Bound all collection <=30min,
two idle GPUs3/4 concurrently, each group<=600s, outputs<=1GB. No foreign
process termination, no camera/Vulkan, no changes to input semantics.

Then evaluate frozen repairedR on all actual GT futures, reporting exact queries
separately from continuation. If needed fit R with nominal replay and TRAIN
families only <=3000 updates/900s; DEV family3 stays out of draws. Require
nominal/startup DEV <=1.1baseline+.1mm and recovery exact/contact improvement,
clipping<=1%; nominal128 GT physical check before candidate actor execution.
Holdout groups guide a Probe decision, not a formal test set.

Actor: start from ref2_8 selected4500, fix chosenR. Train decoder and existing
control adapter only, encoders/World frozen, <=3000 updates/1200s, batch100:
nominal30 +human20 +exact TRAIN takeovers25 +expert continuation25. Keep
trajectory MSE/.05 and fixed-R control-aware loss, World effect coefficient.1.
Initial and500-step panels: exact/first16 TRAIN and held-out family, nominal,
startup and human. Gate for a physical trial: exact mean first8 and PD error
both improve>=10%, heldout first8/PD not worse>10%+.1mm, nominal PD not worse
>10%+.1mm, human all24 not worse>10%+.1mm, clipping<=1%. Select lowest mean
trajectory/control ratio among passing candidates; no unbounded extension.
Only passing candidate gets a full542 rollout with hold45/drop/placement and
calibration checks. If offline gate fails, retain failure and diagnose before
spending another physical trial. New actor training identity must bind exact
R checkpoint; no use of the previous adapter's frozen-R claim for new training.

Decision Note: user explicitly authorized this bounded loop after reviewing
ref31. Main question/claim unchanged. Total <=2GPUh and <=3GB new artifacts,
shared400GB/max6GPUs. Stop nonfinite, admission/units failure, own resource
limits or two consecutive gross (>2x+.1) trajectory/control regressions.
No new branch, default deployed actor/R untouched, no remote push.

Future evidence: more diverse initial conditions, true per-row closed-loop
queries instead of physically comparable command replay, unseen motions, and
matched Cm causality remain deferred. Failure to recover after physical
contact may require different earlier interventions rather than more epochs.

## Collection correction: same-row takeover replaces paired command replay

Initial wave1 r1: t32/38/44 experts held73/75/75 with audited actual futures.
At t50/56 collection stops at47 controls: independent PhysX contact rows cross
rotation comparability threshold .001rad (observed .0010173), before expert
intervention. This is an implementation admission blocker, not unrecoverability.
No paired tolerance is relaxed. Preserve r1 but exclude it from the new cohort.

Add explicit --own-row-recovery: policy queries use actual recovery row3's H4,
q/dq/previous command and object frame, execute there until takeover, then query
and apply expert on the same row. Row2 is only a command-replay diagnostic;
it is no longer a pure autonomous control. Record roles/query ownership and
independently verify prefix actions (including declared perturbation), own H4,
R state and proprio. Same-row takeover requires no hidden-state fork or equality
to row2. r2 sources and follow-up data admission require this ownership flag.
This change strengthens the intended current-policy-state contract.
