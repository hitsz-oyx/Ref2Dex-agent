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

Result:27 actual takeovers collected;19 successful recovery queries admitted;R repaired/GT-qualified;actor realignment running.
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

## First actual-row wave and adaptive sampling decision

r2 same-row takeovers32/38/44/50/56/62 all complete128 controls and audit.
Expert held74/75/73/3/0/5, zero clipping. Before44 this expert can recover;
50 onward it fails under the existing reference clock. This does not prove
physical irreversibility or impossibility for every controller. It changes the
cheapest next collection: concentrate the four prefixes at32/35/38/41/44/47,
retaining original late50/56/62 as diagnostic negatives. Only hold45 expert
continuations with unclipped exact future24 enter success-directed fitting.
Family3 remains entirely held out. Thus total collection can include27 states,
with eligible training count determined by actual recovery, never fabricated.


## ref3_2 steering and baseline scope

User asks whether paired robot fine-tuning is worth trying. Primary-paper and
local-history assessment is recorded in
[PointWAM joint audit](../../../../../../paper/pointwam/robot-joint-finetune-primary-audit-20261012.md).
The current collect→fixed-R→actor loop remains the bounded control; no new joint
arm or larger training is launched merely by this question. Both approaches
can reuse exactly the same same-row recovery corpus. Proposed next decision
comparison first removes frozen-World effect loss with R fixed, then tests
joint R updating with GT-tau mapping anchoring if still necessary. Changing
buffer execution or adding a shared scene head would be separate variables.


Collection wave3 f3/t38 aborts during startup: du observed a Triton temporary
cubin removed/renamed during traversal. This is an engineering false abort,
not policy failure; the worker consequently never ran f3/t44. Preserve both
the partial wave and failed trace. Launcher disk measurement now retries only
ENOENT (maximum3), rejects permission errors/persistent races and never admits
a partial du count. Two regression tests pass. Retry only38/44 in fresh r3
source directories; original physics, checkpoints and resource caps unchanged.


## Completed collection and independent R repair

Collection manifests (all under outputs/trajectory-policy):
`wam-visited-collection-wave1-20261013-r2`, `wave2-20261013-r1`,
`wave3-20261013-r1`, `wave4-20261013-r1`, where each abbreviated wave retains
prefix `wam-visited-collection-`. Wave3 admissions are combined with wave4's
engineering retry; the failed wave3 source is excluded. Total27 distinct
(family,takeover) queries.19 eligible: family0/1/2/3 count5/4/5/5; valid expert
windows335/274/313/335.14 TRAIN exacts,5 whole-prefix DEV exacts. Exclude8
hold45 failures; clip only removes affected future24 windows (family2/t41's
late clipping does not invalidate its unclipped exact/early windows).

| Prefix | t32 | t35 | t38 | t41 | t44 | t47 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none |74|73|75|74|73|1|
| +.5mm x |74|74|74|74|24|28|
| -.5mm x |73|74|75|75|59|12|
| +.5mm y (DEV) |74|74|73|74|75|16|

Entries are maximum expert held frames in128 controls, not autonomous success.
All sources audit actual same-row history/proprio/R-state, expert labels and
bounded declared exploration. Four prefixes are correlated, not4 independent
seeds. No evidence of unseen-reset generalization.

R run `outputs/trajectory-policy/wam-visited-R-20261013-r1` (code660804cd):
3000 updates65.05s onGPU5,~509MiB, select1500 by original passing gates then
mean ratio. Later checkpoints fail gates and are not deployed.

| R GT-input panel | Initial mean/p95 mm | Selected mean/p95 mm |
| --- | ---: | ---: |
| nominal DEV |7.584 /18.683|7.119 /17.173|
| startup DEV |1.076 /1.790|.973 /1.521|
| exact TRAIN |89.026 /174.699|2.022 /3.488|
| contact16 TRAIN |34.376 /140.846|13.742 /42.169|
| exact DEV |115.923 /210.387|62.636 /227.824|
| contact16 DEV |46.771 /202.715|34.067 /143.667|

DEV exact errors t32/35/38/41/44 are4.69/5.41/4.61/18.26/280.21mm: the last
contact state remains poorly covered. This passes the predeclared relative
non-regression gate, not a claim that every held-out state became controllable.
Nominal physical128 GT run `wam-visited-R-GT-20261013-r1`: teacher/old-R/new-R
hold73/75/75, zero clipping, GT_execution_gate true. Independent root audit
reconstructs every panel state/tau/actual command and native current-frame
execution; labels and native state/PD errors0, metric max3.34e-6mm. Auditor
uses the trainer's float32 quantile rank (float64 percentile differed by
.000217mm); tolerance and saved training metrics unchanged. Qualification
receipt binds this exact checkpoint. Single motion/short GT scope only.

Actor run `outputs/goal-point-wam/wam-visited-realign-20261013-r1` starts from
4500 control adapter/decoder with this exact qualified R frozen. Bounds,
sampling and gate remain as originally declared. User's ref3_2 question is
answered separately; no joint/effect-off arm launched yet.
