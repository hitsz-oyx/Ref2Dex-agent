---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-startup-anchor
experiment_id: P-20261012-goal-point-wam-ref2-startup-anchor
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: ddcf7958
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 27
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: collect new actor states only after explicit startup control retention
decision_changed_if_negative: keep original startup actor and revise geometry to control compatibility before further rollout
status: UNCLEAR
run_id: wam-ref2-head-startup-anchor-20261012-r1
---

# Preserve startup while learning earlier true recovery geometry

## Decision Note

Task goal is robot autonomous-state aggregation preceding consequence/RL.
Geometry-only training improved overall nominal nearest8 17.22->14.06mm,
but first-step original-R PD12.12->24.72mm and tick2 wrist h1 5.76->14.68mm.
Actual new actor held0/192 and clipping8.33%; independent review found no
label/anchor/order/identity/R/native mismatch. Whole-panel gates concealed
immediate control regression. Actual expert takeover8 can recover local
grasp; original R's true expert future mapping remains imperfect (all25.10mm,
early8:16 p9530.12 vs same-tick teacher4.59). Do not infer capacity failure.

Root chooses ONE fixed200-update anchored geometry probe from ORIGINAL
startup head, not the failed geometry head. The question is whether earlier
actual support can be learned while preserving the original actor startup
function/control. It changes whether a new short actor-state collection is
worthwhile; another broad thaw, R override or RL is not justified yet.

Cheapest test reuses all saved native data; GPU2, smoke1 then fixed200/final,
<=600s/stage. This NEW probe adds<=20MB including scratch; cumulative ref2
<=120MB, global CAMPAIGN<=300GB (outputs+tmp292.097GB at preregistration).
The previous card26 group remains FAILED its100MB guard; this new explicit
allocation does not retroactively change that status. All old formal evidence
retained; only regenerable root-owned Triton scratch was removed. No new
branch/external modification/GPU expansion or global budget exception.
Stop on source drift, nonfinite, foreign GPU2 or either budget. If startup
retention fails, no new physical rollout; record the valid local result.

## Scoped supervision and fixed recipe

Source: complete192/t8 native trajectory8513480747187963e2d9b5a16d0dbde2997bcab5bbf5e7e0fb7545d7c1cf8259,
root3468/max0/finite PASS. Enclosing resource group failure is preserved.
Independent completed-native measurements, not group success, establish
finite local expert data. Derive NEW actor-specific admission of role3
query8..168 (161windows): all next24 actually applied expert controls,
unclipped, current-q PD reconstruction, own measured H4 and planned goal.
Preserve original blanket actor-future/mixed-future denials; never borrow
role3 futures as pure-row2 expert targets or claim complete placement.

Warm original startup head2b7eff74991a8ef306b192747d716210d4348d3259f628fa73c78dfc7b617fc2.
Same archived human model/original R; scene/goal/action/world/R frozen.
Train trajectory+hand_query only. Nominal8304 windows/env0:16 replay;
nominal64 fixed panel/env24:28 report. Batch128=nominal64 (startup32 with
tick<16, later32) + true recovery64. Cache original startup actor tau on
nominal states. Fixed loss: nominal true-tau Huber/.05 + recovery true-tau
Huber/.05 + 4*original-startup-output Huber/.05 on nominal startup subset.
The anchor is old actor distillation, never falsely labelled expert motion.
No R/PD gradient. Fixed200, lr1e-4/wd.01/clip1; one recipe, not weight search.

## Evaluation and collection gate

Both recovery fit and nominal replay observations are development evidence.
No recovery holdout or independent generalization claim. Report full24,
nearest8, h1/wrist h1 and original-R PD/clipping separately. Pure-row2
only actual current expert PD, no future hand labels. Explicit repeat-H4
(ticks0/1/2) vs actual H4 startup diagnostics; frame0 separately.

Collection requires nominal nearest8<=21.522mm, original-R PD<=50mm,
clip<1%, finite and unchanged frozen/source identities. ADD fixed startup
retention gates versus this original warm-head initial panel: frame0 PDp95
<=1.5initial, repeat-H4 PDp95<=1.5initial and repeat-H4 wrist-h1<=1.25initial.
These are predeclared BEFORE fit; oldcard26 gates/results remain untouched.
If positive, at most one192-step/takeover8 fresh actor-state collection with
original R, bounded within the new20MB allocation. Short local grasp only;
complete placement null. RL/world scientific training remains deferred.

## Limitations / future evidence

Same reset/one seed and previously inspected native data are not formal
validation. Anchor retention does not prove recovery controls or human/world
quality. Actual executed hand world proxy is feedback affected, not native
command causal dynamics. Purged world-only train/late/controller protocol
reviewed; defer it until this actor-control blocker has a useful signal.

## Results

Result: pending.
Decision: implement the fixed anchored probe and inspect startup retention first.

## Fixed200 result and bounded fresh-state collection

Code767e9ec2 fixed200 complete100.45s/GPU2. Nominal nearest8
17.217->17.072mm/PD25.556->24.093mm; frame0 PD12.046->11.855mm,
repeat-H4 PD12.046->11.855mm/wrist h1 11.303->11.399mm. All fixed
startup+nominal collection gates pass. Recovery TRAIN nearest8
18.094->14.175mm/all24 25.333->16.984mm, PD45.084->44.981mm.
Original pure-row2 queried PD945.25->942.57mm remains high.
Frozen scene/goal/action/world/R unchanged, cache-direct0/R no-backward,
200 sampled batches verified32/32/64 and no unlabelled future draw.
Raw metrics independently recomputed difference0. Original-head cached128
vs report smaller-batch BF16 tau coordinate differences1.953/3.906mm
are reported, not falsely claimed bit-identical; weights/source hashes match.

Resource Decision: retained116.146MB leaves too little for a192 collection
under120MB; remove only new root-owned regenerable Triton compile cache
after idle GPU/no bound-cache check. Keep all formal training/raw evidence.
Reduce planned new collection to128 controls/takeover8 (still one collection),
which includes the original teacher's early local hold phase but NO placement
verdict. This is a prospective budget-driven collection shortening, not a
change of observed pass metric. Root checks final incremental<=20MB and
aggregate<=120MB; launcher7c787527 uses120 only for this explicit experiment,
all original experiment limits stay100. Check local calibration and earliest
PD/wrist drift on fresh actor actual states; no unseen/generalization claim.
