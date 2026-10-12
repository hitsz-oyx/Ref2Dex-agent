---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref12-methods
experiment_id: P-20261012-goal-point-wam-ref12-methods
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: e6e42832
claim_id: C3
hypothesis_family: HF-goal-point-wam-ref12-methods
probe_index_in_family: 1
seed_pool: probe
seeds: [280, 281, 282]
decision_changed_if_positive: retain useful geometric method for forward reliability or goal-conditioned action work
decision_changed_if_negative: stop the tested local adaptation and preserve the running large-scale substrate
status: UNCLEAR
run_id: ref12-methods-20261012-r1
---

# ref1_2 intermediate literature methods on remaining GPUs

Class: Decision. Serves the Task's goal-conditioned hand/forward representation,
upstream of Mission's robot/Cm learning claim. User explicitly asks to try methods
in the body of ref1_2, beyond its final goal-injection recommendation. Existing
memory/query and world-on/off screens are not repeated; the main four-rank
10000-update run is preserved. Methods are geometric adaptations, not paper
reproductions. [Primary-source mapping](../../research/2026-10-12-ref12-method-probe-map.md)
records which mechanisms and action semantics the papers actually use.

Result: UNCLEAR; four full-batch engineering smokes and the corrected auxiliary/closed-loop wiring checks passed; scientific methods are launching.
Decision: Use bounded single-seed method Probes and cached-H world diagnostics to identify the next useful mechanism.

## Decision Note: remaining-card authorization

Latest user authorizes remaining cards while main DDP occupies GPU0--3.
Temporarily allow up to8 owned GPUs for this round: method jobs exclusively
idle GPU4--7; never signal unknown jobs or modify the main run's monitored
source/config/data/checkpoints. Cap90min additional wall,360GPUmin,8GiB new
artifacts; global outputs+tmp still<300GB,298GB proactive stop. Restore normal
Campaign4-card boundary when these extra jobs finish. No new external access,
robot execution, large downloads, multi-seed or main-run control is introduced.
The current initial artifact total is about283.1GB, leaving bounded headroom.

## Questions, methods and decisions

1. **ViGAR/SelfWAM inspired target ROI forward training (GPU5).** Does the
   existing executed-action W learn dynamic objects better when supervision
   emphasizes the current target4:1 over context objects? Keep existing motion
   weights, multiply the target object's weights by4 and normalize by actual
   weights. Train scene/action/world only; actor/goal modules are frozen, actor
   inference is not an outcome of this arm. Target is the current anchor flag,
   never selected by future motion. This is geometric target weighting, not
   image ROI or robot-self-mask reproduction. Positive dynamic/target error
   reduction supports investing in forward training before ranking/L_goal.
2. **Achieved-prefix goal diversity (GPU6).** For each sampled H choose
   h uniformly in8/16/24, obtain G from its actually achieved object transform,
   explicitly add learned horizon conditioning to actor queries, and mask actor
   supervision after h. Targets before h are the matching recorded action;
   no arbitrary endpoint substitution. Independent goal-blind W still receives
   the complete actual24-step executed action and complete matching effects.
   Evaluate actor at h24. This is horizon-conditioned prefix augmentation,
   not evidence of same-H/same-h counterfactual action diversity. A promising
   actor/goal response motivates richer goal coverage; failure only concerns
   this prefix adaptation.
3. **Act2Goal inspired intermediate geometric plan (GPU7).** H/G condition a
   one-layer planner decoder with24 target-object/time queries, predicting
   rigid object poses supervised by the recorded intermediate object points.
   Planner features, not GT intermediate poses, append to actor memory; actor
   predicts human trajectory. Joint loss=BC+world+.5*plan, paired32-point
   plan Huber scale.05m. Zero-initialized pose head starts at identity. This
   tests dense goal-conditioned intermediate supervision plus predicted-plan
   access, not the author's pretrained visual diffusion/staged recipe. Positive
   plan accuracy and actor utility motivate staged plan learning.
4. **InternW0/FastWAM inspired future-change auxiliary (GPU4).** Observed object
   tokens plus time embeddings predict adjacent object-point-center changes.
   Labels are measured rigid-point trajectories, loss .25*Huber(change/.01m)
   added to BC+world. Future changes are training targets only; actor reads no
   realized future. This is H-only training-time change supervision through
   shared representation, not full Causal Imprint tokens/4D distillation. Report
   its change prediction versus zero change and actor utility independently.

All four: model-only warm-start the real memory+hand/world1 final1200 weights
from `hand-goal-interaction-20261012-r1/memory-world1/final.pt`, fresh AdamW,
lr.0003/100warmup/cosine tenth/decay.01/clip1/BF16, batch64,3000 additional
updates (192000 occurrences/arm), same source/category recipe and train draw281.
Seed280 initializes new heads/augmentation; there are no independent seeds.
Core parent architecture/weights are strictly checked; only declared new heads
may be missing. New implementations live under tools/run, preserving all live
DDP model/base-trainer hashes. Model-only warm start does not carry old optimizer.

## Ranking and closed-loop checks

Cached-H audit on the parent examines recorded/static/half/1.5x/actor actions,
all using the local9D contract: XYZ, displacement from current hand, step
displacement (NOT velocity/acceleration). Missing previous points invalidate
step-displacement supervision. Repeat the world decode with exactly cached
scene/action to separate numerical noise; compare real recorded-action moving
world error to identity and report W-to-action gradient norms. Recorded action
is an oracle diagnostic candidate, never a deployable choice. Unexecuted
candidates have no counterfactual physical truth; lower surrogate distance is
not verified object improvement. This adapts DreamTrajectory ranking semantics.

A separate2-step engineering smoke tests BC+.1*endpoint consistency through
frozen scene/action/world. Check goal-only gradient reaches actor and frozen W
receives no parameter gradients; freeze buffers/stochastic behavior too. A
successful smoke establishes wiring only. If forward reliability remains weak,
do not promote it to large actor optimization: prioritize forward/plan arms.
Frozen W alone cannot prevent exploiting model error; measured candidate action
execution is future evidence before treating L_goal as a physical constraint.

## Evaluation, stopping and conclusion

Freeze768 unique windows (256/source), seed282, as a random subset of the
already frozen main DDP exploration panel. Attempting a fully new temporal
panel failed before model evaluation because an eligible stratum was too small;
retain this failed attempt and do not pretend the replacement is unseen.
This shared val/PW-selection evidence supports Probe decisions only. Within
panel windows may overlap. Evaluate every1000 and fixed final3000; no best
checkpoint selection. Cache scene for real/static goal ablations. Report by
source: actor EPE/static-goal/zero-hand, moving world error, target-object
error; planner dense point error/static and change-head EPE/zero as applicable.

A named actor adaptation is PROMISING if final real-goal EPE is>=5% below the
same frozen parent's evaluation on>=2 sources and improves on its own static
and zero references on those sources. A forward/auxiliary mechanism is
PROMISING if its own measured dynamic or dense target error beats the appropriate
static/zero reference by>=5% on>=2 sources. Otherwise UNCLEAR; no matched
same-duration continuation means extra training and mechanism cannot be causally
separated. The result decides whether to continue a mechanism, not architectural
superiority. Single-seed and shared-panel limits remain even with larger batch.

Before scientific launch full batch64 two-update smokes must pass: finite
loss/gradients, strict parent transfer and cached actor independence from altered
executed-action/effect labels. Stop own jobs on OOM/nonfinite/input drift/foreign
GPU conflict/deadline/artifact threshold; signal SIGUSR1 then180s cleanup grace.
Only own rolling latest checkpoint is replaced; final is a hardlink, preserving
parent and main-run checkpoints. No source/config editing while guards run.

## Artifacts and deferred evidence

Root: `outputs/goal-point-wam/ref12-methods-20261012-r1/`; `evaluation/` binds
indices, `plan.json` exact configs/devices/inputs, `smoke/` engineering records,
`action_world_parent.json` cached-H audit, `training/` guarded actual launches,
per-arm progress/curves/checkpoints/final goal/auxiliary metrics. Source/PID hashes
and actual Git commit are saved at launch; fixed implementation commit is
recorded after preparation. Scientific training is pending.

Deferred: independent subjects/panels, multi-seed causal comparison, real
candidate execution, faithful video/latent paper reproductions, detached 4D
teacher feature distillation, robot adaptation and final Cm-on/off RL evidence.
None is established by this round.

## Engineering preparation result

Commit1edaafba full batch64 two-update smokes passed all4 methods; losses and
gradients finite, cached actor label-leak max0. The change head initially emitted
large random increments, making its auxiliary loss dominate (26.65 before .25
weight). Its final layer is now zero initialized to start at the zero-change
reference; preserve the original smoke and rerun only the affected arm. A frozen
world consistency smoke is also queued separately. These are wiring results only.

## Decision Note: action-world findings before scientific launch

Parent cached-H768 audit completed: repeated world noise0 on all three sources,
nonzero world-to-action gradient; moving world/identity EPE respectively
16.31/25.97mm (OakInk2),113.38/163.21mm (GRAB),78.69/108.59mm (ARCTIC).
Thus W uses actions and improves on identity on this panel, but absolute errors
remain large. Recorded action beats static-action predicted goal distance in
41.4/52.7/48.0% of windows; recorded action is ranked first in only
10.2/14.1/13.7% of the five diagnostic candidates. These are surrogate rankings,
not measured outcomes of the alternatives, so they do not establish physical
correctness or falsify the recorded action's validity.

Choose forward ROI, legal goal diversity, intermediate plan and future-change
training now; keep the successful frozen-W goal-loss gradient check as engineering
only, with no3000-update closed-loop launch. Further ranking/direct optimization
needs targeted execution or stronger forward reliability. This changes investment
priority without closing the general world-model idea. All input/source/storage
bounds remain as above. New-head corrected smoke loss .01236 rather than26.65;
frozen-W goal-only gradients reach actor and its parameters receive no gradients.

Implementation e6e42832, model-only parent step1200; frozen parent actor EPE on
this same panel19.98/94.41/66.76mm. This differs from the old384 panel and must
not be mixed with the old card's score. Audit binds checkpoint/panel/model hashes:
[action-world audit](../../../../../../outputs/goal-point-wam/ref12-methods-20261012-r1/action_world_parent.json).
No additional main-training seed/control is created; fresh method heads and
extra training prevent attributing later improvement exclusively to mechanism.
