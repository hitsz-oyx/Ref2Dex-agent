---
schema: ref2dex.probe.v2
probe_id: P-20261011-pw-critic-policy
experiment_id: P-20261011-pw-critic-policy
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 430ab620
claim_id: C3
hypothesis_family: HF-trajectory-policy-pw-critic
probe_index_in_family: 1
seed_pool: probe
seeds: [187, 188, 189]
decision_changed_if_positive: retain frozen PW consequence features for a larger matched policy-training probe
decision_changed_if_negative: inspect critic informativeness and update scale before spending on longer RL
status: UNCLEAR
run_id: pw-critic-policy-20261011-r1
---

# Frozen PointWorld action-conditioned Q and bounded policy improvement

## Archive status (2026-10-11)

This bounded pilot is stopped and archived by user choice. Its original
both-competent benefit screen remains UNPROMISING; the favorable first-launch
contrast did not establish a mechanism benefit. Old PW artifacts and their
implementation identities remain intact. See the
[premerge archive guide](../../../../../../docs/archive/2026-10-11-pointworld-premerge.md)
before reproducing cache or audit outputs under the legacy checkpoint.
The protocol and chronological records below are retained as history; RUNNING
instructions do not authorize further execution. No formal negative RL claim
follows from this Probe.

## Motivation / Decision

User authorizes ref3 stage2 after fixed-initialization DAgger baseline validation.
Decision Probe: does PW(H, proposed tau) provide useful action-dependent information
to Q and actual actor training, compared with identical Q/actor training without PW?
The cheapest useful test is one fresh exploratory collection, matched offline Q
fits and constrained return-weighted policy regression, then actual native execution.
This is a bounded RL pilot, not a full PPO/AWR reproduction or formal Cm benefit.

## Fixed protocol

Initialization is validated DAgger joint SHA
7c2d1fe98a27c7c3ba72ba44a217c631aea5e46b0c74384a0cc7c9d4f692677e;
R SHA414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f,
reference SHA1f8d3f17a321738ae512a18b4df3eddc7618e9de7ed0fa6e327393103aada52a.
Neither validation400/401 nor prior evaluation packets are training data.
New seed187 collection:24env,542controls,teacher4/GT-R4/learner16.
Both learner blocks use the same DAgger actor, per-control queries, no teacher
interventions. Exploration: independent Gaussian33 point-coordinate displacement
std1mm, coherent over24 horizon with exp(-h/8) decay. Record mean/noise/proposal,
actual command/PD and all measured states. Collection explicitly training_allowed.
Keep all learner outcomes including failures. Whole env train8:20, val20:24.
Use tick0:519 stride4, complete24-step targets, gamma=.99, no bootstrapping.

At each next measured state, reward = exp(-reference_position_error/.05)
+ .2 exp(-reference_rotation_error/.5) + .5 held + .2 exp(-surface_gap/.02)
- .1 clipped - .01 mean((action[t]-action[t-1])^2). At t0 previous action is0.
held: surface_gap<=.01m, rise>=.03m and not supported within2cm.
Reward/held are actual observed labels; planned object reference is authorized G.
Q24 is a behavior-continuation truncated return: actor replans every control,
so it does not measure open-loop24 execution of the proposed plan.

PW main3 action checkpoint SHA
f70573c4d06d368a7d8ab921ff03c936bb0e1b8565df9363b0f1faeb9ab6ba13,
step46000, frozen authenticated implementation/vendor sources. Use current-object
fixed frame, measured history4, semantic right-hand11 mapping, absent left masked,
512 native-scale canonical airplane points. Proposed future hand only; never
actual future measured hand/object input. z_phys is24 predicted object SE3 poses
(3x4 flattened,288D), including prediction for proposal and unperturbed actor mean.
No PW fine-tuning, no editing checkpoint-pinned source, no online PW inference in actor.

Q-on(H328,G408,tau792,z288) vs Q-off(same inputs, zero z288), identical MLP256x2,
initialization, train rows,3000Adam updates/batch256, train-only normalization,
val whole-env MSE selection every250. Record val constant baseline, tau sensitivity,
phase-matched shuffled-z diagnostic. Physical evaluation alone decides benefit.

Policy improvement uses exp((Q(proposal)-Q(mean))/temperature), clipped .5:2,
normalized train weights; temperature=train advantage std floored .001.
Same actor initialization, fixed1000Adam steps, lr1e-5,batch256, weighted squared
normalized proposal likelihood surrogate plus equal unweighted squared deviation
from initial mean. PW influences weights through Q, not a frozen candidate selector.
After training, backtrack parameter interpolation until global train/val trajectory
coordinate RMS change<=.25mm. No physical outcome model selection. Record weights,
nonzero gradients, actual parameters, per-phase changes and RMS so a null result
can be separated from ineffective updates. This follows advantage/Q-weighted
regression ideas ([AWR](https://arxiv.org/abs/1910.00177)), with fixed exploration
and trust constraints particular to this probe; no likelihood or gradient through PW.

Freeze policies, evaluate seed189, same24env teacher4/GT-R4/PW-off8/PW-on8,
per-control queries, exploration/interventions OFF, same R/reference.
First require teacher and GT-R >=3/4 hold/place. Preserve baseline six gates:
>=6/8 hold/place,clip<1%,medianheld>=.8GT-R,posRMSE<=1.5teacher,
rotation mean<=1.5teacher+.1. A PW benefit signal requires both competent plus
PW-on positionRMSE at least5% below PW-off, without fewer placements or shorter
medianheld. PROMISING only then; valid execution without gain UNPROMISING;
calibration failure, ineffective updates or uninformative Q UNCLEAR.

## Resources / stops / limitations

New probe budget20 GPUmin /2GiB new artifacts, within shared Campaign limits,
one idle GPU preferred3. Each native<=300s, feature cache<=300s,fit<=300s,
audit<=120s. Stop on nonfinite/reset/input drift/new foreign GPU PID or budget.
Record utilization, memory and ETA. Keep main, no Campaign edits, no ref additions,
no unknown process operations, no push. Completed baseline budget is not reused.
Single motion/frame0, correlated env/launch samples and short behavior-continuation
credit; no formal science claim. Multi-motion/seed, randomized resets and longer
RL are deferred until a useful pilot signal.

## Results

Collection `outputs/trajectory-policy/pw-critic-collection-20261011-r1/`, producer
430ab620, seed187:542controls/543states, no interventions. Teacher and GT-R each
4/4hold+place; noisy learner blocks hold6/8 and4/8,place4/8 and1/8, clipping
9.50% and1.75%. All failures retained; this collection is not autonomous evaluation.
Independent full H/G/actor mean/noise/plan/liveR/native PD/outcome audit PASS.

PW cache `outputs/trajectory-policy/pw-critic-cache-20261011-r1/`,899233d7:
2080anchors x2 proposals,51.44s GPU3,595MiB device occupancy. Exact repeated and
batch-permuted inference; proposal/mean z RMS difference .0121486 (mixed rotation/
translation units, not a physical error score). Current fixed-frame inputs and
semantic hand mapping independently rebuilt; no observed future hand conditioning.

Matched fit `outputs/trajectory-policy/pw-critic-policy-fit-20261011-r1/`,
ee926dc2,seed188,17.61s GPU3,461MiB occupancy. Q val normalized MSE off .69304,
on .62330 (constant1.37671), selected500/750; on phase-matched shuffled-z .77859.
This is predictive evidence only. Q proposal/mean absolute delta .02193/.09294
return units; train weight std .50885/.41695, arm weight RMS difference .60644.
Both actor1000updates then parameter interpolation .5: global trajectory RMS
change .15818/.16234mm, first8 .10768/.11041mm, startup .05103/.04518mm;
actor-arm difference .08806mm. Nonzero gradients and parameters, all normalization
frozen. Learned off/on SHA418e9ad111d062b4412ef35e1626eda0aa54a06ff16661bb11ad0198ea81fb13 /
addb2877a9b2bb720f71f79f7edb30accb703b04144f1ef9ea8f18dc1d108356.
Independent eligible-label/reward/return/PW first-paired-batch/Q/weight/interpolation/
learned actor audit PASS (461deb5c). No model selection using physical outcomes.
Next fixed seed189 execution with exploration OFF, under the unchanged gates.

## Post-first-execution Decision Note: bounded block-swap diagnostic

First seed189 execution (25347d43) and full independent audit PASS. Teacher/GT-R
4/4hold/place; PW-on8/8hold/place,median442,clip0,posRMSE .07117m; PW-off8/8hold,
6/8place,median408,clip14.4834%,posRMSE .13928m. Reference-angle .41473/.83474rad.
The original six-gate PW-on passes; PW-off fails clipping/position/rotation.
Thus the predeclared BOTH-competent pilot-benefit gate fails: original decision
UNPROMISING by that strict screen, even though the relative on/off observation
favors PW-on. Do not silently relax or relabel this original gate.

Question: is the favorable PW-on observation tied to its environment block?
Before longer RL, perform ONE fresh seed189 launch with the EXACT same learned
actors/R/reference and controls but reverse student blocks (on8:16,off16:24).
No new training, checkpoint/seed selection, noise/intervention or gate changes.
This is cheaper than new training and may expose a layout dependency; native
launch/contact variation is still not independently controlled by this swap.
If on remains competent and off degradation persists, retain PW/Q-weighted learning
as a mechanism candidate for later validation. If the pattern reverses or
calibration fails, treat the relative observation as UNCLEAR and diagnose layout/
launch variance before expanding. Bound one300s launch plus120s audit, same20GPUmin/
2GiB probe envelope; then stop this pilot. No external authorization needed.

Swap r1(c413dc27) stopped safely after1control when a newly appearing foreign
PID2378894 created a254MiB context on GPU3. FAILED manifest and partial.npz retained;
not a scientific result. Read-only ps/nvidia-smi identifies an external Isaac Sim
sensor job active on GPU0, with130MiB idle contexts on all other GPUs. Restart once
as swap-r2 on GPU3 only if utilization<=5%,totalprior<=512MiB; explicitly register
this already inspected idle PID/command using the existing collector allowance.
Stop on any new PID, context growth>512MiB, nonfinite or300s. No external process
operation, no expansion of GPU/count/storage permissions. The physics pilot still
has only one completed swapped launch; no model/seed selection from failed launch.

## Completed swap and archival decision

Swap-r2 (`a9b0fa64`) completed with full execution audit PASS. The stored
`outputs/trajectory-policy/pw-critic-policy-summary-20261011-swap-r2/`
summary reports calibration and both-arm competence PASS. PW-on and PW-off
both hold/place 8/8; position RMSE is .0704696/.0741192 m, median held
435/438 frames and clipping 0/.046125%. Position gain is 4.9239%, below the
predeclared 5% gate, and PW-on has a shorter median hold. Thus this strict
screen remains UNPROMISING. The large first-launch contrast was not repeated;
mechanism attribution remains UNCLEAR. No further training or rollout is
authorized by this historical card. Original outputs and failed swap-r1 stay
preserved; the current user chooses the new branch implementation as the main
PW route. Multi-motion, initialization and training-seed evidence remain
deferred until a new decision justifies them.
