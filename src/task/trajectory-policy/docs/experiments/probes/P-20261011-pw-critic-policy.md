---
schema: ref2dex.probe.v2
probe_id: P-20261011-pw-critic-policy
experiment_id: P-20261011-pw-critic-policy
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-pw-critic
probe_index_in_family: 1
seed_pool: probe
seeds: [187, 188, 189]
decision_changed_if_positive: retain frozen PW consequence features for a larger matched policy-training probe
decision_changed_if_negative: inspect critic informativeness and update scale before spending on longer RL
status: RUNNING
run_id: pw-critic-policy-20261011-r1
---

# Frozen PointWorld action-conditioned Q and bounded policy improvement

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

Pending implementation, collection, feature/learning and physical audits.
