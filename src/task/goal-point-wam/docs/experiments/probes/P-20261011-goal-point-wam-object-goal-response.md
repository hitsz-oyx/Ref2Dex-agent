---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-point-wam-object-goal-response
experiment_id: P-20261011-goal-point-wam-object-goal-response
date: 2026-10-11
task: goal-point-wam
branch: main
git_commit: 66f42def8266acd5459b3e1d1af2b1625f24e7b4
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 6
seed_pool: probe
seeds: [247, 251]
decision_changed_if_positive: proceed to a bounded stochastic object-goal trial before any robot integration
decision_changed_if_negative: stop training scale-up and inspect object-goal geometric learning contract
status: UNCLEAR
run_id: object-goal-response-20261011-r1
---

# Object start/goal point contract and frozen response Probe

Class: Decision. User explicitly requests the object-goal route described in
`src/task/goal-point-wam/docs/user/ref/ref1_1.md`. This serves the WAM trajectory
representation subgoal, while self-trained RL and matched Cm utility remain the
final mission. The specific conjecture is that replacing the hand endpoint proxy
with the intended object point-set goal yields meaningful conditional trajectory
response, rather than only memorizing scenes. The cheapest discriminator is a
six-train-window fit and frozen endpoint intervention, before mixed-source scale-up.

Result: UNCLEAR: all three sources fit, but symmetric object-goal response remains weak.
Decision: Stop scale-up; preserve the object-goal implementation and diagnose geometric conditioning before further training.

## Contract and protocol fixed before execution

- Target object comes from the current `object_features[:,13]` flag, with exactly
  one required target. It is not chosen using future motion or hand information.
- 32 uniformly indexed points from the target's512 canonical/current points,
  paired with their t+24 positions via the existing rigid effect. Both sets stay
  in the ONE current anchor coordinate frame; no future-anchor recentering.
- Goal encoder is a point-set transformer with start/end type embedding and a
  null token, no hand/keypoint/index IDs. `goal_hand` fields are absent. Current
  hand and future executed hand remain trajectory labels/world inputs only.
- Scene encoder,528-token executed-action adapter, goal-blind world decoder,
  losses and data/statistics identities are unchanged. Hand-proxy checkpoints
  remain supported by the historical default config.
- Reuse exact six train indices from the earlier seed251 selection; no new
  sample selection and no val/test loader. Independently reconstruct current
  object, future object and hand trajectory from raw canonical points/poses,
  with maximum tolerance2e-5m. Audit target flag, masks and permutation invariance.
- Fresh WAM heads and strict PW step50000 scene transfer, seed247. This is NOT
  a matched quantitative comparison against the historical fixed-fit arm, which
  started from a400-update WAM parent. Its role is semantic feasibility only.
- Engineering smoke4updates followed by300 fixed full-batch updates in train
  mode, BF16, AdamW lr1e-3, decay.01,20step linear warmup then constant learning
  rate, gradient cap1. Save one model-only checkpoint/run, no optimizer copies.
- Fit gate: final/initial mean trajectory and moving-world h24 EPE ratios both
  <=.5 on at least two sources. Fit alone does not pass the experiment.
- Frozen response: translate ONLY goal object points by +/-.01m and +/-.1m X,
  keep scene/start object/executed hand unchanged, reuse identical cached scene
  to exclude sparse recomputation noise. Report per-window signed h24 hand X
  gain (hand displacement / signed object shift), full-vector response RMS,
  per-source median gains and world isolation. Record unchanged whole-forward
  repeat noise separately; do not silently equate it to goal leakage.
- Mechanism response gate: for BOTH +.1m and -.1m, median signed hand X gain
  >=.1 on at least two sources. This is an exploratory directional-response
  threshold, not a physical requirement of gain1 or a labeled counterfactual.
  Smaller shifts test whether a large-shift-only failure could be OOD-sensitive.
- Also replace future object goal with current object points and report target
  trajectory EPE. This ablation diagnoses information use, not robot success.
- Overall PROMISING requires fit and response gates and cached-scene world
  isolation <=1e-5. Otherwise UNCLEAR; retain every metric and do not reject
  the full WAM/Cm hypothesis based on six windows.
- One idle GPU2, <=15minutes total, <=600MiB artifacts, project working artifacts
  currently about258GB (below300GB). Stop on nonfinite/OOM, raw target drift,
  goal leakage at identical scene, competing GPU process or deadline. No new
  branch, downloads, video, robot, DAgger or RL.

## Decision Note

The user selects the intended object-goal semantics. Earlier fitting success,
weak hand-goal response and comparable scene/condition token norms motivate
checking this contract before attention/LR tuning. Root chooses this cheap
train-only test within current authorization; positive response justifies a
bounded stochastic trial, weak response stops scale-up and returns to geometric
learning diagnosis. No core mission, claim, resource or permission boundary
changes; no external authorization is needed.

## Limitations / future evidence

Six fixed training windows can be memorized, and object-goal perturbations have
no labeled feasible hand counterfactual. A physically valid hand response need
not equal object translation. Failure can reflect supervision ambiguity, scarce
same-scene goal variation, coordinate scale or optimization, rather than an
implementation defect or proof that object conditioning is ineffective. Formal
validation, matched world-loss-off and robot/RL utility remain deferred.

## Artifacts

- [Input identity and independent raw target audits](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-r1/input_manifest.json)
- [Training result and branch-gradient audit](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-r1/result.json)
- [Frozen goal interventions and static-object-goal ablation](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-r1/goal_response.json)
- [Strict checkpoint reload and input/gradient verification](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-r1/replay_goal_response.json)
- [Frozen replay source](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-r1/replay_checkpoint.py)
- [Final model-only checkpoint](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-r1/final.pt)
- [Saved train batch](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-r1/fixed_train_batch.pt)
- [Successful four-update smoke](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-smoke-r2/result.json)
- [Original pre-training identity-check failure](../../../../../../outputs/goal-point-wam/object-goal-response-20261011-smoke-r1/failed.log)
- [Probe code](../../../tools/audit/probe_object_goal.py)
- [Object-goal config](../../../configs/three_domain_object_goal_probe.json)

## Engineering smoke

7 contract tests passed,1 skipped. The first smoke stopped before training because
the audit compared a source-local sequence ID to a mixed global ID. `66f42def`
corrects that comparison using the registered sequence offset; original FAILED
record/log remain. Smoke-r2 completed4updates, finite two-branch shared gradients,
zero oracle losses and cached-scene world isolation passed. Its UNCLEAR fitting
label is engineering-only and is not scientific negative evidence.


## Completed Probe result and implementation review

Actual run code commit is `66f42def`. All300updates completed on GPU2 in62.8s
including initialization and frozen audits. BF16 fitting sampled GPU utilization
11--69%, training memory about1.45GiB, maximum recorded audit memory1.745GiB.
The successful smoke and Probe together produced about406MiB, within the600MiB
cap. No val/test dataset was constructed and no other GPU process was touched.

Strict PW scene transfer again covers50,436,224parameters. Current-object and
hand-trajectory raw reconstruction errors are0; future-object point error is
at most2.98e-8m. Oracle loss terms are0. Both losses reach the shared scene
projection after fitting (norms .0001025/.003225, cosine+.2111 on this batch).

| Source | Final trajectory EPE (mm) | Final moving-world h24 EPE (mm) | Final/initial trajectory / world |
| --- | ---: | ---: | ---: |
| OakInk2 | 2.163 | 3.255 | .0765 / .0834 |
| GRAB | 2.603 | 3.017 | .0207 / .0121 |
| ARCTIC | 3.081 | 3.512 | .0530 / .0385 |

The fitting gate passes all three sources. However, the +/-.1m object-goal
translation produces only .03348/.03378mm full-coordinate endpoint RMS response.
Per-source signed endpoint X gains are:

| Source | +10cm object shift | -10cm object shift |
| --- | ---: | ---: |
| OakInk2 | 2.73e-5 | 2.59e-5 |
| GRAB | 2.66e-4 | 2.52e-4 |
| ARCTIC | 5.16e-5 | 4.19e-5 |

No source reaches the predeclared.1 gain screen. +/-1cm shifts also have weak
response (.00336mm endpoint RMS), so this observation is not confined to the
largest intervention. Real versus static-object-goal trajectory EPE is
2.6179 versus2.6212mm; this ablation barely changes these memorized predictions.

Every cached-scene world intervention difference is exactly0. Unchanged
whole-forward world repeat max is1.33e-5, above the historical1e-5 tolerance;
this is retained as numerical repeat variation, not called a passed
whole-forward gate or goal leakage.

A separate frozen strict reload reproduces the weak response. A forward hook
verifies the actual goal encoder receives the object start/goal tensors and
mask, with no hand-goal fields. Changing goal coordinates changes goal tokens
(RMS .01161 on +10cm, token mean L2 norm24.24), and the trained endpoint has
finite nonzero coordinate gradients (max1.76e-6). These checks reveal no
missing object-goal input or disconnected gradient path; they do not prove
useful conditioning. Token magnitudes alone cannot establish the failure cause.

Overall **UNCLEAR** under the fixed protocol: changing to the intended object
point-set semantics does not by itself pass this short conditional-response
screen, despite good fixed-window fitting. The evidence does not reject the
object-goal/WAM/Cm route and is not a matched hand-versus-object comparison.
Root preserves the semantic correction, stops further training now, and leaves
geometric feature scale / decoder conditioning / same-scene goal variation as
next discrimination hypotheses. No robot, DAgger, RL, video or mixed-source
scale-up was started; core mission and user-authored context files are unchanged.
