---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref22-gt-anchored
experiment_id: P-20261012-goal-point-wam-ref22-gt-anchored
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 30
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: run one fresh short closed-loop and aggregate actual student-state expert takeovers if drift persists
decision_changed_if_negative: inspect retained physical mapping and coverage before capacity or RL changes
status: UNCLEAR
run_id: wam-ref2-joint-anchored-20261012-r1
---

# ref2_2: GT-anchored R and control-aware WAM

## Decision Note

Serves robot self-trained manipulation before world/evaluator/RL. Horizon
Probe29 found K16 worst-panel36.21mm vs K2439.99mm (<20% improvement),
K8 nominal15.40/t8 5.34/t16 21.66/t32 42.19mm. Retain K24 under the
predeclared rule. This does not exclude predicted far-horizon sensitivity;
it excludes an obvious GT short-window benefit in this cheap recipe.

The next hypothesis is whether separate gradient ownership preserves R's
physical GT-tau mapping while permitting WAM control-aware adaptation.
Start the SAME original startup WAM and qualified original R as prior joint
Probe28, not its drifted final pair or a horizon-selected diagnostic net.
Reuse exact batch64/1000 update/295+296 draws, learning rates and input
admission. Change only the objective/gradient responsibility. All policy
modules scene/goal/trajectory/hand_query train; action/world/statistic buffers
stay fixed. R receives valid GT tau ONLY. Predicted action differentiates
through parameter-frozen R to WAM, never updates R; pure CURRENT-PD queries
provide no GT future and update WAM only. Equal geometry/predicted-action/
GT-action weights; no shared clipping before recording ownership evidence.

OneGPU2; engineering1 before fixed1000/final-only, stage<=1800s; new stage
<=500MB, cumulative ref2<=850MB, global400GB per user-updated Campaign.
One fresh128-control four-role/takeover8 diagnostic<=600s ONLY after unchanged
offline screens from Probe28. Stop on nonfinite, drift, foreign GPU2 or budget.
No new branch, no external writes. This is within current research authority.

## Inputs and decisive checks

See Probe28 for authenticated nominal8304 windows, role3 actual t8 recovery97
and pure CURRENT expert-PD53. Purged late pure current56..104 remains only
development diagnosis. Own H4/current object/known reference object goal are
WAM inputs; measured q/dq enter R only. Pure student failed motion and mixed
policy futures cannot supply geometry or R labels.

For first update inspect ACTION-only gradients into all four WAM modules;
R predicted-action gradient must be0. GT-anchor gradient into R must be
nonzero and into WAM0. Verify all five actual parameter updates and exact
original normalization/action/world hashes. Synthetic contract tests also
change pure-row labels to ensure R GT gradients are unaffected.

Report same initial/final geometry, predicted/GT PD and clipping panels as
Probe28. Physics requires predicted nominal<=40mm/frame0<=20mm/startup<=40mm,
GT nominal<=30mm/startup<=20mm, nominal rotation<=.15rad/fingers<=.3rad and
nominal/startup clipping<1%. Gates stay unchanged; failed models never deploy.
New R must earn local GT physics qualification again; old qualification is
not transferred. Exact learned WAM/R hashes required in both pinned runtimes.

## Fresh short closed-loop if retained

Use128 controls rather than full542. Compare existing startup-anchor source
descriptively (same motion/reset, different live launch) at0/4/8/12/16/24/32:
own wrist trajectory h1 error, actual expert-PD error, clipping, held. Pay
attention to prior9..16 drift. All actual applied commands, measured next
states and privileged training-only expert queries retained. No copied hidden
state and no borrowed nominal future. If still0held, stop loss-only tuning:
next decision is actual student-state expert takeover8/16/32 aggregation.
This short trace cannot establish full placement or independent generalization.

## Limitations / future evidence

Single seed and shared development data, no formal causal Cm gain. Supervised
closed-loop progress does not finish Mission. GT-only ownership prevents one
coadaptation channel but does not guarantee geometric representational meaning
or adequate student-state support. Fresh takeovers and full task/matched
Cm-on/off validation remain subsequent evidence, not assumed outcomes.

## Results

Pending execution.
