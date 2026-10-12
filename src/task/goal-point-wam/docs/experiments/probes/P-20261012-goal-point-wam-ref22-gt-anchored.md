---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref22-gt-anchored
experiment_id: P-20261012-goal-point-wam-ref22-gt-anchored
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 6b8bb3d9
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

Result: **PROMISING** for preserving the local GT-tau execution mapping;
autonomous utility remains **UNCLEAR** after the completed physical diagnostic below.
Fixed1000 completed404.40s/GPU2, peak2158.69MiB. The original offline
screen passes13/13. R predicted-action gradient0, GT-anchor gradient nonzero;
all four WAM ACTION-only gradients and all five parameter updates nonzero,
action/world and normalization hashes unchanged. Engineering1 completed
23.55s/no checkpoint.39 relevant contracts and5 aggregation contracts pass.

| Panel / PD XYZp95 mm | Initial | Anchored final | Prior unanchored final |
| --- | ---: | ---: | ---: |
| Nominal predicted | 25.63 | 22.67 | 35.26 |
| Nominal GT tau -> R | 10.59 | 13.35 | 43.96 |
| Startup frame0 predicted | 12.08 | 5.82 | 2.76 |
| Startup all predicted | 20.51 | 8.31 | 9.87 |
| Recovery TRAIN predicted | 42.88 | 38.72 | 49.12 |
| Recovery TRAIN GT tau -> R | 20.44 | 6.19 | 62.76 |
| Early pure CURRENT-PD TRAIN | 587.93 | 159.87 | 60.89 |
| Late purged pure diagnostic | 940.48 | 965.32 | 955.48 |

Nominal near8 hand17.20->15.52mm. Improvement trades off some early pure
training fit against retaining physical R meaning; late coverage remains
unresolved. Same stream/initialization comparison remains single-seed
development evidence, not Validation. Independent root label/160metric
reconstruction PASS: PD max0/tau1.19e-7m/metric0.

Artifacts: `outputs/trajectory-policy/wam-ref2-joint-anchored-20261012-r1/`
contains exact head/tau pair, result/manifest/predictions and root_joint_audit.
Fresh128/takeover8 group `wam-ref2-execution-anchored-t8-20261012-r1`
failed BEFORE any physics because the calibrated-source preflight rejected
the evolved pd_retargeter.py hash. This is engineering rejection, not negative
method evidence; its group/server/native logs are preserved.0e4f1e76 adds
authenticated historical Git source plus runtime GPU default-K24 differential
replay and unchanged PD-conversion AST checks. All other source/data/config
inputs remain byte-bound.519 actual calibration windows show bitwise equal
old/new R outputs in graspenv2.4.1;18 related tests pass. The ORIGINAL
calibration files remain untouched. Fresh group sibling `...-t8-...-r2`
repeated this check in the pinned2.0.1 native runtime and completed physics;
its terminal bookkeeping failure and recovered evidence are detailed below.

### Completed fresh physical diagnostic and terminal bookkeeping recovery

The r2 trace actually completed128 controls/129 measured states and128
durable queries; server COMPLETED128. Native terminal check then failed because
the relative fit-script path was resolved after chdir into third_party/DExplore.
aa95652d roots evidence paths before native chdir;6 provenance contracts pass.
Original FAILED manifests/logs are preserved. `recover_wam_terminal.py`
authenticates full trace==terminal partial snapshot, every data/checkpoint
hash and historical code at recorded commit, then creates a fresh derivative
`wam-ref2-execution-anchored-t8-20261012-r2-recovered`; NO second simulation.
Derived root2316 checks/max0 PASS, own H4/PD/expert masks and approximate
prefix pair within original bounds. Actual native2.0.1 K24 compatibility
replay also519 windows/max0, unchanged conversion AST.

| Same-motion short physical development diagnostic | Previous startup | New anchored |
| --- | ---: | ---: |
| t0 PD XYZ error mm | 11.85 | 6.76 |
| t8 PD XYZ error mm | 43.79 | 12.93 |
| t12 PD XYZ error mm | 103.44 | 21.98 |
| t12 wrist deviation from same-trace teacher mm | 30.80 | 9.33 |
| t16 PD XYZ error mm | 67.97 | 13.90 |
| t16 wrist deviation mm | 48.61 | 10.03 |
| t24 PD XYZ error mm | 97.15 | 265.83 |
| t32 wrist deviation mm | 43.15 | 354.10 |
| Pure actor maximum held frames | 0 | 0 |
| Pure actor clipping | 7.81% | 5.47% |

New teacher/GT-R/t8-expert held73/75/74 frames, each clipping0: local R
grasp calibration earned anew, without full placement proof. Pure actor still
fails and later drift is severe; early-loop benefit does not establish useful
autonomous manipulation. Overall policy outcome **UNCLEAR**, local mapping
and early-loop mechanism **PROMISING**. Decision: stop loss-only fitting,
collect actual learned-student t16/t32 takeovers and aggregate with this t8
trace (Probe31). Full raw reports are short_rollout_comparison.json and
root_recovery_audit.json in the derivative native directory.
