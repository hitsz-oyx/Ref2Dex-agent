---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-joint-retargeter
experiment_id: P-20261012-goal-point-wam-ref2-joint-retargeter
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: a1bd58e8
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 28
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: continue jointly learned robot policy and retargeter with actual-state aggregation
decision_changed_if_negative: inspect physical trajectory and control compatibility rather than close the joint-training hypothesis
status: UNCLEAR
run_id: wam-ref2-joint-fit-20261012-r1
---

# Jointly fine-tune the complete WAM policy chain and R

## Decision Note

User explicitly requests end-to-end WAM+R fine-tuning and comparison with
PointWAM. Earlier experiments trained one side while freezing the other;
geometry improvement did not preserve controls, and R alone did not transfer
recovery coverage cleanly. This experiment tests whether recorded-action
gradients through predicted tau can align both modules while retaining actual
hand-trajectory meaning. Success changes the next robot adaptation recipe;
training loss alone does not establish autonomous grasp or Cm contribution.

PointWAM robot phase jointly fine-tunes its forecaster and learns R with hand
MSE and recorded-action L1; RoboDojo without scene labels freezes scene head.
Our separate action-conditioned world head is architecturally different and
has no matching forecaster scene targets here. Keep action/world parameters
frozen; unfreeze EVERY parameter of scene/PTv3, goal, trajectory, hand_query
and R.net. No cached/detached scene tokens, predicted tau or R outputs in the
training path. Fixed existing normalization buffers stay unchanged; all modules
run eval mode to match deployment and avoid running-statistics drift while
parameters remain trainable. This is end-to-end policy training, not frozen
backbone head training. See the primary-source note in paper/pointwam.

Cheapest decisive attempt: reuse authenticated normal teacher and the latest
complete128/t8 actual-state trace, original startup WAM and physically qualified
original R. OneGPU2, smoke1 then ONE fixed1000-update run/final only. Batch64,
WAM scene LR1e-5, goal/trajectory/hand_query/R LR1e-4, AdamW wd.01, clip1,
BF16 model forward and float32 geometric/control losses. Stage training wall
<=1800s, whole attempt<=40GPUmin; one screened full542 native diagnostic<=600s.
New outputs/scratch<=500MB, cumulative ref2<=620MB and CAMPAIGN global<=300GB
(current outputs+tmp292.116GB). This prospective allocation includes the full
backbone patch; it never changes previous100/120MB experiment outcomes.
No new branch/external write/other-user process intervention. Stop on identity
drift, nonfinite, missing end-to-end gradients, foreign GPU2 or budget.

## Inputs and supervision

Original nominal teacher: competent-pd-teacher-20261011-r2, trainenv0:16,
all519 current ticks=8304 complete24-future windows. Fixed report64 panel
16 ticks x env24:28 is development/retention evidence, not a new holdout.
Train replay oversamples startup: batch32 nominal=16tick<16 +16later.

Latest actual native source: wam-ref2-execution-startup-anchor-20261012-r1/native,
128 controls/129states, t8 takeover, root2316/max0 PASS, enclosing group COMPLETED,
trace7c586a79eb8c859405cca1be9cda3b7a28505136d4464f2a24a2e8f80bf343a3.
Batch16 true role3 expert windows current8..104=97, all24 causing controls
actually expert/unclipped/current-q target inversion verified. Derived actor
admission enumerates this narrow source; preserve original future/mixed-policy
prohibitions. Batch16 pure-role2 actual expert-PD queries current0..52=53,
unclipped. These provide CURRENT PD labels only; trajectory_valid=false,
no role3 borrowed futures, no failed actor motion as expert tau.

Pure current56..104 unclipped queries are a purged local diagnostic; current
H4 starts at53 after last training current52. Teacher privileged observation
is training-label provenance only, never actor input. All recovery fit windows
are TRAIN; same reset/takeover is not generalization. Actor inputs remain own
observed H4 + decision-known planned object goal; q/dq enter R only. Planned
goal is reference[min(t+24,542)], never actual episode endpoint.

## Joint objective and gradient evidence

L = MSE(predicted_tau/.05, actual_expert_tau/.05) on valid nominal/recovery
windows + L1((R(current_state,predicted_tau)-encoded_actual_PD)/original_R_target_scale)
on ALL nominal/recovery/pure rows, each term weight1. Active right11points,
left absent/masked. R predicts12 independent coordinates; decode/coupling
remains native18D, wrist relative to CURRENT q/fingers absolute. Keep our
existing R initialization/MLP/current-step output rather than silently claim
PointWAM's random-initialized transformer/30-action chunk is reproduced.

First update independently compute gradients of ACTION LOSS ALONE into scene,
goal, trajectory, hand_query and R.net; all five must be finite/nonzero. Check
actual update hashes for all five, no gradient/weight change in action/world,
all WAM and R normalization buffers byte-identical. Smoke saves no checkpoint.
Report actual throughput/GPU memory/ETA after first100 updates. No extra seed,
control arm, repeated checkpoint selection or loss-weight search.

## Reporting and physical diagnostic gates

Report initial/final raw near8/all24/h1/wrist errors on each valid panel and
actual expert-PD XYZ/rotation/finger/clipping on all. Separate startup0/1/2,
startupall0:15, pure training and purged diagnostics. Also send TRUE tau
through SAME new R to detect losing its physically interpretable mapping.
No training fit or old correlated query diagnostic establishes recovery.

One full542 four-role/takeover8 physics diagnostic if finite/gradient/frozen
identity checks and: nominal predicted-PD XYZp95<=40mm, frame0<=20mm,
startupall<=40mm, nominal+startup clipping<1%; GTtau->newR nominal XYZp95<=30mm,
startupall<=20mm, clipping<1%; nominal predicted and GT rotation p95<=.15rad,
finger p95<=.3rad. These are bounded safe-control screens, not success criteria.
Failed screens preserve the run without physics. Physics teacher/GT arms
must newly establish calibration: original R qualification is not transferred.
Same paired learned WAM+R in all required runtimes; reject mixing learned WAM
with original R. Authenticate both checkpoint hashes, exact module scope,
unchanged norm buffers and actual training input hashes. Save current actions,
expert queries and measured consequences as before; no hidden-state fork.

## Limitations / future evidence

One seed/current motion and small recovery support cannot close the joint
training hypothesis if this attempt fails. Full robot policy-chain adaptation
changes scene features consumed by the frozen world head, so human/world
quality is unverified. No Cm/RL causal contribution tested. PointWAM full
scene+hand forecaster, architecture and action-chunk output are not reproduced.
Larger demonstrations and matched formal comparisons remain future evidence.

## Results

Result: **UNCLEAR** for the joint policy route; mixed offline signal, no physics
admission and no autonomous-success evidence. Fixed1000 completed458.44s on
GPU2, peak2158.69MiB; batch64. Source/input hashes held, all five ACTION-only
gradient sums finite/nonzero, all five module hashes changed, action/world
weights and every WAM/R buffer byte-identical.

| Panel / metric | Initial | Final |
| --- | ---: | ---: |
| Nominal predicted PD XYZ p95 mm | 25.63 | 35.26 |
| Nominal near8 hand EPE mm | 17.20 | 14.76 |
| Nominal true tau -> same R PD XYZ p95 mm | 10.59 | 43.96 |
| Startup frame0 predicted PD XYZ p95 mm | 12.08 | 2.76 |
| Startup all0:15 predicted PD XYZ p95 mm | 20.51 | 9.87 |
| Recovery TRAIN predicted PD XYZ p95 mm | 42.88 | 49.12 |
| Recovery TRAIN near8 hand EPE mm | 19.19 | 13.41 |
| Recovery TRAIN true tau -> same R PD XYZ p95 mm | 20.44 | 62.76 |
| Pure early CURRENT-PD TRAIN XYZ p95 mm | 587.93 | 60.89 |
| Purged late pure CURRENT-PD diagnostic XYZ p95 mm | 940.48 | 955.48 |

All reported clipping0. Exactly one preregistered screen failed: nominal
GTtau->newR43.96mm >30mm. The unchanged screen blocks physics; did NOT swap
back to old R to bypass this gate, nor inherit original R qualification.
Training loss reduction or early fit is not an actual-policy rollout result.

Root independently reconstructed160 metrics (max error0), actual current
PD labels (max0) and current-anchored actual expert future tau (max1.19e-7m),
with pure future masks denied and consumed hashes unchanged. GPU R-only
cross-input diagnostic held each measured state fixed: nominal newWAM+oldR
24.94mm vs exact learned pair35.26mm; recovery newWAM+oldR32.56mm vs learned
pair49.12mm. NewR+oldWAM nominal30.72mm; newR+true tau43.96mm. This local
evidence isolates a harmful contribution of R updating to nominal/recovery
control, alongside a beneficial coadaptation on early pure TRAIN
(newWAM+oldR330.21mm -> learned pair60.89mm). All late pure combinations
remain approximately938--955mm: no local coverage transfer demonstrated.

Independent pre-fit review found a deployment-only persistent=False
voxel-origin loading KeyError; a1bd58e8 fixes it before the full run.43 CPU
contracts passed, including this actual model buffer semantics. Smoke1 at
5b052b93 completed27.79s/no checkpoint and verifies training gradients only;
its result is not invalidated by the deployment-only fix. Original source
and every completed run remain preserved.

Artifacts:
- `outputs/trajectory-policy/wam-ref2-joint-fit-20261012-r1/{manifest,result}.json`
- paired `head.pt` (full scene/goal/head patch) + `tau.pt`, `predictions.npz`
- `root_joint_audit.json`: reconstructed labels/metrics and GPU cross-R checks
- `root_policy_roundtrip.json`, `root_native_checkpoint_audit.json`: actual
  checkpoint loading and rejection checks in the pinned runtimes
- smoke `outputs/trajectory-policy/wam-ref2-joint-smoke-20261012-r1/`

Decision: retain the jointly differentiable implementation; do not close the
joint-training hypothesis or launch this pair in physics. Next distinct
decision is whether adding **true tau->R recorded-PD supervision** alongside
the same predicted-tau end-to-end loss preserves physical R meaning while
keeping early benefits. Preregister that variant before running it; do not
silently change this completed objective. Later actual-state expert coverage
remains necessary evidence rather than a demonstrated improvement.
