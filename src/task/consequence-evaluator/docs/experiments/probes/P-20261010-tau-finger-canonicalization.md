---
schema: ref2dex.probe.v2
probe_id: P-20261010-tau-finger-canonicalization
experiment_id: P-20261010-tau-finger-canonicalization
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 6dc24eb9c5633100739385de89e731841325723b
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 4
seed_pool: probe
seeds: [279, 280, 281, 283, 284, 285]
decision_changed_if_positive: retain tau state closed-loop commands with protected wrist and learned feasible finger preload
decision_changed_if_negative: distinguish feedback state coverage from action parameterization before more PPO
status: PROMISING
run_id: ref7_4-tau-finger-fit-20261010-r3
---

# Preserve the tau tracker while learning feasible finger commands

## Motivation / Decision Note

User asks to continue resolving unstable tau-only execution, without true future
q/object references or tactile input. C3 execution subgoal is unchanged.
Previous fixed-final short PPO holds45 in16/16 but ends holding only4/16;
frozen migrated tau controller already has13/16 near-teacher, clipping41.01%.
The cheapest next decision is command canonicalization of this owned tau-only
feedback policy, protecting its wrist mapping rather than another full PPO.

Ranked hypotheses: (1) unnecessary overdrive and broad policy updates damage
existing preload; (2) adapted previous residual/live-state coverage breaks the
frozen encoder; (3) the approximate finger geometry or residual range limits
physical tracking. Current source analysis confirms eleven of twelve r2 drops
have no clipping in the preceding16controls. Clipping is not established as
their direct cause. Encoding actual executed PD targets reconstructs the same
native commands to2.98e-8: overdrive is unnecessary to reproduce dispatch.

Decision: learn only the six finger output rows of the existing897-input actor
from all16 frozen tau tracker trajectories (8672states), preserving wrist
rows, encoder, critic and logstd bitwise. Labels are **actual applied PD targets
from our own tau-only controller**, not loaded next q, teacher future q/object,
or official actor actions. The input remains future tau + current state and
previous residual. Warmstart still carries historical oracle training knowledge.
This is controller-command distillation, not a claim that geometry has a unique
inverse. A changed previous residual can shift live feedback; evaluate it.

## Fixed implementation and protocol

Source frozen eval: outputs/consequence-evaluator/
ref7_4-tau-tracker-frozen-eval-20261010-r1. Teacher weights: initial.pt of
ref7_4-tau-tracker-train-20261010-r2 (migrated owned fixed reference tracker).
Verify recorded897 features match geometry error and declared teacher actions.
Use every tracker row, including failures, with no outcome-based selection.

Fit1000full-batch updates, Adam1e-3, seed279. Targets clip recorded applied
six finger PD targets to the interior [.005,.995] of native range; thumb-pitch
maximum becomes.54725 instead of.55rad. Loss is mean squared normalized raw
target error plus10 times squared interior-bound excess. Gradient mask freezes
first6 output rows, all other parameters require_gradFalse. No model/checkpoint
selection by rollout, held-out outcome or loss minimum: use fixed final.

Deployment is **unchanged**: q_hat from fixed tau geometry + wrist FF + same
tanh residual limits and native clamp/coupling. No extra preprojection or
intrinsically bounded decoder is added to hide clipping. Raw intended/applied
clipping retains its original meaning. Targets change through learned weights.
Small command fit/testing can use CPU; full repeated fit uses GPU2.

First eval seed280,64env,542controls. Randomized16rows each: old privileged
oracle, zero-residual tau nominal, new student, independently executing frozen
tau teacher (replaces already-tested shifted tau). All arms use own live states;
these are descriptive PhysX rows, not matched state forks/independent seeds.
Teacher is a separate arm, not the student's action source. Reconstruct897
actual inputs, kinematic base, residual, native action and PD targets offline.

Strong gate unchanged: usable oracle>=8/16 hold45; student>=8/16 with>=433
continuous-heldframes, no recorded intermediate loss, median>=nominal+45,
raw clipping<1%. Additionally report terminal held, all unsupported gravity-like
drops, and frozen tau teacher. Source reference ends holding, not placement.

If first fit fails behavior/clip gate and actual wiring passes, allow **one**
DAgger coverage repair: annotate all16 student rows' actual inputs from first
eval with the frozen owned tau teacher on those same live states. Teacher
uses only current897 input and tau-derived base. Combine original and student
states, refit1000updates from the same original weights; fixed final eval seed281.
No outcome-selection, future object or loaded future q label enters annotations.
If that still fails, one bounded64update finger-only adaptation may be considered
only after recording what it will discriminate; no automatic seed/LR sweep.

## Budget / stop

New bounded follow-up authorized2026-10-10, distinct from completed prior card:
one idle GPU2,<=25GPUmin/2GiB, fit<=120s each, eval<=300s each, optional
adaptation<=600s. Preserve foreign processes, require<=512MiB passive use,
<=10%util,>=20GiBfree at launch; stop own run on new heavy foreign process.
Stop for nonfinite, provenance mismatch, wrist/encoder drift, command/input
contract failure, unexpected termination or deadline. Fresh run_id for each
artifact, no checkpoint overwrite, new branch or remote push.

## Limitations / future evidence

Single-motion GT-tau, frozen reset calibration, previous oracle-warmstart knowledge;
no scratch learning, new-tau generalization, exact PhysX causal comparison,
placement or Cm benefit. Command fit is engineering only. Live Probe tags remain
PROMISING/UNPROMISING/UNCLEAR, not formal conclusions. Recovered holding must
not be credited to reduced clipping without a matched mechanistic comparison.

## Runs

Completed; manifests record actual code/input identities and monitored resources.

Engineering smoke (470b9af,debug44,one update) passes the source latent/input
checks, differentiable target loss and wrist/trunk protection. Fixed1000update
fit r1 (470b9af,seed279) completes8.38s onGPU2 (~465MiB,6–7%util): source
raw clipping41.01% ->2.48%, command RMSE8.50 ->3.16mrad. Fit is engineering
only; no behavior-based checkpoint selection.

Complete eval r1 (470b9af,seed280): student16/16 hold45, median483, near43313,
terminal13, clipping584/8672=6.73%; frozen tau median478,near14,terminal14,
clipping3486/8672=40.20%. Oracle median483.5,near14,terminal14,nominal0.
Original strong gate still UNPROMISING for clipping, with adequate live oracle.
Input/nativePD reconstruction passes. Preserved wrist weights do not imply
an identical wrist trajectory when the live/history inputs change.

Predeclared coverage-repair condition met: learning restores long holding but
live raw clipping exceeds training-state clipping. Annotate every first-eval
student row from the frozen owned897-input tau policy on the exact recorded
live/history state; do not pick only its successful rows. Combine both state
sets; same original checkpoint, seed279, fixed1000updates/margin/loss. Eval
seed281 is a new launch, not Validation. This probes feedback coverage before
altering representation or another broad PPO. Added cost~15sfit/~100seval.

Command-fit provenance is transitive through its pinned source manifest; the
audit verifies that hash before checking reference identity, since fitting does
not read future-q/object labels from the reference packet itself.

Coverage fit r2 (0040485,seed279) completes8.92s/GPU585MiB,17344state examples;
training raw clipping2.91%,commandRMSE3.77mrad. Final eval r2 (seed281) gives
student16/16 hold45, median479,near43313,terminal13,clipping193/8672=2.23%.
Oracle16/16 near/terminal, frozen tau12/16 near/terminal, nominal0. All input/
action checks pass. Strong screen remains UNPROMISING only for clipping.
31thumb-yaw and162thumb-pitch overdrives, other finger coordinates0.

Decision Note: coverage repair reduced live overdrive but mean-bound loss still
tolerates rare corner violations (max native excess .1102yaw/.0235pitch),
while long holding is preserved. Before spending another PPO batch, use the
cheaper informative loss intervention: **worst1% coordinate-bound loss** instead
of its mean. Keep coefficient10, original+first-student source states, original
initialization, seed279,1000updates, all six finger rows and target margin fixed.
No new scene/outcome selection or actor inputs. This distinguishes insufficient
weight on rare constraints from state coverage; it is not a seed/LR sweep.
Fixed final eval seed283. Cost~15sfit/~100seval; within25min/2GiB. If it fails
the existing gate, stop this canonicalization experiment and record the specific
remaining limitation rather than automatically doing the optional PPO.

Tail fit r3 (6dc24eb,seed279) completes9.37s/GPU587MiB: source clipping.0577%,
commandRMSE6.82mrad. Fixed-final seed283 evaluation passes the unchanged strong
gate:12/16 qualifying>=433 and terminal held, median479, clipping6/8672=.06919%.
Oracle16/16, frozen tau15/16 (clipping3482), nominal0/16. Student feature error
2.86e-6, command2.38e-7,PD0, reset alignment exact. All12 qualifiers have no
recorded intermediate loss; two loss events belong to failed rows. Three failed
rows (33,57,58) have unsupported gravity-like descent; env42 separates near
table support. Four terminal failures remain. Independent review confirms this.
This is PROMISING for feasible majority holding, not better stability than the
same-launch frozen teacher or total removal of drops.

Positive-gate follow-up Decision Note: now there is a valid low-clipping policy
but four terminal holds fail. Classify one64update finger-only PPO as Decision:
can actual-holding feedback improve contact robustness without changing the
successful wrist mapping? Initialize declared tail-fit final (not the old full
PPO). Freeze actor encoder/wrist/logstd, train only six final finger rows and
critic; sample only finger Gaussian noise, wrist remains deterministic mean.
PPO likelihood includes only the sampled six dimensions. Same actual-lift/hold
reward and.20 raw excess cost; coefficient10 worst1% interior-bound loss on
current mean finger targets. The bound term uses tau-derived q from actual
student feature q+geometry error, not measured future joint labels.
Seed284/64env/64updates, fixed final eval285 with same four roles. No outcome/
checkpoint selection inside training. Added cost~4mintrain/~2mineval, stays under
25GPUmin/2GiB. Preserve the predeclared canonical candidate regardless of this
follow-up; report the new learned candidate honestly if it regresses. Stop
after this one adaptation; broader robustness Validation is deferred.

Finger-only PPO smoke (be738cc,debug45,8env/2updates) passes protected-parameter
and finite likelihood checks. Main Probe run284/64updates completes131072
transitions in226.44s, GPU7455MiB/util0–24%; only two finger-head and six critic
tensors change, wrist/encoder/logstd remain bitwise identical. No foreign GPU
process. Stochastic training clipping31.48% is distinct from mean-policy eval.
Fixed final SHA d9876e210919138fe9ff4407a61f4426f9580c061fa21bdb6bd7b1cf65f9f539.

Final eval285 (be738cc,77.14s): student5/16 hold45,median11,near4330,
terminal0,clipping38/8672=.43819%. Oracle15/16 near and terminal,median483;
frozen tau13 near/14 terminal,median479,clipping3707/8672=42.75%; nominal0.
Feature/native-command/PD/reset checks pass; ten failed rows have unsupported
gravity-like separation. Low raw clipping does not guarantee contact stability.
The finger-only adaptation is UNPROMISING; no intermediate-checkpoint rescue,
seed sweep or further training is run. This is a local negative Probe, not
refutation of tau execution or proof of the exact contact failure cause.
Independent read-only review verifies actual frozen tensors and recomputes
held/clipping from the recorded trajectory, with no clear implementation defect.
Training latents were not saved per step, so sampled logprobs cannot be separately
recomputed from artifacts; source sampling/likelihood semantics were reviewed.

Retain the **predeclared tail canonicalization candidate**, not the regressed
PPO final. It preserves majority holding while satisfying the original gate;
it does not outperform the frozen tau policy in physical holding. Across the
canonical fits/evals and one PPO follow-up, declared run elapsed sums are below
12GPUmin (plus small audits), under25min/2GiB. GPU2 is idle after completion.
Four canonical terminal failures remain; the next useful decision is how to
protect commanded finger preload during contact adaptation, rather than a
longer unconstrained PPO run. Formal matched/multi-seed/new-tau Validation is
deferred; no global C3/Cm claim changes.

Pinned retained checkpoint:
`outputs/consequence-evaluator/ref7_4-tau-finger-fit-20261010-r3/final.pt`,
SHA `79ee282e6b2287375c6480eb8f793864c858311b968ffa97b95c65842502fa45`.
Controller config/decoder bounds/input hashes/reproduction argv:
`outputs/consequence-evaluator/ref7_4-tau-finger-controller-20261010-r1/controller.json`.
Original and failed candidates, train manifests and full trajectories remain
in their own run directories; no checkpoint has been overwritten.

## Related sources

[Previous tau Probe](P-20261010-tau-geometry-tracking.md) remains the source of
failure evidence. [DexTrack notes](../../research/20261010-ref7_3-tracking-control-review.md)
already audit reference tracking/PD command semantics. CAPG primary paper
https://arxiv.org/abs/1802.07564 describes accounting for bounded actions in
policy-gradient estimation; we do not implement its estimator here or infer
that clipping proves the physical cause. Bibliography: paper/capg/schema.json.
