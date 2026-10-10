---
schema: ref2dex.probe.v2
probe_id: P-20261010-tau-finger-canonicalization
experiment_id: P-20261010-tau-finger-canonicalization
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 3aa7f65
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 4
seed_pool: probe
seeds: [279, 280, 281, 283]
decision_changed_if_positive: retain tau state closed-loop commands with protected wrist and learned feasible finger preload
decision_changed_if_negative: distinguish feedback state coverage from action parameterization before more PPO
status: UNCLEAR
run_id: ref7_4-tau-finger-fit-20261010-r1
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

Pending; manifests record actual code/input identities and monitored resources.

## Related sources

[Previous tau Probe](P-20261010-tau-geometry-tracking.md) remains the source of
failure evidence. [DexTrack notes](../../research/20261010-ref7_3-tracking-control-review.md)
already audit reference tracking/PD command semantics. CAPG primary paper
https://arxiv.org/abs/1802.07564 describes accounting for bounded actions in
policy-gradient estimation; we do not implement its estimator here or infer
that clipping proves the physical cause. Bibliography: paper/capg/schema.json.
