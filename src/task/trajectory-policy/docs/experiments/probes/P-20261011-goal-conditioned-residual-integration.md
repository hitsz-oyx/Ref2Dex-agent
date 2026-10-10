---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-residual-integration
experiment_id: P-20261011-goal-conditioned-residual-integration
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 89dc95e
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 7
seed_pool: probe
seeds: [314]
decision_changed_if_positive: retain the shared nominal-plus-residual executor and consider one bounded on-policy continuation
decision_changed_if_negative: close this student integration variant while retaining the fixed tau residual-basin evidence; do not launch PPO
status: PLANNED
run_id: goal-conditioned-residual-integration-20261011-r1
---

# Goal-conditioned residual policy through the shared executor

## Motivation and decision note

The direct native-action BC/PPO route failed its frozen physical gate, while two
matched fixed-tau residual-basin waves retained a low-clipping `thumb_pitch_minus`
basin. The REGRIND source audit says the portable part to test next is the closed
loop `nominal target + bounded residual + measured-state feedback`, not REGRIND's
asset-specific phase, reset distribution, or large training configuration.

This Probe asks whether a small goal-conditioned policy can distill the retained
r3 tau actor's raw 12-D residual latent and pass it through the same
`GoalConditionedExecutor`. The result changes whether a PPO continuation is worth
running; it does not establish a deployable high-level policy or a REGRIND claim.

## Student and provenance contract

The student receives measured `q`, `dq`, hand keypoints, object pose/velocity,
the previous bounded residual `tanh(raw_latent[t-1])` (zero at tick 0), and a
24-step hand/object goal. It emits an unconstrained raw 12-D latent. The shared
executor applies `tanh`, the fixed residual limits, wrist velocity feedforward,
coupling, native conversion and clipping on every measured step. No future q,
phase, clock, tactile input, or Cm value is a policy feature.

Labels are taken from the exact `role == tracker` rows (16 of 64 environments)
of the retained `ref7_4-tau-finger-final-eval-20261010-r3` trajectory. The
canonical retained checkpoint is independently required to be
`ref7_4-tau-finger-fit-20261010-r3/final.pt` with SHA256
`79ee282e6b2287375c6480eb8f793864c858311b968ffa97b95c65842502fa45`; outcome
selection is not used. The engineering packet supplies only hand/object goals
and remains `engineering_only=true, training_allowed=false`.

For this minimal integration diagnostic, nominal q and position-derived velocity
come from the retained `geometry.npz` cache. This is privileged nominal-reference
information and is explicitly **not** evidence for a deployable no-future-q
goal-to-nominal planner.

## Protocol and gates

1. Fit one GPU BC run with an environment-held-out split, at most 3,000 updates
   and 600 seconds. The offline gate is finite loss and validation raw-latent
   L1 `<= 0.10`; this permits physical execution only.
2. Execute the frozen checkpoint in 16 environments for 542 controls with the
   native DExplore actor only providing the simulator substrate. Record live
   state, all 24 goals, previous/raw/bounded residuals, nominal q/velocity,
   executor target/intended/applied command, PD targets and clipping.
3. Require at least 12/16 long-held terminal episodes and clipping `< 1%` for
   `PROMISING`. Any other finite completed run is `UNPROMISING`; an interrupted
   or provenance-invalid run is not evidence. Do not run PPO after a failed gate.

The output directories are
`outputs/trajectory-policy/goal-conditioned-residual-bc-20261011-r1/` and
`outputs/trajectory-policy/goal-conditioned-residual-integration-20261011-r3/`.
Both must be fresh and retain GPU/ETA monitoring. An independent audit must
replay the residual features, policy output, executor target/command/clipping,
PD conversion, provenance hashes and outcome gate.

## Results

Pending execution.

## Limitations and follow-up

This is a single-motion, single-seed distillation and a privileged nominal
reference diagnostic. A positive Probe authorizes at most one small PPO
continuation under a separate card; it does not support a generalization,
REGRIND reproduction, or Cm benefit claim. A negative result closes this
student interface at the current budget while preserving the fixed residual
basin as a control diagnostic.
