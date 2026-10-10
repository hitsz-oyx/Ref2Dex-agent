---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-finger-residual
experiment_id: P-20261011-goal-conditioned-finger-residual
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 81b59f4
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 9
seed_pool: probe
seeds: [316]
decision_changed_if_positive: retain analytic-wrist/finger-residual decomposition and design a separately bounded continuation
decision_changed_if_negative: close this low-level student decomposition and proceed only with fresh teacher provenance or Stage 2 offline forecasting
status: UNPROMISING
run_id: goal-conditioned-finger-residual-20261011-r2
---

# Analytic-wrist, finger-only residual student

## Motivation and decision note

The full 12-D goal-conditioned residual student reached partial holds but failed
its physical gate, while the retained tau executor has a nominal wrist
feedforward and the user-specified ref2 route explicitly allows a learned
low-level controller to focus on contact/preload. This Probe tests whether the
student's learned wrist correction is the failure source. It keeps the same
24-step goals, live measured-state feature contract, retained teacher source,
executor and physical gate, but assigns all six wrist coordinates to the
analytic nominal controller and trains only the six active finger residuals.

The experiment distinguishes two hypotheses: (A) full residual output damages
tracking through an under-conditioned wrist inverse, in which case finger-only
student execution should recover the gate; (B) the retained teacher labels or
goal-conditioned student distribution are insufficient even with analytic wrist,
in which case the gate remains failed. It does not claim a deployable H-to-tau
planner or a REGRIND reproduction.

## Contract and protocol

The input is the existing 1170-D residual observation. Previous residual history
has six zero wrist slots and six bounded finger slots. The network emits six raw
finger latent coordinates; the shared executor inserts zero wrist residual,
applies tanh/limits, coupling, native conversion and clipping. No future q,
phase, clock, tactile input, contact label, native action or Cm enters the
policy. Teacher labels are exactly the retained `role==tracker` raw latent
coordinates 6:12; the engineering-only packet supplies goals only.

Fit one environment-held-out BC run (seed316, at most 3,000 updates/600s), then
execute one fresh 16-env/542-control physical run. The predeclared physical gate
is at least 12/16 long-held terminal episodes and clipping `<1%`; otherwise the
Probe is `UNPROMISING` and no PPO is launched. Independent audit must replay
features, masked latent, bounded residual, executor, PD targets and outcomes.

Outputs:
`outputs/trajectory-policy/goal-conditioned-finger-residual-bc-20261011-r1/`,
`outputs/trajectory-policy/goal-conditioned-finger-residual-20261011-r2/`, and
the matching audit directory.

## Results

The offline fit was `PROMISING` for a finite validation loss: 6,504 train rows,
2,168 held-out rows, and best raw-finger latent L1 `0.0208398` after 1,300
updates on GPU0. The valid physical rerun is
`outputs/trajectory-policy/goal-conditioned-finger-residual-20261011-r2/`.
The first physical attempt (`...-r1/`) stopped at control 384 because a
transient GPU process guard fired before any trajectory was written; it is a
setup failure and is not counted.

The completed r2 run used 16 environments and 542 controls. It reached `0/16`
long-held terminal episodes and `7/16` hold45 episodes, with median held `5`
frames and clipping `13.5148%`. All controlled-placement success flags were
false, so the physical gate failed and the Probe is `UNPROMISING`.

The independent replay audit is `PASS`: feature error `2.86e-6`, latent error
`5.96e-7`, bounded-residual error `5.36e-7`, executor target error `3.58e-7`,
command error `4.77e-7`, exact clipping flags and exact PD reconstruction. The
finger-only intervention therefore does not rescue the current residual
student; no PPO or further wrist/finger mask sweep is authorized by this card.

## Limitations and stopping rule

This is a single-motion, single-seed privileged nominal-reference diagnostic.
Even a positive result only retains the decomposition for a new card; it does
not prove generalization, H-to-tau forecasting, Cm utility, or full REGRIND
reproduction. A negative result closes this decomposition at the current budget.
