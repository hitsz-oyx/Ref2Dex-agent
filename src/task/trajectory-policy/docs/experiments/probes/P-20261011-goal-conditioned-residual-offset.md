---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-residual-offset
experiment_id: P-20261011-goal-conditioned-residual-offset
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: a72f84f
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 8
seed_pool: probe
seeds: [315]
decision_changed_if_positive: retain a fixed thumb preload adapter and design a separately bounded student/PPO follow-up
decision_changed_if_negative: close the current goal-conditioned residual student route and require a fresh training-eligible teacher or new preload parameterization
status: PLANNED
run_id: goal-conditioned-residual-offset-20261011-r1
---

# Fixed thumb preload offset on the residual student

## Motivation and decision note

The residual student reached 10/16 long-held terminal episodes but failed its
clipping gate (`6.6075%`). Two earlier frozen tau-basin waves independently
identified `thumb_pitch_minus` at raw residual coordinate 11 with `-0.05` as a
low-clipping direction. This Probe tests that narrow explanation by adding the
same fixed offset to the frozen student latent; it does not add training or PPO.

## Protocol

Use the exact residual BC checkpoint, goal packet, geometry cache, native inputs,
16 environments and 542 controls from the preceding integration card. Add only
`fixed_residual_offset[11] = -0.05` to the raw student latent before the shared
executor. Keep the 24-step goals, bounded tanh residual, measured q feedback,
native coupling and all provenance/audit fields unchanged. The nominal q/velocity
source remains privileged geometry cache and is not a deployable high-level claim.

The predeclared gate is at least 12/16 long-held terminal episodes and clipping
`<1%`. A finite completed run failing either condition is `UNPROMISING`; no PPO
is launched after failure.

Output: `outputs/trajectory-policy/goal-conditioned-residual-offset-20261011-r1/`
and audit `outputs/trajectory-policy/goal-conditioned-residual-offset-audit-20261011-r1/`.

## Results

Pending execution.

## Limitations and stopping rule

This remains a single-motion, single-seed privileged integration diagnostic. A
positive result only supports retaining the fixed adapter for a new card; it is
not evidence for generalization, REGRIND reproduction, Cm benefit, or a learned
preload policy. A negative result closes this student route at the current
budget.
