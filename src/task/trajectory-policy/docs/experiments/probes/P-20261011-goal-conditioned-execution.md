---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-conditioned-execution
experiment_id: P-20261011-goal-conditioned-execution
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 0b5466e9914d14d54f7a8b9e5c10eaf7371ed468
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 4
seed_pool: probe
seeds: [312]
decision_changed_if_positive: keep the goal-conditioned low-level interface and start a bounded PPO residual continuation
decision_changed_if_negative: retain the shared executor but obtain a fresh explicitly training-eligible teacher rollout or revise the goal/action interface before PPO
status: UNPROMISING
run_id: goal-conditioned-execution-20261011-r1
---

# Stage 1 physical execution gate

## Motivation and decision note

The offline BC Probe fit a frozen MLP to the native actions of the self-owned
reference tracker. This physical Probe asks the decision-relevant question:
does the same student, given only live state and a fixed eight-step hand/object
goal, preserve grasp/hold behavior in the real DExplore controller? The earlier
tracker wave is the calibration reference (32 tracker rows, 32/32 long-held and
31/32 near-teacher under its declared oracle contract); this run has a separate
`goal_bc` role and does not silently reuse its q or action inputs.

The student consumes no future q, teacher action, phase, clock, contact force, or
Cm value. The engineering-only packet remains a goal-only source: its actions and
q are rejected by the runner. Native action dispatch, PD conversion, measured
state capture, geometry gaps, support and table gates are recorded for all 16
environments.

## Fixed inputs and gate

* native input manifest:
  `outputs/consequence-evaluator/hand-execution-inputs-contact-capture-20261010-r1/inputs.json`;
* goal-only reference packet:
  `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`;
* geometry/reset contract:
  `outputs/consequence-evaluator/ref7_4-tau-geometry-20261010-r1/`;
* frozen BC checkpoint and result:
  `outputs/trajectory-policy/goal-conditioned-bc-20261011-r3/`;
* one idle GPU, 16 environments, 542 controls, and a 300-second execution
  timeout. Outputs are written to a fresh directory and all source hashes are
  captured before simulation.

The predeclared physical gate is at least 12/16 environments with both 433-frame
long hold and terminal held geometry, and native clipping rate below 1%. A
failure is local evidence about this fixed single-motion student and does not
refute REGRIND's residual-control idea or the later H/PointWAM route. A pass is
still only a Probe result; PPO is a separate experiment.

## Results

The valid run completed in
`outputs/trajectory-policy/goal-conditioned-execution-20261011-r1/` in 61.7
seconds on GPU0. IsaacGym/Graspenv Torch 2.4.1+cu121, 16 environments and all
542 controls completed; peak Torch allocation was 148,920,832 bytes (about 7.54
GiB native process memory). The run produced finite commands/PD targets and no
early reset.

The student reached `0/16` long-held terminal episodes, `0/16` 45-frame holds,
and median held frames `0`; the predeclared gate therefore returns
`UNPROMISING`. It used no native action clipping because the policy output was
already bounded. The earlier tracker calibration remains separate evidence and
is not silently counted as a student row.

The independent audit
`outputs/trajectory-policy/goal-conditioned-execution-audit-20261011-r1/audit.json`
passed the wiring/provenance checks: feature reconstruction max error
`8.45e-3`, policy replay max error `6.41e-7`, requested/applied max error zero,
finite PD targets, and goal packet authorization preserved. The first student
action differs from the same-state tracker label by only active-action L1
`0.00208`; standardized feature distance stays below `0.05` through tick 30
but grows to `0.73` by tick 44 after the object has already dropped. This
supports a closed-loop covariate-shift/hidden-state failure of one-step BC, not
an executor wiring mismatch.

## Limitations and follow-up

This is one motion and one seed. It does not test high-level forecasting,
generalization, failure-bin sampling, matched Cm-on/off training, or a fresh
training-eligible teacher corpus. The next route decision is a bounded
goal-conditioned PPO/residual continuation or a fresh explicitly
training-eligible teacher rollout; the failure does not refute REGRIND's
measured-state residual-control principle.
