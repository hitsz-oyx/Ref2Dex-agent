---
schema: ref2dex.probe.v2
probe_id: P-20261011-regrind-executor-gt-generated
experiment_id: P-20261011-regrind-executor-gt-generated
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 7748c42
claim_id: C3
hypothesis_family: HF-trajectory-policy-execution-interface
probe_index_in_family: 2
seed_pool: probe
seeds: [297]
decision_changed_if_positive: keep the shared executor and continue diagnosing generated tau coverage
decision_changed_if_negative: repair the executor/control semantics before changing the high-level tau proposal
status: UNPROMISING
run_id: regrind-executor-gt-generated-20261011-r1
---

# Same-executor GT/generated tau physical comparison

## Motivation and decision

The REGRIND source audit identified `reference target + bounded residual + measured-state
feedback` as the portable idea, while rejecting REGRIND's asset-specific gains, phase,
reference reset, and tracking reward. The shared `GoalConditionedExecutor` passed an 8,672-row
offline equivalence replay, but the physical separation between a valid GT tau and a generated
tau should be checked at the current code commit before any new training.

This is one bounded Decision Probe. It keeps the DExplore native controller, real frame-0 reset,
24-step plan/8-step execution, frozen tau residual tracker, and existing metric decoder. The four
roles are `tau_gt`, `dense_fk`, `metric48`, and `metric48_repeat`, with four rows per role. GT and
dense rows are calibration; generated rows are descriptive evidence for the high-level proposal.

## Protocol and fixed inputs

Run `tools/run/probe_decoder_execution.py` at code commit `7748c42` with the existing immutable
inputs:

* native input manifest: `outputs/consequence-evaluator/hand-execution-inputs-contact-capture-20261010-r1/inputs.json`;
* hand reference: `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`;
* geometry: `outputs/consequence-evaluator/ref7_4-tau-geometry-20261010-r1/geometry.npz`;
* frozen residual checkpoint: `outputs/consequence-evaluator/ref7_4-tau-finger-fit-20261010-r3/final.pt`;
* metric decoder basis: `outputs/trajectory-policy/metric-trajectory-decoder-20261010-r1/basis.npz`.

The run uses one idle GPU, 16 environments, 542 controls, and a 300-second timeout. It must
record requested/intended/clipped/applied commands, native PD targets, measured state, all 68
replans, and gate outcomes. Stop on input drift, nonfinite values, foreign GPU work, early reset,
or timeout. The output directory is
`outputs/trajectory-policy/regrind-executor-gt-generated-20261011-r1/` and must not overwrite
any prior result.

## Predeclared interpretation

The calibration requires at least 3/4 long-held terminal episodes for both `tau_gt` and
`dense_fk`. If calibration fails, the executor/control contract remains the blocker. If
calibration passes and both generated arms fail the unchanged long-held/terminal screen, keep
the executor and investigate tau coverage; this is a local negative result for the fixed metric
decoder, not a refutation of H-to-tau or REGRIND-style residual control. No result from this
single wave is a formal claim about REGRIND or Cm utility.

## Results

The valid run completed at `7748c421388e0a5e47cfadcf85b46b1b66ceb9de` in 57.76 seconds on
GPU0. Graspenv Torch 2.4.1+cu121 and IsaacGym were used; peak Torch allocation was
150,638,080 bytes and the native process used about 7.54 GiB. The GPU was idle after the run.

The calibration passed: `tau_gt` and `dense_fk` each reached 4/4 long-held terminal episodes
(median held frames 478.5 and 482.5; clipping 0% and 0.09225%). The generated metric decoder
reached 2/4 (`metric48`, median 387 frames, clipping 0%) and 0/4 in the repeated arm
(`metric48_repeat`, median 291.5 frames, clipping 0.04613%). Six of the eight generated rows
therefore lost the terminal hold. The predeclared result is `UNPROMISING` for this fixed metric
decoder and executor combination; it does not refute H-to-tau or residual feedback in general.

Independent audit
`outputs/trajectory-policy/regrind-executor-gt-generated-audit-20261011-r1/` completed with
`calibration_pass: true`. It reconstructed all 68 replans and 897 control inputs. Maximum
absolute errors were `features=2.86e-6`, `velocity=9.54e-7`, `intended=2.38e-7`,
`command=2.38e-7`, and `pd=0`; requested/applied commands and outcomes matched. Thus the
generated failure is not explained by a current executor wiring mismatch.

The earlier system-Python and wrong geometry-file invocations stopped before simulation and are
not scientific runs; the completed manifest uses the fixed directory input contract.

## Follow-up decision

Keep the shared executor, real frame-0 reset, H/D/R contracts, and no phase/future-reference
inputs. Do not launch the REGRIND 4096-env/reference-tracking training configuration or failure
bin sampling yet. The next useful work is high-level tau coverage/conditioning (or a small
GT-centered tau perturbation basin), because the calibrated GT path succeeds while generated
metric tau loses terminal holds despite an independently audited executor. Any H/PPO follow-up
needs a new bounded card; this result is not a formal REGRIND or Cm claim.

## Limitations and future evidence

This is a single-motion, single-seed frozen execution comparison. It does not test an H actor,
PPO, failure-bin sampling, interaction-mesh retargeting, or matched Cm-on/off training. Those
remain deferred unless this executor-level decision is positive.
