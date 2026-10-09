---
schema: ref2dex.probe.v2
probe_id: P-20261010-hand-action-fixed-wrist-teacher-finger
experiment_id: P-20261010-hand-action-fixed-wrist-teacher-finger
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: dda2835
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 4
seed_pool: probe
seeds: [282]
decision_changed_if_positive: attribute the r2 native failure to the learned finger branch and retain fixed wrist for a narrow command-decoder follow-up
decision_changed_if_negative: close the fixed-wrist/native upper-bound attribution for this source and keep later routes frozen
status: UNPROMISING
run_id: ref7_2-hand-action-fixed-wrist-teacher-finger-20261010-r1
---

# Can live teacher fingers rescue the fixed-wrist native execution?

## Decision note

The valid fixed-wrist/v2-finger Probe held only `4/6/8` frames, while its wrist
coordinate RMSE stayed below `8 mm`. A purely offline comparison cannot tell
whether the contact loss is caused by the learned finger branch or by the
fixed-wrist/source-state mapping. This is one same-process attribution audit,
not another R training or decoder sweep.

The env-0 native teacher remains unchanged. At each dispatch tick, the three
test envs receive the current live env-0 teacher action for coordinates `6:18`,
while coordinates `0:6` still come from the fixed-wrist decoder. Thus the
finger command is an intentional live-teacher oracle and is not a deployable
input. A positive result only justifies a narrow commanded-finger decoder
follow-up; a negative result closes this fixed-wrist native explanation for
the source.

## Contract and protocol

- Source: `ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`.
- Checkpoint: frozen contextual v2 R; its predicted finger coordinates are
  ignored by the test envs, but its query path remains unchanged for the
  wrist future.
- Runtime: four roles, seed `282`, GPU2, CPU tensor pipeline with GPU PhysX,
  542 controls, query period 24, live `q/dq` at every dispatch tick.
- Env0 is the native teacher. Env1--3 use the current env0 teacher
  `6:18` action and the analytic current-anchored wrist decoder for `0:6`.
- Required engineering contract: all 542 controls, finite packets,
  requested equals applied, and no clipping. The behavioral screen is the
  existing `>=90%` of the source/live teacher hold reference; hand RMSE below
  `40 mm` remains secondary.

## Scope boundary

This oracle control is deliberately privileged and cannot be used as a
deployability or hand-representation claim. It is one source and one seed;
the result cannot generalize across finger representations or backends. No
new R training, contact threshold sweep, native v3 checkpoint, H-to-hand,
PointWorld, evaluator, selector, MPC, or Cm integration is unlocked by this
audit alone.

## Result

The run completed all 542 controls in 44.7 seconds. The live teacher held
`482` frames; the three fixed-wrist/live-teacher-finger roles held only
`8/10/9` frames, with hand RMSE `7.81/7.87/7.85 mm`. Overall, wrist, and
finger clipping were all zero, and requested controls matched applied controls
exactly. An independent packet check found zero difference between every test
env finger dispatch and the current env-0 teacher finger command, while the
wrist dispatch differed from the model wrist output as intended.

The teacher-finger oracle therefore did not rescue the fixed-wrist native
execution. This makes the preceding r2 failure a combination-level negative;
it does **not** isolate or refute the learned finger decoder. The narrow native
finger-decoder sweep remains stopped.

## Artifacts

The run output is task-owned under
`outputs/consequence-evaluator/ref7_2-hand-action-fixed-wrist-teacher-finger-20261010-r1/`.
The implementation and unit test are in commit `dda2835`.
