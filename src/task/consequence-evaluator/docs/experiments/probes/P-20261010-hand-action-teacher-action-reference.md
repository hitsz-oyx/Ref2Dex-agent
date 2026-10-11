---
schema: ref2dex.probe.v2
probe_id: P-20261010-hand-action-teacher-action-reference
experiment_id: P-20261010-hand-action-teacher-action-reference
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: d8db591
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 5
seed_pool: probe
seeds: [282]
decision_changed_if_positive: establish same-process teacher-action cloneability and attribute the fixed-wrist failure to the wrist replacement
decision_changed_if_negative: do not attribute any native failure to the fixed-wrist decoder without a same-state clone control
status: UNCLEAR
run_id: ref7_2-hand-action-teacher-action-reference-20261010-r3
---

# Can same-process test envs replay the complete live teacher action?

## Decision note

The fixed-wrist/live-teacher-finger audit still held only `8/10/9`, so it did
not establish whether the wrist replacement or the multi-env execution
contract was responsible. This is the cheapest remaining control: keep the
same native process and initial reset, but broadcast the complete current env-0
teacher action `0:18` to env1--3. It is an attribution/same-state audit, not a
retarget model or deployment experiment.

If the three test envs reproduce teacher behavior, the fixed-wrist replacement
is implicated by the preceding negative. If they do not, the clone contract
itself is unresolved and no decoder-level causal claim is allowed.

## Contract and protocol

- Source packet and inputs are the same source-matched CPU contract used by
  the two preceding audits; seed `282`, GPU2, four roles, 542 controls.
- Env0 remains the native teacher. At every tick env1--3 receive the current
  live env-0 action for all 18 coordinates; no model action or fixed-wrist
  replacement reaches the test envs.
- The expected engineering contract is finite, requested equals applied, and
  no clipping. Teacher hold is the primary comparison; hand RMSE to the
  separate source packet is diagnostic only because the source packet is not
  the same live rollout.

## Scope boundary

This privileged control does not establish deployability, a hand
representation claim, or a Cm result. It cannot distinguish hidden PhysX
state effects from reset variation if the clone fails. No further native
decoder sweep, R training, H-to-hand, PointWorld, evaluator, selector, MPC,
or Cm integration is unlocked by this audit.

## Result

The first two launches were stopped before dispatch by the new runtime identity
guard: the source packet declares legacy `tensor_device: cuda:0`, while the
actual CPU tensor pipeline reports task device `cpu`. They produced no behavior
data and are not scientific runs. The corrected r3 records both values and
compares the invariant backend fields (`name`, pipeline, sim device, and
PhysX-GPU flag) plus direct-row actor layout.

In r3 the env-0 teacher held `482` frames. Broadcasting its complete live
action produced holds `482/318/482` for env1--3, with zero clipping and exact
requested/applied controls. All four envs were bitwise identical at reset, but
their q/hand/object states diverged at tick 1 despite identical actions. Thus
two rows reproduced the teacher behavior, while one row exposed immediate
GPU-PhysX row-level divergence; this is partial cloneability, not a strict
same-state A/B control.

The result makes fixed-wrist replacement a strong practical suspect because
the preceding teacher-finger oracle failed in all three rows, but it does not
causally isolate the wrist decoder from hidden backend state. The native route
therefore remains closed at the combination/contract level.

## Artifacts

The output is task-owned under
`outputs/consequence-evaluator/ref7_2-hand-action-teacher-action-reference-20261010-r3/`.
The preflight-failed r1/r2 directories are retained. The runner option and
tests are in commit `d8db591`.
