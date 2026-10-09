---
schema: ref2dex.probe.v2
probe_id: P-20261010-hand-action-retarget-data
experiment_id: P-20261010-hand-action-retarget-data
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 5e62a1787f2a27bf5e83f9374b835a906335978b
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 1
seed_pool: probe
seeds: [401, 402, 403]
decision_changed_if_positive: retain a full-action execution representation and run the GT-hand execution upper-bound gate
decision_changed_if_negative: diagnose action/state coverage before any H-to-hand, PointWorld or evaluator integration
status: UNCLEAR
run_id: hand-action-retarget-data-20261010-r1
---

# Structured full-action rollouts and hand retargeter

## Motivation and decision note

The ref7_1 upper-bound probe found repeatable wrist-control signals but did not
establish the finger/contact mapping. This Probe tests the cheaper next question:
can a standalone execution model recover the commanded native action from a
future 11-point hand trajectory and the current robot state? The result changes
whether we spend the next budget on GT-hand execution or return to contact
control diagnosis; it does not change the global Cm claim or old Y/U.

The minimal discriminating experiment is an episode-split dataset of ordinary
single-world rollouts. A frozen self-trained actor supplies the nominal action;
bounded structured residuals cover wrist translation, wrist rotation, finger
open/close, asymmetric finger preload, and combined wrist/finger motion across
approach, contact, and hold phases. The saved target is the actual full native
action after composition and clipping. No future command, object force, or
actor action is an input to the retargeter.

## Frozen protocol

Three independent launch splits are planned: 96 train, 64 validation, and 64
test episodes, each with a distinct seed and complete 542-step frame-0 episode.
The native runtime remains the existing `graspenv` Torch 2.4.1 route with GPU
PhysX and CPU tensor exchange, one airplane motion, no forks, and early
termination disabled. Every tick records `q_t`, `dq_t`, 11-point hand geometry,
the captured 18-dimensional native action, clipping, structured mode and phase.

Training windows use stride 2, no reset crossing, and the contract

```
(hand[t+1:t+25] - hand[t], q[t], dq[t]) -> action[t:t+24]
```

The model is a two-layer temporal decoder with 24 learned horizon queries. L1
is computed after train-only normalization independently for every horizon and
action coordinate. Validation selects the checkpoint; test is read only after
freezing it. The offline result can only be `PROMISING`, `UNPROMISING`, or
`UNCLEAR`.

## Resource and stop conditions

GPU2 is the sole requested device. Each launch is bounded to 900 seconds and
2 GiB of new artifacts; the fit is bounded to 2400 steps/600 seconds. Stop if
the runtime identity, action capture, state alignment, residual bounds, or
episode split contract fails. Do not connect H, PointWorld, evaluator, RL, or
old Y during this Probe.

## Evidence to report

## Results and attribution

The three structured launches completed without early termination or clipping:
96/64/64 episodes, each 542 actions, with `(hand, q, dq, action)` shapes
`(543,N,11,3)`, `(543,N,18)`, `(543,N,18)`, `(542,N,18)`. Every declared mode
and phase was observed; the phase counts were 120/120/302 ticks. The captured
action was exactly the clipped nominal-plus-residual command. The rollout
trajectory hashes are recorded in each manifest.

The first fit (`ref7_2-hand-action-fit-20261010-r1`, 2400 steps) produced
test L1 `0.01637`, wrist translation/rotation MAE `0.01071/0.00848`, finger
MAE `0.01975`, contact-onset/hold finger MAE `0.01815/0.02156`. A follow-up
fit added three preserved teacher motion anchors to train, one to validation,
and the held-out source teacher to test, with unique launch-directory episode
IDs. This reduced test L1 only to `0.01596` and contact/hold finger MAE to
`0.01750/0.02108`; it is still offline fit evidence only.

The aligned GT execution gate used the follow-up checkpoint and frozen source
hand futures. The teacher retained 481 frames; the three R roles retained
10/30/21 frames with hand-coordinate RMSE `44.68/43.59/57.71 mm`, so none met
the `>=90% teacher` and `<40 mm` gate. Native action capture was exact and all
clipping counts were zero. A receding-query diagnostic (`query_period=1`) did
not rescue the representation: teacher 482 frames, R roles 0/0/8 frames and
RMSE `177.12/226.69/207.27 mm`. This rejects “24-step dispatch alone” as a
sufficient explanation, but does not prove that all hand geometry is unusable.

The decisive limitation is attribution: ordinary structured rollouts fit the
command numerically, yet contact/preload state is not recoverable from the
current `future hand displacement + q_t + dq_t` contract. The preserved teacher
anchors modestly improve offline error but do not pass native Gym behavior.
Therefore this Probe is `UNCLEAR` overall, with the current retarget execution
route `UNPROMISING` for the hard gate. No H-to-hand, PointWorld, evaluator, or
Cm claim is authorized from these results; the next work returns to the
ref7_1 finger/contact-command diagnosis.

## Artifacts

Collection outputs belong under
`outputs/consequence-evaluator/ref7_2-hand-action-*-20261010-r1/`; fits are
`ref7_2-hand-action-fit-20261010-r1` and `-r2`. GT execution outputs are
`ref7_2-hand-action-exec-20261010-r3` through `-r6`; r4 is the first aligned
execution, r5 is the teacher-anchor follow-up, and r6 is the receding-query
diagnostic. The collection entry point is
`tools/run/run_hand_bridge_rollout.py --mode retarget`, the fit entry point is
`tools/run/train_hand_action_retargeter.py`, and the execution entry point is
`tools/run/run_hand_action_retargeter.py`.

The r3 pre-alignment run is retained as implementation evidence only. All
later runs verify input hashes, exact native dispatch, no clipping, and the
teacher behavior screen; none is a formal task-success or Cm validation.

## Source/backend isolation follow-up

The contextual checkpoint was then tested against a fresh teacher packet made
with the same CPU tensor pipeline and direct four-env actor layout as the
retarget runner. The packet is
`outputs/consequence-evaluator/ref7_2-same-cpu-teacher-packet-20261010-r1.pkl`;
its source rollout kept all four baseline roles for 481--484 held frames and
records the CPU/GPU-PhysX provenance explicitly.

With the source env0 future broadcast (the original runner contract), the
contextual R roles held `6/9/76` frames and had hand RMSE
`61.43/75.83/77.40 mm`. This is materially better than the frozen GPU-source
run (`0/0/0`, `109.03/111.12/113.91 mm`), so the source backend/layout mismatch
was a real confound; it is still far from the `>=90%` and `<40 mm` gate.

A single matched-layout diagnostic used each source env's own future instead
of broadcasting env0. It produced R held `0/0/0` and RMSE
`117.23/101.24/81.34 mm`, with no clipping. The same CPU teacher packet itself
has 26--37 mm hand variation between envs, and repeated baseline env0 has
about 42 mm RMSE against the saved source, so this is not a same-state causal
validation. It does show that a per-env future substitution is not a rescue.

The follow-up therefore remains `UNCLEAR` for attribution and `UNPROMISING`
for the native execution gate. No further query-period or contextual-fit
sweep is warranted; future contact/preload work must use source-matched replay
and record native contact-force proxies before making a representation claim.

## Same-CPU contact-proxy audit

To separate the remaining attribution question from another retarget sweep, I
audited the already-completed same-CPU packet without launching simulation.
The source teacher has a saved `pair` proxy, but the R execution packet has no
R-side pair, contact force, or impulse label. The audit therefore finds the
first contiguous execution `surface_gap <= 10 mm` run and reports its exit;
it does not call that exit a recovered contact-loss event.

For all three R roles, the first near-gap run starts at tick 43, before the
teacher `pair` proxy turns on at tick 45. The runs end at ticks 67/70/136,
then the sampled gap is 15.87/14.07/11.76 mm while the teacher pair proxy is
true. The corresponding full-trajectory hand RMSE against the broadcast
source is 61.43/75.83/77.40 mm; at the geometry-near exit, wrist errors are
5.56/6.03/30.99 mm and finger errors are 17.01/24.72/51.09 mm. Same-env
teacher q/dq and requested-action differences are descriptive only because
these environments are not same-state counterfactual replays. Source trajectory
and source packet fields were bitwise checked; zero clipping
and the existing exact requested/actual-action contract remain intact.

This audit makes the next blocker more specific but does not identify a
mechanism: the saved geometry proxy shows a short near-gap interval followed
by divergence, while the packet cannot tell whether preload, contact manifold,
or another hidden simulator state caused it. Result and hashes:
`outputs/consequence-evaluator/ref7_2-contact-proxy-audit-20261010-r14/`.
No new native run, force claim, H-to-hand integration, PointWorld inference,
or evaluator training is justified by this proxy-only result.

## Decision note: source-matched force capture

The remaining blocker is observability rather than another R architecture
sweep. The Isaac task already exposes hand-body `_contact_forces` and target
`_tar_contact_forces`, but the existing R packet discarded them. One bounded
follow-up is authorized: replay the same CPU-pipeline/direct-actor layout,
seed, source future, checkpoint, query period, and four-env control flow while
also saving those vectors and an R-side force-pair boolean. The output changes
only logging, not the requested commands or policy.

Success means the arrays are finite, state-aligned (543 state frames/542
commands), shape-checked, and exact action capture remains intact; then the
force/pair transition can be compared descriptively with the existing gap
exit. Failure or a mismatched source/backend stops this branch. The run is
CPU tensor exchange with GPU PhysX, bounded to one 542-step execution and
less than 300 seconds/2 GiB. It cannot by itself establish a preload
mechanism or authorize H-to-hand/evaluator integration.
