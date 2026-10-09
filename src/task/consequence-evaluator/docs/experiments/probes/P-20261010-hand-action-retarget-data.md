---
schema: ref2dex.probe.v2
probe_id: P-20261010-hand-action-retarget-data
experiment_id: P-20261010-hand-action-retarget-data
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: a9f9f01713fda3f319949f001e1e352e461c756b
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 1
seed_pool: probe
seeds: [401, 402, 403]
decision_changed_if_positive: retain a full-action execution representation and run the GT-hand execution upper-bound gate
decision_changed_if_negative: diagnose action/state coverage before any H-to-hand, PointWorld or evaluator integration
status: RUNNING
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

Report per-split L1, wrist translation/rotation MAE, finger MAE, horizon-0 and
horizon-23 error, contact-onset and hold-stage finger error, action clipping,
mode/phase coverage, and hashes of every rollout and checkpoint. These are
execution-model diagnostics; they do not establish task success. If the fit is
usable, the next Probe will feed held-out GT future hand geometry into native
Gym and apply the predicted full action, with the existing teacher hold gate.

## Artifacts

Collection outputs belong under
`outputs/consequence-evaluator/ref7_2-hand-action-*-20261010-r1/`; the fit
belongs under
`outputs/consequence-evaluator/ref7_2-hand-action-fit-20261010-r1/`. The
collection entry point is `tools/run/run_hand_bridge_rollout.py --mode retarget`
and the fit entry point is `tools/run/train_hand_action_retargeter.py`.
