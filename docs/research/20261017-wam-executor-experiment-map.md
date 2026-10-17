# WAM / executor experiment map

Date: 2026-10-17

This note is a navigation layer over the existing experiment cards. It does not
change historical card status and does not replace their manifests or outputs.
The cards remain the source of detailed protocol and evidence.

## Family index for maintenance

This is an index, not a merge of protocols. Each original card keeps its own
status, manifest, run id, and artifact links; the labels below only make the
current decision boundary easier to find.

| Family | Cards to read together | Current reading |
| --- | --- | --- |
| E1 executor interface | executor contract; generated-`tau` interface diagnosis/native execution; reference tracking/feedforward; competent-PD physical | Measured-state execution and fixed native dispatch are qualified locally. Generated-plan semantics still need a clean GT-to-predicted comparison. |
| E2 direct goal-conditioned BC/PPO | goal-conditioned BC → execution → PPO → PPO execution → finger/residual integration | Offline BC fit did not survive frozen physical execution; this exact direct-action recipe is closed for the declared motion. |
| E3 trajectory RL | decoder/BC initializers; trajectory PPO; long-credit; stochastic execution; step-replay | Historical initialization and direct trajectory-RL branches are closed or diagnostic; do not restart architecture or weighting sweeps. |
| W1 native PPO and adapters | independent-motion PPO; official native PPO; shadow BC/PPO; runtime correction; coordination bonus | Native PPO qualification and the official initialized checkpoint are useful baselines. WAM adapter gains remain unclear after runtime correction. |
| W2 WAM representation and teacher control | PTv3 pretraining; teacher/state/native coverage; current components; source transfer and pooling | Mostly offline or teacher-only evidence. It cannot be promoted to closed-loop WAM benefit. |
| W3 WAM goal semantics and interaction | task readout; learned short goal; v4 interaction head | The next semantic bridge is an achievable short goal. The v4 `C` head is an engineering interface until a matched executor test passes. |

The three active decision rows are therefore: (A) GT-`tau` executor
qualification, (B) an achievable short-goal bridge for WAM, and (C) a matched
`state+tau` versus `state+tau+C` executor test. Historical cards are not
rewritten to make these rows look cleaner.

## Current route decision

The active research question is no longer whether a hand trajectory can be
converted into a nominal command by a fixed redirector. The next decision is
whether a robot-specific, closed-loop executor can use a future hand plan as an
observation and produce the next native action directly:

```text
current measured state + future hand plan tau -> 12 active native controls
                                                        |
                                      fixed Inspire coupling -> 18D command
```

The executor should read the measured state again at every control step. It
should not use a nominal action, future robot state, teacher action, or a
residual added to a frozen redirector. A BC checkpoint is only a warm start;
PPO is considered after the BC policy passes a physical closed-loop gate.

The larger route remains:

```text
G_task -> realizable short goal -> WAM -> (tau, optional C) -> executor -> action
```

The current WAM-v4 interaction head and tokenizer are interfaces for later
matched tests. They are not evidence that `C` improves robot execution.

## Evidence that should remain active

| Role | Card | What it establishes | Limit |
| --- | --- | --- | --- |
| Contract | [P-20261010-goal-conditioned-executor-contract](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-goal-conditioned-executor-contract.md) | The shared goal executor reproduces the old requested-command semantics within numerical tolerance. | Engineering evidence only; no grasp result. |
| Local upper bound | [P-20261011-competent-pd-physical](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-pd-physical.md) | A qualified GT/PD route can execute a limited physical behavior. | Does not show predicted `tau` or generalization. |
| Reference baseline | [VAL-20261011-reference-goal-dagger-repeat](../../src/task/trajectory-policy/docs/experiments/validations/VAL-20261011-reference-goal-dagger-repeat.md) | The fixed reference-conditioned baseline is repeatable on its declared panel. | Fixed initial-state/reference panel; not multi-motion generalization. |
| WAM task interface | [P-20261016-wam-task-readout](../../src/task/wam-rl-policy/docs/experiments/probes/P-20261016-wam-task-readout.md) | Whole-task goals are often not realizable as the next short motion; a short-goal bridge is needed before using forecasts for control. | The readout itself did not qualify an online PPO route. |
| Runtime correction | [P-20261017-wam-runtime-correction](../../src/task/wam-rl-policy/docs/experiments/probes/P-20261017-wam-runtime-correction.md) | Earlier WAM adapter comparisons were confounded by residual duplication and batch history clearing. | Corrected bounded adapter remains `UNCLEAR`; no long continuation is authorized. |

These cards are calibration or boundary evidence. They should not be combined
into a claim that WAM already improves a robot policy.

## Routes that are closed or superseded for the current decision

| Route | Cards | Current interpretation |
| --- | --- | --- |
| Full-action hand retargeter | [P-20261010-hand-action-retarget-data](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-retarget-data.md), [P-20261010-hand-action-contact-context](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-contact-context.md) | The implementation and native action capture are real, but the physical gate was not met. Adding query-time contact proxies did not rescue it. Keep as prior evidence, not as the new baseline. |
| Learned `tau -> PD` inverse | [P-20261010-learned-retargeter-inverse](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-learned-retargeter-inverse.md) | `tau` changes predictions, but held-out first-control errors were too large for physical qualification. This does not disprove `tau`; it rejects that inverse under that data and conditioning. |
| Direct goal-conditioned BC/PPO | [P-20261011-goal-conditioned-bc](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-bc.md), [P-20261011-goal-conditioned-execution](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-execution.md), [P-20261011-goal-conditioned-ppo-execution](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-ppo-execution.md) | Offline action error was good, but frozen BC and PPO both failed the physical hold gate on the declared single-motion panel. Do not repeat the same data/interface recipe. |
| Nominal/residual continuation | [P-20261011-goal-conditioned-residual-integration](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-integration.md), [P-20261013-wam-residual-ppo](../../src/task/trajectory-policy/docs/experiments/probes/P-20261013-wam-residual-ppo.md), [P-20261014-wam-residual-curriculum](../../src/task/trajectory-policy/docs/experiments/probes/P-20261014-wam-residual-curriculum.md) | Some local contact basins exist, but the declared routes did not pass a robust task gate. They do not authorize another residual sweep. |
| Pure forecast execution | [P-20261010-generated-tau-native-execution](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-native-execution.md), [P-20261010-tau-geometry-tracking](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-geometry-tracking.md) | A GT upper bound can execute in a narrow setting; generated or history-only plans were not a deployable executor. Keep the GT/predicted gap as the next staged comparison. |

"Closed" here means closed for that exact protocol and evidence boundary. It
does not mean that a different data contract or executor is impossible.

## WAM representation work

[P-20261017-wam-v4-interaction-head](../../src/task/ptv3-shared-wam-v4/docs/experiments/probes/P-20261017-wam-v4-interaction-head.md)
is an engineering design for an object-centric interaction head. It has no
robot-benefit result yet. The remote tokenizer Probe completed with a hand
reconstruction error around 1.16--1.20 m and used only about 10--11 of 128
codes; this is not a positive qualification signal. We should therefore keep
the interaction output as a hypothesis and avoid spending a long pretraining
budget before the tau-only executor gate is known.

## Next decision card

The next experiment is recorded in
[P-20261017-gt-tau-executor](../../src/task/trajectory-policy/docs/experiments/probes/P-20261017-gt-tau-executor.md).
It is intentionally a small qualification probe:

1. Audit that the source sequence is training-eligible and that action labels
   are the actual pre-physics native controls. If the source is marked
   `training_allowed=false`, stop before fitting.
2. Fit matched state-only, GT-`tau`, and same-tick shuffled-`tau` BC arms with
   sequence-disjoint splits. Use the current measured state and a short future
   plan; do not use future robot state or teacher commands.
3. Run the frozen state-only and GT-`tau` policies closed loop on unseen
   sequences. Report hold, lift, task success, clipping, and plan/state drift.
4. Only if GT-`tau` passes the physical gate, warm-start PPO from that BC model.
   Then replace GT-`tau` with WAM `tau_hat` using the same executor and panel.

This separates three failure sources that were conflated in earlier cards:
plan information, inverse/action conditioning, and on-policy recovery after the
first action.

## Card maintenance rules for this route

- Do not rewrite historical status because a later route is better; add a
  synthesis note or a `superseded_by` link when needed.
- Treat `PROMISING` offline cards as offline or engineering evidence unless
  their card explicitly reports a qualifying frozen physical result.
- Split `GT tau`, `predicted tau`, and `tau+C` into separate decision rows.
- Use complete-sequence splits. Overlapping windows are samples, not independent
  demonstrations.
- The root `docs/experiments/INDEX.md` is generated. New cards belong to the
  owning Task and should be discovered by the index generator.
