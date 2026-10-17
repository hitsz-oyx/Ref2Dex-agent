---
schema: ref2dex.probe.v2
probe_id: P-20261017-gt-tau-executor
experiment_id: P-20261017-gt-tau-executor
date: 2026-10-17
task: trajectory-policy
branch: main
git_commit: PLANNED
claim_id: C3
hypothesis_family: HF-trajectory-policy-executor-transition
probe_index_in_family: 1
seed_pool: probe
seeds: [to-be-assigned]
decision_changed_if_positive: use a direct full-action executor as the BC-to-PPO replacement for the old redirector, then compare predicted tau against GT tau
decision_changed_if_negative: do not retrain WAM or add interaction heads; diagnose executor state/action coverage and label eligibility first
status: PLANNED
run_id: not-run
predecessor_cards:
  - P-20261010-goal-conditioned-executor-contract
  - P-20261010-learned-retargeter-inverse
  - P-20261011-goal-conditioned-bc
  - P-20261011-goal-conditioned-ppo-execution
---

# GT-tau closed-loop executor qualification

## Decision

This probe tests whether a fixed Inspire-specific executor can use a future hand
plan as an observation and produce the next native action directly. It is a
qualification gate for the new BC-to-PPO route, not a WAM validation and not a
claim about multi-hand generalization.

The intended causal contract is:

```text
measured state at t + tau[t+1:t+K] -> actor -> 12 active native controls
                                                   -> fixed 18D coupling/dispatch
```

The actor is queried again after every simulator step. It does not receive
future robot state, teacher action, phase, clock, force, or a nominal action
from the old redirector. The first probe uses `K=8`; the existing 24-step
trajectory can be retained for a later horizon comparison.

The reusable contract is implemented in
[`tau_executor.py`](../../../src/trajectory_policy/tau_executor.py). Its unit
tests check finite batched inputs, bounded twelve-coordinate output, and the
single expansion point into the eighteen-coordinate native layout. This is an
engineering contract only; no training or physical result is implied by the
module or its tests.

## Data and model contract

- `state_t`: measured `q`, `dq`, current hand keypoints, current object pose and
  velocity, and the previous actually applied native command.
- `tau_t`: GT future hand displacement in the existing object-frame convention.
- label: the actual native command captured at the pre-physics boundary, using
  the twelve active coordinates; fixed Inspire coupling reconstructs the
  dependent coordinates.
- split: complete sequence or environment split; no overlapping windows may
  cross train and validation or test.
- training eligibility: source manifest must explicitly allow training. An
  engineering-only packet or a source with `training_allowed=false` can be
  used for an oracle diagnostic only, never for BC or PPO fitting.

## Matched arms

| Arm | Input | Purpose |
| --- | --- | --- |
| A | measured state | state-only control baseline |
| B | measured state + GT `tau` | executor qualification |
| C | measured state + same-tick shuffled GT `tau` | checks that B uses plan information |

All arms use the same architecture, optimizer, normalization, split, and seed.
Offline action loss is only a screening metric. The decision metric is frozen
closed-loop behavior on unseen sequences.

## Gates and follow-up

The BC stage must be finite, replayable, and materially better for B than A;
the shuffled arm C should lose the corresponding plan benefit. Physical
qualification requires finite native commands, clipping below 1%, and at least
80% of the teacher's declared hold rate on the same held-out panel. Hold, lift,
task success, maximum held duration, and plan/state drift are reported
separately.

If B passes, fit PPO from B's checkpoint with the same observation and action
contract, then run a separate frozen evaluation. If B fails, do not start PPO,
WAM retraining, or `tau+C`; first determine whether the failure is caused by
label coverage, state distribution shift, or an inadequate plan horizon.

## Evidence boundary

This probe can establish GT-plan executor qualification only. It cannot establish
that WAM predictions are sufficient, that `C` helps, or that the policy transfers
across robot embodiments. Those require later matched cards with the same
executor and a declared predicted-`tau` input.
