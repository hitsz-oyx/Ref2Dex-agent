---
schema: ref2dex.probe.v2
experiment_id: P-20261018-direct-tau-executor-r3
task: trajectory-policy
branch: main
git_commit: 3774db68
date: 2026-10-18
classification: Decision
status: UNPROMISING
hypothesis: After fixing reference-frame tau semantics, preserving tau temporal order, and training on multiple teacher sequences, a closed-loop executor can use GT tau as a high-level plan and emit direct active Inspire actions on an unseen sequence.
question: Does pi_exec(live state, reference-frame GT tau future8) preserve grasp/lift/hold on unseen s1_airplane_lift?
decision_changed_if_positive: advance from GT tau to WAM-predicted tau, keeping the direct closed-loop executor interface.
decision_changed_if_negative: do not interpret the old negative probe as a tau-plan refutation; first distinguish remaining BC/executor generalization limits from missing interaction context before spending PPO or WAM-C budget.
protocol: Sequence-level BC on 40 successful lift sequences from the RL660 teacher rollout, with 28/6/6 train/validation/test split; tau is computed only from the reference hand/object frame, the encoder preserves temporal order, and the executor receives no task goal or C input. Frozen direct-action execution uses 8 teacher calibration environments and 8 direct environments on corrected unseen s1_airplane_lift.
train_source: outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1
unseen_source: outputs/trajectory-policy/gt-tau-new-sequence-20261017-r4/teacher/trajectory.npz
selected_sequences: 40 RL660 lift sequences; s1_airplane_lift held out from the BC split
action_contract: policy emits 12 active native coordinates; passive coordinates are zero and Inspire coupling is applied once by DExplore.
input_contract: live measured state87 + reference-frame future hand tau8 + previous applied active action; no task goal and no interaction-context C.
gate: direct arm requires hold45 >= 6/8, success >= 6/8 and clipping < 1%; teacher is calibration only.
runtime: /home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train, Torch 2.0.1+cu118, GPU 0 with pre-existing 0.4 GiB context.
---

## Motivation and decision note

The previous direct-tau probe mixed teacher future world coordinates with the live
student hand, discarded tau time order with `mean(dim=1)`, and trained on one s3
motion duplicated across 32 environments. Its 0/8 physical result therefore could
not distinguish a bad tau plan from a bad executor contract or sequence shift. This
probe repairs those three implementation issues and removes goal conditioning and
oracle-C from the first clean test. The decision is whether this repaired BC
executor is sufficiently viable to justify PPO or WAM-predicted tau work.

## Repairs audited

The repaired contract is

\[
\tau_k = R^{ref}_t(h^{ref}_{t+k}-h^{ref}_t),
\]

computed from reference hand/object state only. The student live state remains the
feedback signal and is never substituted into tau. The tau encoder uses learned
position embeddings followed by an order-sensitive temporal MLP over the eight
future steps; it does not mean-pool the horizon. The BC source contains 40 distinct
RL660 lift sequences with sequence-level train/validation/test splits. The old
teacher-C arm is not part of this probe, so this run makes no C claim.

An independent audit recomputed tau at ticks 0, 1, 50, 100, and 400 for all eight
direct environments. The maximum absolute difference from the logged
`tau_features` was 0.0; the initial live direct state also matched the source
state exactly. This checks the repaired tau wiring, not task success.

## Offline BC

The best checkpoint was selected at step 800 by validation active-action L1.
Sequence-held-out validation active-action L1 was 0.04030 and the held-out test
L1 was 0.04584. Replay on the held-out unseen s1 source gave active-action L1
0.04047. This is a useful offline improvement over the old single-s3 unseen replay
L1 0.1121, but it is not closed-loop task evidence.

Artifacts:

- BC fit: `outputs/trajectory-policy/direct-tau-bc-r3-20261018-r2/`
- execution: `outputs/trajectory-policy/direct-tau-execution-r3-20261018-r1/`
- execution manifest: `outputs/trajectory-policy/direct-tau-execution-r3-20261018-r1/manifest.json`

The execution manifest records the pre-commit HEAD used when the run was launched;
the executed source files and their input hashes are preserved and were committed
as `3774db68` immediately afterward.

## Closed-loop results

On corrected unseen `s1_airplane_lift`, the teacher calibration was 8/8 hold45
and 5/8 success. The repaired direct-tau BC arm was 0/8 hold45 and 0/8 success,
with median held frames 0, clipping rate 0, and endpoint position error
0.0325 m. The direct arm never entered stable grasp in these eight environments.
The fixed BC executor therefore fails the predeclared physical gate and is
classified `UNPROMISING` for this implementation.

This result does not refute the proposition that tau can be a high-level plan. The
reference-frame and temporal-order audits pass, but the remaining physical failure
can still come from BC action generalization, the chosen state/action contract, or
missing interaction information. No PPO, WAM-tau, or WAM-C conclusion is made here.

## Limits and next action

This is a single-seed Probe with one unseen physical sequence and no C input. The
40-sequence source is selected from RL660 with lift heuristics; it is not a balanced
benchmark over object, subject, or motion families. The result is enough to stop
interpreting the old invalid 0/8 as evidence against tau semantics, but not enough
to spend PPO budget or close the tau-plan hypothesis. Before any online training,
the next comparison should use a demonstrably stronger closed-loop executor or a
proper current-state interaction oracle while preserving the repaired tau contract.
