---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-preserving-candidate-bank
experiment_id: P-20261010-history-preserving-candidate-bank
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: pending-clean-commit
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 3
seed_pool: probe
seeds: [411]
decision_changed_if_positive: audit the saved actor-H panels under the full H contract before any evaluator fit
decision_changed_if_negative: close this candidate-bank collection route and keep evaluator/selector integration frozen
status: PLANNED
run_id: history-preserving-candidate-bank-20261010-r1
---

# Does a history-preserving structured rollout retain H-matched candidate pairs?

## Motivation and decision note

The preceding offline audit found non-collapsed future hand trajectories and
different frozen U32 outcomes among hand/q-near states, but the original
structured packet did not save the actor observation `H`. That leaves open
whether the apparent candidate diversity survives the full evaluator input
contract. This Probe is the cheapest direct test: rerun the same structured
retarget protocol once with a bounded 64-env launch and save the actor
observation at every state frame. No model is fitted and no native execution
claim is made.

The result changes only whether a larger episode-split history-preserving
candidate panel is worth collecting. It does not reopen the failed R gate,
change the global Cm claim, or authorize online PointWorld/selector/MPC work.

## Frozen protocol

Use the frozen ref7_2 actor, motion, CPU tensor/GPU-PhysX runtime, structured
residual sampler, 542-step frame-0 episodes, and exact action-capture checks.
The only new field is `actor_observation[t] = obs["obs"]` aligned with each
saved trajectory frame. Runtime metadata records the one runner-hash refresh
needed because the old input JSON predates this optional capture field; no
frozen checkpoint or input file is modified.

The offline follow-up will compare pairwise H RMS, hand/q state distances,
24-step current-object-frame tau spread, and frozen 32-state U32 labels at
ticks 16/24/32. A positive result requires at least one tick with a usable
number of full-H-near pairs and strict label differences; the exact counts
are an engineering screen, not a validation gate.

## Stop conditions and boundary

Stop if the actor-observation array is not frame-aligned, any native command
is mutated/clipped, runtime identity changes, or all full-H-near pairs remain
label-tied. A positive screen authorizes only a larger history-preserving data
collection and contract audit. It does not authorize evaluator training or
selector deployment without a separate matched panel decision.

## Artifact

The collection entry point is
`src/task/consequence-evaluator/tools/run/run_hand_bridge_rollout.py`
with `--mode retarget --save-actor-observation`. The planned output is
`outputs/consequence-evaluator/history-preserving-candidate-bank-20261010-r1/`.
