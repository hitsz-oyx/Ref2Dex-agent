---
schema: ref2dex.probe.v2
probe_id: P-20261009-ref13-panel-inventory
experiment_id: P-20261009-ref13-panel-inventory
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 89b2a87
claim_id: C3
hypothesis_family: HF-native-gpu-execution-contract
probe_index_in_family: 5
seed_pool: debug
seeds: []
decision_changed_if_positive: recover a complete historical same-prefix candidate panel and recompute the current physical-reference/TCC Y
decision_changed_if_negative: discard the historical summary as Gate1 input and keep strict native-GPU Gate1 blocked
status: UNPROMISING
run_id: ref13-panel-inventory-20261009-r1
---

# Is the complete ref13 candidate panel recoverable?

## Decision Note

The native GPU execution probes do not provide a strict same-state candidate
contract. The old ref13 record was the only plausible saved-panel shortcut, so
this read-only inventory checks whether its complete `32 anchor × 7 candidate`
panel can be reused with the current physical-reference bank and TCC delta-
progress labels. No simulator, GPU, training, or source-worktree write is used.

## Inventory result

The original panel files are absent from the current worktree, the baseline
worktree, and the searched `/home2/wyy/oyx_ws` tree. The historical paths named
in the logs were:

```text
/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/cm-interaction-oracle/
  oracle-y-utility-s263-sync-{reference,repeat,merged-candidate1..6}/panel.pt
  oracle-y-utility-extra-s264-sync-{reference,repeat,merged-candidate1..6}/panel.pt
```

The surviving `tmp/ref13` files are only batch summaries, implementation-review
text/JSON, and `raw_stat_replay.json`. The raw replay summary reports 32 anchors,
baseline `Z=23/32`, GT-Y `Z=24/32`, and GT-Z upper `Z=25/32`; it contains two
opportunity rows and 14 aggregate intervention records. It does not contain the
per-anchor candidate controls or the full future arrays. Its source hash is
`fce5ebab1e4ae0e0ecb891c557f459c3c4148c13093d9ddb254c4ac47a3ece1a`.

The historical `panel.pt` contract required `before`, `history`, `actor_obs`,
hand/root and 11-point geometry, native q, candidate actions/PD targets,
90-step height/pair traces, prefix errors, timestamps and candidate deltas.
None of those complete tensors can be reconstructed from the surviving
summaries. The full audit manifest is
`outputs/consequence-evaluator/ref13-panel-inventory-20261009-r1/audit.json`.

The old panel also used GPU PhysX with a CPU tensor pipeline and one PhysX
thread. Even if the binary panel were recovered, it would require an explicit
contract audit before reuse; its backend is the behavior-mismatched host route
for the current policy and it predates the current physical-reference/TCC
label contract.

## Decision

The historical summary is not a recoverable Gate1 dataset. Do not promote its
`32×7` headline, two opportunity rows, or old host-backend result into current
physical-reference/TCC evidence. Keep Y, the reference bank, TCC, and policy
weights frozen. Strict Gate1 remains blocked pending a verifiable native-GPU
execution contract or an explicit Decision Checkpoint that changes the claim.

This is an inventory result, not evidence that the Value definition or Cm
hypothesis is false.
