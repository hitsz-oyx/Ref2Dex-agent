---
schema: ref2dex.probe.v2
probe_id: P-20261009-act-receding1
experiment_id: P-20261009-act-receding1
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: b880148
claim_id: C3
hypothesis_family: HF-consequence-act-proposal
probe_index_in_family: 5
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain newest-first-action replanning and diagnose temporal mixing as a behavior confound
decision_changed_if_negative: rule out executing only one newest action as a remedy for this frozen checkpoint and prioritize deployment-history diagnosis
status: UNCLEAR
run_id: act-receding1-20261009-r1
---

# Can one newest ACT action per query recover grasp?

## Motivation and decision

The user asks whether executing one step each time changes the previous
negative result. Previous `temporal1` queried every step but **averaged** up to24
covering predictions; it did not test `a_t = proposal(H_t)[0]`. This Decision
Probe distinguishes temporal-mixing lag from inaccurate latest-action behavior,
serving the self-trained manipulation subgoal.

The cheapest test adds `receding1` to the existing executor and runs one native
full542-step screen. If it restores held45 while temporal1 did not, mixing is a
specific confound worth investigating. If it still fails despite a passing
teacher and exact dispatch, reducing the execution horizon to1 is insufficient
for this checkpoint; do not rerun aggregation modes or retrain in this Probe.

## Fixed contract and stopping conditions

Keep the exact historical r1 proposal SHA256
`ed26abd5a02b9c706750b94ea28100f028564ee4bfd9054fabf3eb6a54c3d707`
and stored normalization, frozen self-trained actor, seed282, native GPU PhysX
and GPU tensor pipeline, four environments/64 actor copies each, original
spacing and same teacher/ACT/repeated teacher/repeated ACT roles. Predict24
controls from the current raw observation every tick, discard older chunks,
and execute only the newest horizon0 control. No blending, training or Y change.

One idle GPU2, one542-step launch,150s worker deadline and210s process deadline,
outputs below100MB. Stop on input/source drift, nonfinite control, early done,
resource conflict or timeout. The CPU audit reconstructs requested native
dispatch from saved chunks and is not neural computation. Compare descriptively
with previous frozen-checkpoint screens; unexposed PhysX state is not restored,
so this remains engineering-only, not a matched causal/Gate1 claim.

Audit542 queries, at most1 active chunk, `actions[t,ACT] == chunks[t,ACT,0]`,
current-frame raw input, native/requested control identity and repeated ACT
stream. Teacher held45 is required to interpret ACT failure; ACT held45 is
the screening signal and full-task success is reported separately.

## Tools and artifacts

- [Executor and tests](../../../tests/test_action_chunk_execution.py)
- [Native worker](../../../tools/run/run_gate1_gt_progress.py)
- [Independent execution audit](../../../tools/audit/audit_action_chunk_execution.py)

Run: `outputs/consequence-evaluator/act-receding1-20261009-r1/`, with immutable
command/source/checkpoint manifest, native packet and worker log. Audit:
`outputs/consequence-evaluator/act-receding1-audit-20261009-r1/`.

## Limitations / future evidence

One launch cannot establish repeatability or a general negative about ACT.
Horizon0 queries now include non-stride8 histories outside the original
training-query grid. Deployment-conditioned supervision, new training, other
weights and stricter simulator control comparisons are deferred.
