---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-progress-reference
experiment_id: P-20261013-goal-point-wam-progress-reference
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: ea2575aa
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 44
seed_pool: probe
seeds: [292]
decision_changed_if_positive: keep object-progress goals and retrain the actor on that index
decision_changed_if_negative: stop inference-only reindexing; the next change is the training label, not another clocked rollout
status: UNCLEAR
run_id: wam-ref2-execution-progress-20261013-r1
---

# Object-progress goals on the admitted realign actor

One 542-step closed loop. The actor and Chunk-R are the admitted realign pair.
The only change is the actor's object-goal index. Teacher and GT-tau rows stay
on their local oracle and are not queried with these goals.

## Decision Note

The anchored joint update failed its offline gate, so that pair is not
deployed. The reference clock is the next change. While the live object is
within 5 mm of reference[tick], goals stay at t+8/t+16/t+24, matching
training. Once it is behind, a monotonic cursor holds the last matching frame
and the three goals are 10/20/40 mm ahead of that pose, clamped so none is
more than 30 mm above it. On this reference that clamp lands on frame 62
(+15 mm) instead of the airborne clock frames. Success is actor hold of at
least 45 frames. Hold 0 means this frozen clock-trained policy does not
recover from the reindex; this probe does not start a retrain.

Weights are the realign actor
`outputs/goal-point-wam/wam-visited-realign-20261013-r1/selected.pt` and
visited R `outputs/trajectory-policy/wam-visited-R-20261013-r1`, with effect
checkpoint step 50500 and head
`wam-ref2-joint-multitakeover-20261012-r1`. The failed joint pair is not
used. One idle GPU, deadline 600 s. Code is `ea2575aa` plus the uncommitted
progress-reference cursor.

## Completed result

GPU2, 149 s, about 18.6 GB and 34% utilization. 542 controls finished.
Execution audit passed. Teacher hold 437 and places. GT-tau hold 438 with
no clipping, then misses the final placement, the same pattern as earlier
full episodes. Actor hold 0, clipping 41.0%. Minimum gap 0.24 mm at tick 50.
Height reaches 31.5 mm at tick 54 while the object is still table-supported,
returns to the table by tick 64, then is thrown to 680 mm at tick 210 and
falls.

The goal contract itself is active: through tick 49 the indices match the
demonstration clock, and from tick 50 they sit on frame 62 (+15 mm) instead
of the airborne clock frames. The actor command already differs from the
clocked realign partial at tick 6, while the object goal is still identical.
That prefix split is outside this index change, so the later throw is not
evidence for or against the progress cursor. Local judgment UNCLEAR. No
retrain on these labels.
