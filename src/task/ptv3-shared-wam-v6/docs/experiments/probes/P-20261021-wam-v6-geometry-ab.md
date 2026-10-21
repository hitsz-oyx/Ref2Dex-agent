---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-geometry-ab
experiment_id: P-20261021-wam-v6-geometry-ab
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-v6-geometry-ab
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171,246]
decision_changed_if_positive: retain geometric supervision only if matched continuation and common-frame rollout errors improve
decision_changed_if_negative: reject this geometry implementation or optimization setting as an improvement over the parent
status: PREPARING
run_id: geometry-ab-rollout-20261021-r1
---

# Matched geometry ablation and old-video rollout comparison

User requests two Decision Probes: distinguish a geometric-loss effect from
ordinary matched continuation degradation, and measure predicted-history drift
against the old videos. No changes to the active frozen geometry run or other
users' processes. No source data/checkpoint deletion. Keep all new control
checkpoints, bounded independent storage budget and20GiB floor.

The cheapest distinguishing setup reuses original fixed DEV panels at500/4500
and existing four frozen GRAB TRAIN anchors plus the apple TaskController case.
Geometric500/4500 checkpoints are hard-linked into independent pinned/ storage
before the active run's existing rotation; no extra payload copy or modification.

## Rollout

GPU7 small-batch inference sharing only available memory (other process6.7GiB,
17GiB free at preflight); never stop or modify that process. Single model at a
time, float32 inference matching old videos, seed246;30min inference budget,
1GiB artifact ceiling,20GiB free-space floor. CPU rendering uses two ffmpeg
threads. Fixed parent4371 and geometry4500 compared with old v4-step80000.

Same four GRAB sequences, source anchors, initial H4 poses/hands, forecast24/
consume8, true object t+24 endpoint instruction, rigid Kabsch object feedback,
common11 scoring points and common consumed frames. Full native joint21 and
initial observed skin are available for v6 GRAB; generated skins are transported
from generated joints, never refreshed from future GT. Initial online inputs
must reproduce the training reader. Event UNKNOWN on this old-interface
comparison, not teacher future-contact labels. This is a model comparison,
not a matched-architecture proof of a geometry-loss effect; the parent v6
provides the relevant same-architecture comparison.

Apple reuses identical initial measured11 points and object H4. Missing10
joints and skin are explicitly invalid, with no invented robot FK. Compare
fixed replayed goals and per-generated-tick TaskController with unknown events;
add a labelled scheduler-planned-event arm for v6's intended extra condition.
Motion-only feedback, no forces or physical policy. Do not merge these arms.
Stopped tails are marked absent, common-frame errors report coverage/stop and
must not be advertised as full-length success. Four TRAIN cases do not establish
generalization, and reproduction distance alone is not physical invalidity.

Outputs: outputs/ptv3-shared-wam-v6/geometry-ab-rollout-20261021-r1/.
Pinned sources are in pinned/; eval and videos in evaluation/. Each prediction
artifact retains generated history/deltas for audit. No MANO display fitting.

## Matched continuation control

Proposed waiting plan: allow current geometry run to finish naturally, confirm
its owned processes exited and GPUs4/5/6 available, then start the control on
the same three cards. Await user's resource-plan answer before enabling an
automatic launch. Never silently substitute single-GPU accumulation or stop
an existing run. Prepare the concrete executable while waiting.

Use exactly source4371 model/normalizer weights, fresh Adam/RNG/step/warmup/
selection, same seed170, DEV171, TRAIN207579/DEV3356, local96/global288,
8 workers/rank, LR1e-5/warmup50, event dropout and original draw streams.
Same8-update initial stage,2-update strict recovery and sustained continuation
bound10000 updates/4h. Disable only training geometric loss/its autograd graph;
retain geometric diagnostics during evaluation. Same fixed DEV initial/20/
500/4500 and intervening500 steps. Preserve all new checkpoints under a declared
separate storage budget; checkpoint retention is an operational difference,
not an objective/data/model change. Compare physical metrics, not totals of
different objectives. Record exact config diff and checkpoint/source identities.

No result yet: initially geometric-run DEV4500 is worse than exact4371 baseline,
but that alone does not isolate the loss's causal effect. Both source4371
and geometric500/4500 remain preserved independently of active run rotation.
