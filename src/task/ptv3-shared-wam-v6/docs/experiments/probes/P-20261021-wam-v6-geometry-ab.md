---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-geometry-ab
experiment_id: P-20261021-wam-v6-geometry-ab
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: deb1298b9fe02c51e37c5f425884fdc7ca501450
claim_id: C1
hypothesis_family: HF-v6-geometry-ab
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171,246]
decision_changed_if_positive: retain geometric supervision only if matched continuation and common-frame rollout errors improve
decision_changed_if_negative: reject this geometry implementation or optimization setting as an improvement over the parent
status: AWAITING_RESOURCE_PLAN_REPLY
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


## Rollout results and engineering evidence

Inference COMPLETED on GPU7, code b6e9be29 plus bounded continuation fix deb1298b;
peak allocated436.53MiB, no interference with active GPUs4/5/6 or other users.
First pass completed all four GRAB cases for each model, then failed in apple
fixed-goal inference because an unused controller was observed
after terminal state. Fix stops consulting the controller in fixed replay mode;
resume skips completed immutable cases and only computes missing apple modes.
Logs/first attempt preserved; result continuation_commits records both versions.

CPU initial-builder check reproduces native sample_window exactly for geometry,
joints, skin and histories; goal difference7.45e-9m. Static skin transport error
7.45e-9m. All eight GRAB generated-history/reconstruction audits pass: stored
hand histories and coordinate reconstruction differences0; object reconstruction
max3.26e-8m; raw metric reproduction difference0mm. No future recorded hands or
skins are inserted into subsequent histories. Complete v6 suite31 passed8.12s,
including exact equality of control/base loss and coordinate-head gradients,
rejection of any geometry graph in control training, and retained DEV diagnostics.

Common11-point mean errors (mm), forecast24/consume8/predicted history:

| TRAIN case | Old v4-80000 | Parent v6-4371 | Geometry v6 +4500 | Geometry versus parent |
| --- | ---: | ---: | ---: | ---: |
| eyeglasses | 67.07 | 90.36 | 88.40 | -2.18% |
| bowl | 220.00 | 292.50 | 302.08 | +3.28% |
| stamp | 52.19 | 179.84 | 195.66 | +8.80% |
| bunny | 43.79 | 178.22 | 228.01 | +27.94% |

All common frames retained180/44/180/161; no numerical/workspace stops. Compared
with old v4, geometry errors increase31.8%,37.3%,274.9%,420.7%. Geometry endpoint
errors88.63/886.68/250.09/995.21mm. Native v6 uses observed joint21/skin while v4
has11 points, so architecture/input differences preclude loss-only causation.
Same-architecture parent comparison still has unmatched extra training time.

Apple sparse11 input diagnostic:

| Goal/event mode | Old v4 mean mm | Parent4371 | Geometry4500 | Coverage |
| --- | ---: | ---: | ---: | --- |
| fixed replay, unknown event | 674.87 | 285.26 | 228.32 | all647 predicted frames |
| TaskController, unknown event | 664.59 | 260.58 | 330.26 |447 common frames, timeout15s |
| TaskController, planned event | 664.59 | 179.88 | 318.39 |447 common frames, timeout15s |

Both scheduler variants establish grasp0 ticks and stop acquire_timeout at450.
Geometry improves means versus old v4 by66.2%/50.3%/52.1%, but remains severe:
fixed-goal endpoint1272.47mm, scheduler endpoints581.93/540.23mm. Versus parent,
fixed replay improves20.0%, scheduler unknown worsens26.7%, planned-event arm
worsens77.0%. Planned events are additional v6 input, not an identical v4 event
condition. Apple has sparse11 only, missing skin/extra joints explicitly masked;
this cannot establish full native-v6 robot deployment efficacy.

Artifacts: evaluation/result.json, comparison.json, data-audit.json, raw npz
and seven four-panel videos (recorded / old-v4 / parent4371 / geometry4500).
The model named geometry4500 means4500 new-objective updates after initializing
from4371; no claim that this is the latest active step. Current run continued
untouched to5620 at14:14, loss9.44425; this operational observation is separate
from the frozen4500 evaluation. No files/checkpoints were deleted by this task.

## Matched-control readiness, not yet a scientific result

New flag geometry_loss_disabled=true is the sole training config difference.
It removes the geometry training graph entirely and sets its training weight0;
DEV still computes all geometry diagnostics with zero objective contribution.
Same actual entrypoint and8→10 strict-recovery stages; all source weights,
normalizers and fresh-optimizer semantics unchanged. Preserve every control
checkpoint using a separate CLI32GiB/stage ceiling (retention is operational).
Queue compares fixed DEV500/4500 automatically as files arrive and verifies
initial hand DEV against exact4371 within.01mm. Frozen dataset/seed equality
and per-metric coverage are checked; no hidden single-GPU accumulation.

Concrete queue command/config diff/environment saved in control-queue-plan.json.
User was asked to choose automatic three-card launch after natural release or
prepare-only. No response received yet; no watcher/training started without
that resource-plan answer. Current GPU7 contains another user's6.7GiB process,
so it was used only for bounded~0.44GiB inference. GPUs0–3 untouched. The matched
control is still pending; do not claim the loss alone caused observed decline.

Additional Decision diagnostic before interpreting UNKNOWN-event rollouts:
repeat the same four GRAB cases with the recorded next-contact/release plan
supplied to both v6 checkpoints, separately labelled planned. This distinguishes
an absent planned-event instruction from a failure of the native conditioned
model; a positive result would narrow the drift finding to unknown-event use.
The frozen teacher event timeline is an oracle instruction, just like the
recorded t+24 object endpoint; it is not generated-state scheduler feedback or
an extra measured hand history. Do not mix this arm with the old v4 interface
comparison. Reuse the same GPU7 budget, eight short sequences, preserve all
previous outputs and add separate planned video filenames.


Native planned-event GRAB arm completed with79dae6b5. Parent4371→geometry4500
means in mm: eyeglasses162.59→181.38 (+11.56%), bowl298.25→286.39 (-3.98%),
stamp134.02→188.94 (+40.98%), bunny236.79→259.43 (+9.56%). Same180/44/180/161
common frames, no stops; geometry endpoints134.62/867.36/290.17/857.55mm.
Supplying the recorded future-event instruction does not remove the drift
problem in these four cases; do not present UNKNOWN-only failure as the entire
v6 result. This oracle instruction diagnostic still does not isolate loss
causality from further training. Separate planned videos are complete.


All11 individual four-panel MP4s and two combined GRAB MP4s completed and passed
ffprobe dimensions/fps/frame-count checks;15fps playback of30Hz trajectories
is2x slower. Selected late bowl/apple frames visually inspected. Total evaluation
artifacts remain below50MiB, far below1GiB ceiling. Extended16 GRAB history/
coordinate/metric audits pass. Matched-control results still pending the user's
resource-plan answer, not silently approximated with existing older baseline.
At14:22 active geometry run is RUNNING6000 (loss7.31889, preclip norm854.83),
free192.85GiB; no signal or STOP was sent to it or any other user's task.
