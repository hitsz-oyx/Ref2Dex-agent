# PointWAM video loss contract audit

Checked 2026-10-12 against the primary paper v1, local source at
`4ef2952d2dbe94373bdec8d86f1a1e252aafaef4`, and the active video experiment
card. This is a method-contract comparison, not a scientific verdict on video
supervision or an official PointWAM reproduction.

| Item | Published PointWAM | Local video point loss |
| --- | --- | --- |
| Motion magnitude | Adjacent-frame distance `||p[t+τ]−p[t+τ−1]||`, threshold 5mm. [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5) | Distance from the current query `||target[τ]−current||`, threshold 2mm by default/inherited config. [loss](../../src/task/goal-point-wam/src/goal_point_wam/point_supervision.py#L14), [config](../../src/task/goal-point-wam/src/goal_point_wam/model.py#L364) |
| Raw weight | `sigmoid(5*(δ/.005−1))`; no added constant floor. Static weight remains positive. [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5) | `.1 + .9*sigmoid(5*(δ/.002−1))` at the default floor; optional confidence multiplier. [loss](../../src/task/goal-point-wam/src/goal_point_wam/point_supervision.py#L30) |
| Reduction | Normalize scene weights over all valid point–step pairs in the batch; coordinate squared-error mean. [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5), [Eq.7](https://arxiv.org/html/2610.02840v1#S3.SS4) | Normalize points separately within each timestep, then equally average valid times per window and valid windows. Coordinate Huber mean. [loss](../../src/task/goal-point-wam/src/goal_point_wam/point_supervision.py#L38) |
| Target scale | Separate scene/hand coordinate means and standard deviations for every horizon; fine-tuning retains pretraining statistics. [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5) | Scene positions divided by one scalar `.05m`; no scene per-horizon statistical standardization in this loss. Trainer fixes `point_loss_scale_m=.05`. [trainer](../../src/task/goal-point-wam/tools/run/train_video_points.py#L113) |
| Scene clipping | Every scene displacement coordinate clipped to ±0.5m before standardization. [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5) | No equivalent displacement clipping in the inspected loss or video loader. Current observation workspace checking is a different operation. [loss](../../src/task/goal-point-wam/src/goal_point_wam/point_supervision.py#L25), [loader](../../src/task/goal-point-wam/src/goal_point_wam/video_data.py#L79) |

The local Huber operates on metric positions, but subtracting the same current
point from prediction and target leaves its residual unchanged; this is not
by itself an erroneous absolute-coordinate target. Default Huber transitions
at a 5cm coordinate residual. The paper instead applies squared error after
per-horizon statistical scaling. Statistical means cancel from their residual,
but coordinate/horizon standard deviations still change relative penalties.
These facts follow from the [local loss](../../src/task/goal-point-wam/src/goal_point_wam/point_supervision.py#L38)
and [paper B.5](https://arxiv.org/html/2610.02840v1#A2.SS5).

## Recorded intent and decision boundary

The [video card](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-video-points.md)
explicitly chooses common metric Huber supervision for native/video point
outputs, a fresh point head/optimizer, and a density invariant window loss.
The [loss docstring](../../src/task/goal-point-wam/src/goal_point_wam/point_supervision.py#L16)
specifies the nested reduction. Inspected records do not require exact B.5
weighting, clipping, or standardization. Therefore differences from PointWAM
are adaptations; they are not established implementation bugs. Exact weighting
constants should be recorded with the diagnostic rather than retroactively
describing earlier runs as a faithful B.5 reproduction.

From the formulas, a static raw weight is 0.006693 in the paper and 0.106024
locally; saturated moving/static ratios are approximately149.4 and9.43.
After per-time normalization, a common weight cancels: an entirely stationary
timestep still contributes a full local timestep mean. A point that moves and
then remains displaced retains high cumulative weights locally, whereas its
later adjacent weights return toward the static value. These are mathematical
implications, not measured causes of held-session error.

## Cheapest discriminating check and limitations

The root's bounded paired head-only Probe holds Huber/.05m, the .1 floor,
nested normalization, fixed final5000 backbone, 54 train/6 held windows,
2000 draws/seed288/lr1e−3 constant; it changes only cumulative2mm to
adjacent5mm motion weighting. Unknown previous targets get zero motion for
weight calculation while current valid supervision remains. This avoids
inventing a jump from a missing predecessor, but is an explicit missing-data
convention, not one specified by B.5. The first adjacent difference must use
the current query. Compare fixed-final held point EPE with stationary and
session summaries before further training; do not select a checkpoint on held
performance. The design and result belong in the
[video card](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-video-points.md).

This paired check tests one local weighting adaptation, **not full B.5**.
It changes motion definition and threshold together, so cannot attribute an
effect to either independently; it does not test the floor, global reduction,
MSE, statistical scaling, clipping, teacher geometry, or representation
learning. A frozen-head result is a single-seed diagnostic, cannot establish
robot/Cm utility, and cannot invalidate the published method. No official
implementation was inspected; this note verifies the published text and local
source, not undocumented author code. No training was run for this audit.
