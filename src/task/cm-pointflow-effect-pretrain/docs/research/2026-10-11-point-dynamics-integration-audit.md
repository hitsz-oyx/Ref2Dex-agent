# Point-dynamics integration audit (2026-10-11)

## Scope and decision

This audit asks which parts of the ref10 point-dynamics route can be borrowed
without importing a checkpoint or changing the native `H=4,K=24` contract.  It
reads the current PointWorld adapters and pinned public source checkouts.  The
public checkouts are only an audit record under `tmp/`; no external source is
vendored or modified.

The cheapest useful next step is a **video-only motion-token Probe**, followed
by a matched correspondence-free geometry-loss Probe if the token arm is
implemented cleanly.  The current native rigid-effect loss should remain the
primary objective.  A conditional flow-matching policy objective is not a
replacement for future point dynamics and should remain deferred.

## Current implementation contract

### Native PointWorld adapter

`oakink_wm.pointworld.PointWorldWM` projects the 18-dimensional scene feature
contract through `scene_proj` (lines 114-115), appends 24×2×11 observed-future
hand/action tokens, and restores the PTv3 features to every original point
after voxel pooling (lines 141-190).  The output is a rigid SE(3) effect per
object and horizon (lines 191-196).

The loss at lines 198-217 already has exact point correspondences: the target
is obtained by applying the ground-truth object transform to the same 512
canonical points, and the prediction applies the predicted transform to those
points.  Its point term is a motion-weighted Huber in normalized flow space.
Adding an unordered loss here would throw away identity information and should
not replace this term.  A small geometry term can be an auxiliary only when the
labels are weak tracks without reliable correspondence.

### Temporal adapter

`pointworld_temporal.py` keeps physical time in sparse keys at every pooling
level (`TemporalGridPooling`, lines 29-68) and uses a separate action summary
for each of the 24 future horizons (lines 147-164).  This fixes cross-time
voxel collisions, but it is not an observed-motion history encoder: the
`history` arm receives only the current scene.  There is no interval-level
velocity/acceleration token in this route.

### Weak video adapter

`video_points.sample_window` selects points using all four history frames and
constructs per-point `[zero, velocity, acceleration, displacement, kind/static
metadata]` features (lines 20-42).  `VideoPointWorldWM.forward` accepts only
`xyz`, `features`, and `point_valid`, applies the scene projection and PTv3,
then predicts per-point 24-step flows with a time embedding (lines 26-48).
Thus the current video model already sees local history kinematics, but it has
no explicit global interval token and no set-level geometry supervision.
`balanced_flow_loss` in `video_points.py` balances object/background groups and
is the right masking contract to preserve for any auxiliary loss.

## Public implementation evidence

The following audits were made at fixed local commits:

| Source | Commit | What is reusable | Boundary |
|---|---|---|---|
| [Point-LeWM](https://github.com/fafraob/point-lewm) | `d2ba2f7e3a1cacdddb3b2991dcc7bd463b733a48` | Fixed affine coordinate normalization, FPS centers, fixed-radius local groups, LayerNorm PointNet tokens, sinusoidal center position, ViT/CLS latent, action-conditioned AR JEPA predictor | It predicts latent cloud states for simulated range-sensor planning; it is not a per-point metric-flow head and its checkpoints are not native weights |
| [DynamicWAM](https://github.com/Autumn1337/DynamicWAM) | `db91c4fe856b42d5500150e11e4f86f407f21dbc` | Exact-time 12-D displacement/duration/velocity/acceleration contract, normalization and invalid masks, MLP motion tokens inserted as a separate token sequence | Its flow maps and WAN action expert target video-action generation, not 3-D point effects |
| [PointFlowMatch](https://github.com/robot-learning-freiburg/PointFlowMatch) | `d46cc98cf07a15fc264dddfddf5085880749a0ab` | Conditional flow-matching interpolation and Euler sampling for point-cloud-conditioned robot action | It is a policy generator: `z_t=t z_1+(1-t)z_0`, target `z_1-z_0`, and action-state loss; it does not supervise future point clouds |
| [ParticleFormer paper](https://arxiv.org/abs/2506.23126) and [project page](https://suninghuang19.github.io/particleformer_page/) | paper only | Hybrid Chamfer plus differentiable soft-Hausdorff loss and five-step autoregressive rollout over unordered particles | No public drop-in adapter was found; correspondence and compute contracts differ |

### Point-LeWM

`pc_encoders/pointvit_encoder.py` implements one fixed physical grouping stage:
FPS centers, ball-query neighbors, a LayerNorm PointNet, and a coordinate
position code before a ViT transformer and CLS readout (source lines 112-138,
247-363).  The coordinate affine is fixed by config and explicitly avoids
per-cloud normalization (lines 60-65, 256-263), which is compatible with the
current fixed PTv3 workspace principle.  `module.py::ARPredictor` adds learned
temporal positions and conditions a causal transformer on action embeddings
(lines 300-341).  This is a useful architecture comparison, but replacing the
current PTv3 encoder would be a high-cost route with no native transfer proof.
Its ball-query subsample is stochastic in both train and eval
(`_group`, lines 397-413), so it would also need an explicit deterministic
evaluation variant before being used in a reproducible matched Probe.

### DynamicWAM

`absolute_motion.py` defines twelve named features:

```text
mean displacement x/y, mean displacement magnitude, p99 displacement,
delta-t, mean velocity x/y, mean speed, p99 speed,
mean acceleration x/y, mean acceleration magnitude
```

`models/absolute_motion_tokens.py::AbsoluteMotionTokenModule` validates the
`[B, history_count, 12]` shape, uses persistent train statistics, masks invalid
intervals and acceleration channels, then applies `Linear -> SiLU -> Linear`
plus position/type/invalid embeddings (source lines 11-121).  In
`models/small_wam.py::_build_action_tokens`, these motion tokens are inserted
between the state/action core tokens and registers (lines 487-533), so motion
magnitude/timing is available to every joint-attention block rather than only
to a final head.  This is the most direct low-cost idea for the current weak
video arm: derive the same interval statistics from the four history frames,
then add one global token sequence without importing WAN or video weights.

The released twelve-dimensional contract is image-plane (`x/y` displacement,
velocity, and acceleration), while this task's tracks are metric 3-D.  A probe
must declare this adaptation explicitly: either project to a fixed camera
`x/y` plane and keep 12-D, or use a 15-D 3-D analogue (three vector components
plus magnitude/p99, `delta-t`, speed/p99, and acceleration magnitude) with a new
normalization schema.  It must not label a 3-D feature vector as the exact
DynamicWAM contract or silently discard depth.

### ParticleFormer

The paper states a hybrid loss

`L = alpha * CD + (1-alpha) * HD`

where Chamfer covers average local nearest-neighbor alignment and a smooth
Hausdorff term prevents large structural misses.  It applies this to raw point
streams and autoregressively rolls out five steps.  This matters only for weak
or unordered point labels.  The current native OakInk2/GRAB/ARCTIC object
points are correspondence-preserving, so the Huber flow term remains the
identity-preserving anchor.

### PointFlowMatch

`pfp/policy/fm_policy.py::FMPolicy.calculate_loss` samples Gaussian noise,
forms a linear interpolation between noise and a future robot-state sequence,
and regresses the constant velocity with a global point-cloud observation
condition.  `infer_y` Euler-integrates the velocity field.  This is suitable
for an action policy head, but using it as the native future-point target would
require a new stochastic set-valued state contract and would make the current
deterministic object-effect metrics incomparable.  The repository is GPL-3.0;
do not copy code into this task.

## Recommended low-cost probes

### Probe A: DynamicWAM-style history token (preferred next step)

**Hypothesis.** The existing point-wise history features contain motion, but a
shared interval token improves long-horizon weak video flow by exposing global
motion magnitude and timing.

**Preparation.** Extend the video packer (without changing point selection) to
store three history intervals for `H=4`.  For each interval and point group
(object/background), compute either the declared 12-D camera-plane statistics
or the declared 15-D metric-3-D analogue from history points and actual history
timestamps only.  Store train-only mean/scale and valid/acceleration masks.  Do
not use any future target frame to choose points, normalize, or fill interval
features.

**Model arm.** Add a small `Linear(D_motion,128) -> SiLU -> Linear(128,128)` token
module with interval position and invalid masks.  Mean or attention-pool the
object/background tokens into a global history token and add it to
`local[:, :, None] + time_embedding` in `VideoPointWorldWM`; keep the existing
point features and head unchanged.  A strict `motion_token=False` control must
zero this new path while reading the same pack.  The forward contract should
explicitly include the new history tensor, so the old three-key boundary cannot
silently change.

**Decision gate.** Reuse the existing h8 weak-video protocol and source split:
500 updates, seed 226, same static/CV baselines, one free GPU and the existing
15-minute budget.  Require the pre-registered object clip-macro improvement
against the better baseline, no qualifying clip degradation over 20%, and
background EPE no worse than 1.2× static.  A positive result justifies native
encoder-transfer testing; a negative result closes this *global-token recipe*,
not all point dynamics.

### Probe B: correspondence-free geometry auxiliary

Run only on weak video targets or another explicitly unordered contract.  For
each valid `(sample, group, horizon)` select a deterministic maximum of 64
points from the existing mask, form predicted and target positions as
`xyz + flow`, and compute:

```text
CD = 0.5 * (mean_i min_j ||p_i-q_j|| + mean_j min_i ||q_j-p_i||)
HD_soft = 0.5 * (tau logsumexp_i(min_j ||p_i-q_j||)/tau
                  + tau logsumexp_j(min_i ||q_j-p_i||)/tau)
L_geom = alpha * CD + (1-alpha) * HD_soft
```

Use a numerically stable smooth maximum (`tau*logsumexp(distance/tau)`) with a
fixed metre-scale `tau` (start at 5 mm), divide both terms by the existing
10 mm flow scale, and average object/background groups exactly as
`balanced_flow_loss` does.  Invalid future tracks must be removed before
`cdist`; never multiply NaN coordinates by a zero mask.  Start with
`alpha=0.8`, `lambda_geom=0.05`, horizons h8 and h24 reported separately.

The geometry arm must be matched against the current correspondence-Huber arm,
with the same initialization, windows, and checkpoint schedule.  Keep the
correspondence Huber term as the primary loss so object identity is retained.
`torch.cdist` plus the hybrid term was smoke-tested on CPU for finite forward
and backward gradients with B=2, h8/h24 and 64/128 points; this is an
engineering smoke, not evidence of learning.

### Deferred routes

* Do not replace PTv3 with Point-LeWM's FPS/ViT encoder until a token or
  geometry auxiliary gives a positive signal; that comparison is expensive and
  would confound representation with supervision.
* Do not add PointFlowMatch's stochastic flow field to the native world model;
  reserve it for a later policy-side action-generation comparison.
* Do not add ParticleFormer's loss to native rigid effects without a matched
  unordered-label test; the native 512 canonical points have exact identity.

## Evidence and source paths

* Current native code: `src/oakink_wm/pointworld.py`,
  `pointworld_temporal.py`, `video_pointworld.py`, and `video_points.py`.
* Audited external source snapshots: `tmp/repo-audit-20261011/point-lewm`,
  `DynamicWAM`, and `PointFlowMatch`.
* ParticleFormer source: <https://arxiv.org/html/2506.23126>.

The audit changes the next-step choice from “search for a complete replacement”
to “test a global history token while retaining the validated point-flow
contract.”  None of the sources supplies official weights used by this route.
