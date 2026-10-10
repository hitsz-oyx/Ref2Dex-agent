# External repository cross-check (2026-10-11)

## Question

The question was whether other public repositories had already solved the missing
pieces around video-derived point dynamics, tactile/contact supervision, and wrench
conditioning.  This audit checks the public data and code contracts, rather than
assuming that a released checkpoint can be transferred to the native model.

The result is **not** that all other repositories fail.  Several provide directly
reusable contracts.  None is a drop-in replacement for the current native
`H=4,K=24` OakInk2/GRAB/ARCTIC contract, so each route still needs a matched Probe.

## Repositories and what they actually provide

| Source | Reusable contract | Direct gap for this Task | Action |
|---|---|---|---|
| [NVlabs/PointWorld](https://github.com/shawnyang99/pointworld) and its [DROID data release](https://github.com/NVlabs/PointWorld/tree/3872ec6ee73146aa671192ef79b5dfbedc0246e3) | RGB-D, calibrated geometry, robot action point-flow and 11-step scene-flow H5/WDS samples | Released clips are context+10 prediction steps, not native 4+24; no native object/action identity contract | Keep the verified `K_aux=10` video-data auxiliary route; do not load official weights as a proxy |
| [ZhuoyangLiu2005/T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex) and [dataset card](https://huggingface.co/datasets/zekaiwang/trex_dataset) | 5,464 episodes at 30 Hz; three RGB cameras; 58-D robot state/action; `(60,)` per-fingertip 6-axis tactile wrench; ten raw/deformation tactile videos; quickstart publishes camera calibration and URDF | Wrench is estimated at each fingertip, not a calibrated object-COM net wrench; the released model code is a VLA/tactile expert, not a point-flow head; full data is ~1.53 TB | Stream a bounded episode/metadata subset for a tactile+wrench temporal auxiliary; keep it separate from CHORD object-wrench claims |
| [tu-rbo/articulated-objects](https://github.com/tu-rbo/articulated-objects) and [RBO dataset page](https://tu-rbo.github.io/articulated-objects/) | 358 RGB-D sequences, tracked rigid/articulated object poses and meshes; 78 sequences include measured interaction wrenches and sensor/object calibration metadata | Only 67 minutes and mostly tool-mediated articulated-object interactions; no dexterous hand point-flow/action contract | Use as a small object-coordinate wrench/point-dynamics contract Probe if acquisition is within budget |
| [fafraob/point-lewm](https://github.com/fafraob/point-lewm) | Action-conditioned point-cloud JEPA world model, CEM latent planning, released point-cloud data/checkpoints | Simulated range-sensor observations and goal/planning schema differ from native per-point SE(3) effects; no tactile or CHORD labels | Reuse action-conditioned predictor/goal-head design as an architecture comparison, not a checkpoint transfer |
| ParticleFormer [paper](https://arxiv.org/abs/2506.23126) | Transformer over object/end-effector particles; hybrid Chamfer + differentiable Hausdorff loss; autoregressive five-step rollout | Public paper does not provide a drop-in repository/data adapter for this Task; its real setup is stereo+segmentation at 10 Hz | Add a bounded geometry-loss/attention comparison only after the current flow contract is stable |
| [Autumn1337/DynamicWAM](https://github.com/Autumn1337/DynamicWAM) | History optical flow plus displacement/interval/velocity/acceleration tokens with joint world-action attention | Video VAE and WAN action schema target dynamic-video action generation, not metric 3-D point flow | Reuse the kinematic-token ablation idea for a small point-flow head, without importing weights |
| [PointFlowMatch](https://github.com/robot-learning-freiburg/PointFlowMatch) | Conditional flow matching over point-cloud observations for RLBench policy actions | It is a policy/action generator rather than a future point-dynamics supervisor | Keep as a policy-side comparison only |
| [NVIDIA video-to-data CHORD](https://github.com/nvidia-isaac/video_to_data/tree/main/docs/chord) and [paper](https://arxiv.org/abs/2607.00033) | Public friction-cone contact-wrench-space support computation and RL reward | Requires object-frame contacts, normals, geometry and simulator sensors; DROID H5 alone has none of these | Use on the native simulator reward/auxiliary arm; do not fabricate CHORD labels from normalized glove pressure |
| [TouchAnything/EgoTouch](https://github.com/Jianyi2004/TouchAnything) and [EgoTac](https://github.com/Mr-Zwkid/EgoTac) | Processed hand-surface pressure/contact representations and temporal encoders | No verified object-world calibration or Newton force scale in the current downloads | Use pressure/contact auxiliary head first; fuse with native point cloud only after a calibrated bridge |

The T-Rex checkout also contains a reusable tactile representation
implementation, not only a dataset schema: `tactile_vqvae/` defines a per-hand
`[T,5,6]` force/torque encoder, EMA codebook and magnitude-weighted
reconstruction loss. Its intended downstream interface writes discrete
per-hand codes alongside each episode. This is a viable adapter design for F6
data, but the local audit has no T-Rex episode parquet/video and no checkpoint,
so it is evidence of an available representation contract rather than a result
on this Task.

## What this changes

The repository contracts were also checked from pinned local checkouts, not only
from project pages.  T-Rex `ba4047ce3e9f3d2013d60de600e136897e90ccb8` defines
`TACTILE_FORCE_DIM = 60` as two hands times five fingers times six force/torque
components, and its quickstart decoder loads one episode as a `(T, 60)`
`observation.tactile_force` array.  Its HDF5 writer stores each hand as
`(T, 5, 6)`.  RBO `b191d0cd3a076757fc60b8e1feed8bf5d0bd1103` documents the
358 RGB-D sequence/78 force-torque subset and its downloader exposes the
`with_ft` group; its visualizer reads the released
`ft_sensor_netft_data.csv`.  These are concrete reusable data contracts, not
claims that either repository already contains the current model's adapter.

The previous raw `jq_pressure` experiment used 512 byte channels and predicted
future raw glove values.  It did not use the released `(21,21)` pressure grids,
T-Rex wrench signals, or RBO object-frame wrench labels.  Its persistence result
therefore closes only that raw-sensor recipe.

The strongest new data route is T-Rex for temporal hand/tactile/wrench
representation and RBO for a small calibrated object-wrench contract.  The
T-Rex VQ-VAE code gives a concrete way to turn the per-fingertip F6 stream into
a compact auxiliary token once a bounded episode is available.  The
strongest point-dynamics design references are PointWorld's released H5/WDS
contract and ParticleFormer's correspondence-free hybrid geometry supervision.
CHORD remains a simulator-side contact-wrench reward/representation, not a
supervised target that can be read from DROID or EgoTouch by itself.

## Decision

Keep the mainline split explicit:

1. Run the already audited official PointWorld H5 `K_aux=10` data route as a
   video-data auxiliary pretraining arm.
2. Build the pressure-map adapter from existing local EgoTouch files; compare
   persistence, history+hand, aligned future-hand and masked pressure/contact
   prediction. The bounded processed-history and future-hand probes are now
   negative, so do not silently promote them to a native Cm claim.
3. If that signal is learnable, add T-Rex metadata/one-episode streaming as a
   separate tactile+wrench auxiliary arm.  Do not download the full 1.53 TB
   release.
4. Add CHORD support matching only in calibrated simulator/native data.  If a
   small RBO subset is acquired, use it to validate the object-wrench coordinate
   and unit contract before claiming transfer.

All four routes require matched downstream `Cm-on`/`Cm-off` evaluation before
they can support the Mission claim.  Repository availability is evidence that
the route is implementable, not evidence that it improves the final policy.
