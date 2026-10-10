# Touch, contact pressure, and point dynamics review

Date: 2026-10-11
Task: `cm-pointflow-effect-pretrain`
Purpose: decide how real tactile supervision can improve the point-dynamics route in `ref10.md`, and how the requested CHORD wrench idea should enter the experiments.

## Evidence from released primary implementations

### TouchAnything / EgoTouch

The authors' repository is [Jianyi2004/TouchAnything](https://github.com/Jianyi2004/TouchAnything). Its current data contract is unusually useful for an auxiliary contact experiment:

* each episode is HDF5 with three synchronized RGB streams (`chest_color`, `left_color`, `right_color`), 42 bimanual 3-D joints, masks, camera poses, and timestamps;
* the tactile labels are `pressure/left_pressure_grid` and `pressure/right_pressure_grid`, each `(T, 21, 21)` and normalized to `[0,1]`;
* the pressure group records `grid_size`, `separate_normalization`, and tactile/bend maxima as attributes;
* the released files are nominally 30 FPS and trajectories have variable length;
* the model uses a shared DINOv2 image encoder, learnable view embeddings, cross-view attention, and view dropout. The latter is a useful control for testing ego-only versus wrist-view conditioning.

The project page states that the glove records dense continuous pressure and that all modalities are synchronized at millisecond precision. The repository also says that the data upload may be incomplete, so a run must record the exact dataset revision and validate episode lengths, timestamps, and pressure attributes before training. Sources: [repository data format and training instructions](https://github.com/Jianyi2004/TouchAnything#-data-format), [project description](https://jianyi2004.github.io/TouchAnything/).

This is **real pressure supervision**, but it is not automatically a Newton-valued 3-D force or a simulator contact wrench. It should be treated as a normalized hand-surface pressure field. Converting it to force requires a per-sensor area and calibration model; the public contract does not provide enough evidence to make that conversion for the current task.

### EgoTac

[EgoTac](https://github.com/Mr-Zwkid/EgoTac) provides a second, independently implemented route. Its processed loader expects Zarr arrays `ego_cam_rgb`, `mano_tactile_left/right`, and `mano_contact_left/right`, with each tactile/contact target shaped `(T, 778)` on a MANO surface. The repository explicitly separates continuous force-like supervision from binary contact supervision and maps mixed data sources to a common MANO surface. It also states that processed Zarr training does not require raw HDF5/SVO conversion or the ZED SDK. Source: [processed-data contract](https://github.com/Mr-Zwkid/EgoTac/blob/main/docs/process_data_aether.md).

This makes EgoTac a good source for a compact contact-plus-intensity target, but its hand-surface coordinate system is different from the 21x21 palm grid in EgoTouch and from the robot point cloud. It should not be concatenated as if it were a common sensor without an explicit projection or separate head.

### FeelAnyForce

[FeelAnyForce](https://github.com/prgumd/FeelAnyForce) is a practical calibration reference rather than a drop-in dataset for our point model. The authors collect tactile images, depth images, and 3-D force vectors from a robot pressing a GelSight Mini, train a multi-head transformer, and expose a calibration procedure. The released dataset schema includes image paths and a six-dimensional force/torque vector; the README recommends a separate fixture and calibration recording for transferring to another tactile sensor. Sources: [paper/project description](https://www.prg.cs.umd.edu/FeelAnyForce), [code and calibration instructions](https://github.com/prgumd/FeelAnyForce#calibration).

The reusable lesson is to keep a calibration transform and force units explicit. A normalized pressure-map head and a force/wrench head must not share an implicit scale.

### CHORD contact wrench

[CHORD](https://arxiv.org/abs/2607.00033) represents contact effects in object coordinates with a six-dimensional wrench (force and torque), rather than rewarding only contact position. The paper defines the reference using hand keypoints and object poses, extracts contact locations, normals, and friction constraints, and compares the induced object-level effect. Its ablation reports a consistent gain of wrench guidance over position-only and no-contact guidance on the evaluated tasks. NVIDIA's [open project documentation](https://github.com/nvidia-isaac/video_to_data/blob/main/docs/chord/index.html) describes the same representation and the associated video-to-data pipeline.

CHORD is therefore relevant to the **robot/simulator arm of this project**, where contact points, normals, forces, and object poses are available. It does not justify inventing Newton forces from EgoTouch's normalized glove grid. For human pretraining, pressure/contact should be an auxiliary representation; for simulator RL, wrench can be a separate reward or prediction target.

## What can be reused in this repository

The current point-dynamics model can use a separate tactile auxiliary head without changing the native point-flow contract:

1. Keep the existing point-flow target and evaluator unchanged.
2. Add an auxiliary target with either `(left,right,21,21)` EgoTouch pressure grids or a fixed-size hand-surface vector. Predict a binary contact logit and a nonnegative pressure/intensity value.
3. Use a masked loss:

   `L_touch = BCE(contact, contact_hat) + lambda_p * contact * smoothL1(log_pressure, log_pressure_hat)`.

   The contact mask prevents the many zero-pressure cells from dominating regression. Compute normalization from the training split only; preserve the release's per-sensor normalization attributes in the manifest.
4. Align windows by the released timestamps, not by array index alone. Resample only after checking monotonicity and gaps. Use future pressure/contact at the same physical horizon as the point-flow target; do not silently treat 30 Hz human sequences as 10 Hz DROID clips.
5. Use view dropout as a matched ablation: ego-only, ego plus wrist RGB, and pressure/contact auxiliary loss. This tests whether the tactile signal helps the shared dynamics representation rather than merely adding camera context.
6. For CHORD, first add a simulator-side `wrench` target/reward from contact geometry and force sensors. Compare `point/contact position`, `point+normal`, and `point+normal+wrench` under the same policy training budget. Keep the wrench in object coordinates and normalize force and torque separately per task/sequence as CHORD does; do not mix pressure-grid units with Newton units.

## Decision and limitations

The previous raw-sensor experiment predicted 512 glove channels. It did not test the released pressure-map/contact representation, so its persistence result cannot reject tactile supervision. The first useful probe is a small EgoTouch processed-HDF5 audit and a short auxiliary pretraining run, with no official model weights. A second probe should use simulator contact data to test CHORD wrench conditioning or reward. Cross-embodiment pressure-to-wrench transfer remains unsupported until a calibration/force pairing is available.

The TouchAnything README currently says dataset upload is in progress; record the exact file revision and reject incomplete episodes during audit. EgoTac's MANO targets and EgoTouch's pressure grids are complementary sources, not directly interchangeable labels.

## Source links

* [TouchAnything code and HDF5 schema](https://github.com/Jianyi2004/TouchAnything)
* [TouchAnything project page](https://jianyi2004.github.io/TouchAnything-Website/)
* [EgoTac code and processed Zarr contract](https://github.com/Mr-Zwkid/EgoTac)
* [FeelAnyForce code, dataset, and calibration](https://github.com/prgumd/FeelAnyForce)
* [CHORD paper](https://arxiv.org/abs/2607.00033)
* [NVIDIA CHORD documentation](https://github.com/nvidia-isaac/video_to_data/blob/main/docs/chord/index.html)
