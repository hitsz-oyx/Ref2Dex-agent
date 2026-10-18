# 2026-10-18 v4 input/label contract audit fixes

`task`: `ptv3-shared-wam-v4`  
`code_commit`: `812c26aa` (fixes; audit baseline `c0f5907b`)  
`environment`: local `graspenv` (Python 3.8, PyTorch 2.4.1+cu121) for v4 model tests;
local `fastwam` (Python 3.10, PyTorch 2.7.1+cu128) for reader/online tests.

## Scope

This note records the concrete fixes made after the second v4/GT-tau audit. It does
not claim that the running remote v4 pretraining is invalid, and it did not modify
or restart that run.

## Fixed

1. **18-D target/static features are unified.** Main3, native and the online
   runtime now set `features[:,12]=1` for the target-only view and compute
   `features[:,13]` from the same H4 displacement threshold (`0.002 m`). Native
   previously zeroed both; the online path set only target.
2. **C voxel origin comes from the current observation.** Both readers pass
   `origin = min(current interaction points) - 5cm`. `interaction_field` no longer
   derives an origin from future t+1. The sample schema is
   `ref2dex.wam-v4-target-object.v4`; `INTERACTION_ORIGIN_SEMANTICS` is recorded.
3. **Moving-object non-contact windows are separate.** `classify_window` returns
   `moving_object_noncontact` when the hand is still but the object moves; it has
   its own retention ratio (`moving_object_keep_ratio`, default 0.35) instead of
   the 10% `static_idle` rate.
4. **GT-tau action prefixes are canonical.** The closed-loop runner records the
   action before the simulator step and passes a clone to `pre_physics_step`,
   which rewrites wrist channels in place. The oracle does the same. The manifest
   now records `action_prefix_matches_executed`.
5. **Oracle packets separate current and future object pose.** `object_pose` is
   read before the teacher fork; `future_object_pose` is stored separately.
6. **Realtime hold metrics.** `success_frames` now counts frames with
   `goal_distance <= 0.05`; `success_frames_ever` keeps the upstream cumulative
   flag, and `longest_hold_frames`, `first_success_tick`, `drop_events_after_success`,
   `final_goal_distance_m` and the final state after the last action are recorded.
7. **v4 online adapter.** `wam_rl_policy.wam_v4_runtime` loads a v4 checkpoint,
   builds the 512/32/128 target-only batch from H4 observations, accepts a
   scheduler same-ID `goal_delta` or position-only goal, applies a content-bound
   dense overlay, and exposes `encode()` summary and full `tau`/`C` separately.
   `FeatureController` accepts v4 only through this prepared-geometry path and
   marks its summary mode as `v4-summary-encode-only`.
8. **Tau contract.** `wam_rl_policy.tau_contract` prepends `tau(0)=0`, interpolates
   v4's 30 Hz/24-step anchor-frame hand trajectory onto the 60 Hz/8-step student
   clock, selects the right/left slot and rotates displacement to world when
   requested. Slicing the first eight v4 rows is explicitly tested as wrong
   (32 mm vs 16 mm for a constant 0.12 m/s motion).
9. **Conditional contracts.** Invalid C queries are cleaned before the query MLP;
   valid NaN fails closed; non-unit probability scales are rejected; v3 warm
   start checks full PTv3/voxel/scene configuration; inherited `.pt` files are
   read with their portable selection sidecar; the fast filter can apply the dense
   surface overlay.

## Verification

- Reader tests: 28 passed (`3d-data-preparation` v4 data/interaction/lookup/windows).
- Online/contract tests: 31 passed (`wam-rl-policy` fast_bc, features, runtime,
  tau, action, scheduler).
- Model/checkpoint tests: 19 passed (`ptv3-shared-wam-v4` contract, audit
  regressions, resume selection).
- A real native window with the dense overlay produced
  `surface=512, flow=32, interaction=128`, `target=1`, `static=0`, a current
  origin and nonempty contact labels.
- The same real physical window built by the native reader and by
  `v4_online_samples` matched on `xyz`, 18-D features, hand/history, flow and
  interaction points/normals to `max |diff| <= 6.6e-17`.
- On 20 random real native windows, replacing the old future-origin with the
  current origin shifted the origin by 5.52 mm on average and changed 2884 of
  61440 contact labels (4.69%); the largest single-window change was 444 of 3072.

## Consequences and open items

- `data_identity` now includes `input_semantics` and `interaction_semantics`.
  A checkpoint created before this commit cannot silently resume on the changed
  input/label view. It can still be used as `--init-v4` model-only
  initialization for a fresh run: weights load strictly, but optimizer, step,
  RNG and historical best selection reset. That new run is a new trajectory and
  its curve is not comparable to the old run. The remote run was not touched.
- The fixed C labels and features change the training distribution; pre/post
  curves are not one continuous trajectory.
- The PointWorld Hilbert serialization depth still depends on the batch maximum
  grid coordinate. This is a structural batch-composition risk; it was not
  changed because it needs an existing-weight batch-sensitivity measurement.
- `V4Runtime` exposes full `tau`/`C`, but no executor policy has been trained on
  that v4 output yet. The existing summary PPO path still uses `encode()`.
- Scheduler goals are accepted as SE(3) point deltas, but the distribution match
  between scheduler local goals and real t+24 training goals has not been measured.
- C remains a parallel interaction head; it does not condition on a candidate
  action and cannot be used as a counterfactual action scorer.
