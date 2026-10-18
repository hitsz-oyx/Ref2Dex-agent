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
- The current contract-fixed run still uses the pre-fix window filter manifests.
  `rebuild_v4_window_filter_transition.py` can recheck only the old blocked
  windows and write a new filter; a clean run should regenerate the recipe/stats
  and start a new data identity rather than replacing filters under a live run.
  A remote probe on the first 2000 of 1,587,404 TRAIN blocked indices took 1.57 s
  and restored none (1755 `approach`, 245 `static_idle`). This is only a prefix,
  not a corpus statistic; it mainly shows that a full transition scan is cheap
  enough to run separately.
- C remains a parallel interaction head; it does not condition on a candidate
  action and cannot be used as a counterfactual action scorer.

## Follow-up: running training and online geometry identity

The contract-fixed production run is pinned to remote commit `2183eb46`,
`pretrain-nine-v4-contractfix-warm13000-20261018-r1`. Its checkpoint identity
records `v4.target-static-h4-2mm.current-hand.v1` inputs and
`v4.current-interaction-origin-minus5cm.v1` labels. At the follow-up check it
reached step1860 with finite loss1.18921/gradient4.59354, about1.605 updates/s
and a provisional remaining13.52h. This is numerical/operational evidence,
not proof of downstream execution quality. Keep this run and its frozen filters;
the coverage question below does not establish that the run is invalid.

An independent CPU-only full filter transition scan was launched against
`b9c0800d` with Python3.8.20, NumPy1.24.4 in remote `graspenv`. Run ID is
`filter-transition-20261018-r1`, supervisor2644469, two processes, one-hour
deadline and20GiB free-space guard. Inputs are the unchanged audit-bound
recipe/dense overlay and old TRAIN/DEV filters; outputs/status/logs live under
the NAS root's `data/filter-transition-20261018-r1/`. The active training code,
recipe, normalization and filters are untouched. It rechecks1,587,404 TRAIN
and276,176 DEV blocked indices. TRAIN results are **pending**, so prefix
counts are not used to decide on a training restart. Decide whether to change
coverage after completed counts by source and quality-admission exclusions are known.
This scan uses conservative AABB contact candidates, not exact contact labels.
It is a Decision probe: distinguish negligible coverage change from a material
overall or per-source omission. The former supports retaining the frozen run;
the latter motivates a separately versioned coverage run. Rechecking old blocks
with the metadata classifier is cheaper than regenerating all C labels or
launching another training run before knowing the affected counts.

The full DEV scan completed in215.12s:1,476 old blocked windows now classify as
`moving_object_noncontact`, and426 pass the deterministic0.35 retention ratio.
All426 survive the independent quality-admission exclusion mask:417 from
OakInk2 and9 from UniDexGrasp. Admitted DEV grows545,459->545,885 (+0.0781%).
This is a coverage calculation, not a measurement of model quality, and cannot
be extrapolated to TRAIN. `dev-coverage.json` beside the new filter records old
and new admitted counts by source. The transition field `restored_counts` is
misnamed in the existing tool: it counts **all rechecked classes**, not just
restored indices; use `restored` and the coverage report for restoration counts.

The online overlay previously accepted a correctly hashed NPZ whose32 flow
queries had all moved1m. Unlike the training overlay, it did not compare them
to the original geometry. The runtime now requires explicit32 distinct cached
flow queries, rejects missing/changed overlay queries, and requires payload
digests for v2 manifests. When a checkpoint binds a dense-manifest digest,
runtime uses that binding automatically; an explicit relocated manifest is
accepted with the same content, while a different digest is rejected. Legacy
v1 overlays remain readable when the checkpoint has no dense binding. Existing
v3 cache builders must supply the original flow queries for use with v4;
resampling32 new identities from a512-point surface is no longer accepted.

Regression tests first reproduced missing/changed query acceptance and the
unbound checkpoint geometry, then passed after the runtime fix. The related
local model/data/runtime suite passed103 tests (8 warnings) in33.20s, using
Python3.8.20, PyTorch2.4.1+cu121 and NumPy1.24.4. A subsequent focused model
check covers the already-present `scene_token_count` in real v4 forward output;
it was incorrectly reported missing during review and requires no model fix.
No simulator rollout or new neural training was launched for this runtime fix.
