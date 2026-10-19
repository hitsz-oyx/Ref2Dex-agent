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
and276,176 DEV blocked indices. Both scans completed; decisions below use
whole-split counts and independent quality-admission exclusions.
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

The full TRAIN scan completed in1231.32s:8,839 old blocked windows now classify
as moving-object non-contact, restoring2,471 at the fixed0.35 retention ratio.
All2,471 survive quality admission:2,454 OakInk2,15 UniDexGrasp,2 HOT3D.
Admitted TRAIN grows3,218,740->3,221,211 (+0.0768%); even the largest relative
source increase is only0.1064% (OakInk2). These are small overall and per-source
coverage changes. Keep the active run and frozen filters; the count result
does not justify stopping/restarting it. Preserve the new candidate filters
separately for the next data identity. It does not prove that rare release or
rolling cases are unimportant to a targeted downstream task.
Local evidence copies of `train-coverage.json` and `dev-coverage.json` are in
`outputs/ptv3-shared-wam-v4/filter-transition-20261018-r1/`.

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

## Decision probe at contract-fixed step5000

The early DEV score improves mainly through C while trajectory distances do
not yet show a source-macro improvement. Distinguish a strong opposing C
gradient from trajectories on a general optimization plateau. The cheapest
check is three fixed source-balanced TRAIN batches (two windows/source,
seeds181/182/183), computing weighted hand/scene/C gradient norms and cosine
on the complete final shared transformer layer at step5000. Use eval mode,
the actual AMP/config/reader, one GPU,180s budget, no optimizer or checkpoint
write; do not alter the production run. A consistently dominant opposing C
direction would motivate a separate bounded loss-balance comparison; absence
of that pattern redirects diagnosis to per-source fit/conditioning rather
than changing C weight from scalar loss magnitudes alone. This small layer
probe cannot establish whole-trunk or full-DDP training dynamics.

The probe completed on real step5000 with dependency commit `2183eb46`,
tool commit `685bba0c`, remote Python3.8/Torch2.0.1+cu118. It took9.79s,
peak allocated6239.6MiB, inspecting7,087,872 final-layer parameters. Weighted
C/hand+scene gradient norms are.565/.682, .406/.573, .474/.444; their cosine
values are.00821, .02860, -.00201. This limited probe does not show a
consistently dominant opposing C gradient. Do not lower C weight solely from
its larger scalar loss; examine per-source fitting/conditioning next, while
preserving the current run. No optimizer or checkpoint was written.

At step5000 the fixed DEV objective is1.98219175 vs initial2.06847247
(-4.17%). Source-macro hand all24 error is35.1115->35.3817mm, object all24
12.4189->12.3721mm, positive-contact part accuracy60.308->64.369%.
Improvement is mainly interaction prediction; broad trajectory improvement
is still not established. The plot and compact snapshot are local
`outputs/ptv3-shared-wam-v4/filter-transition-20261018-r1/training-progress.png`
and `training-curve-snapshot.json`; gradient evidence is
`component-gradients-step5000.json`. The temporary GPU probe has exited;
production training continues on the unchanged recipe/objective.

## Decision probe: larger DEV panel outside selection sample IDs

The active selection panel contains32 sampled windows per represented source.
After the run passed12k, its source-macro score decreased mainly through C,
while trajectory metrics fluctuate. Check whether that pattern also appears
on a larger, disjoint panel before interpreting it as general learning.
Run `heldout-dev-panel-20261018-r1`, work version
`wam-v4-disjoint-dev-panel.v1`: select128 unique DEV indices per source,
excluding every existing selection-panel index. Use the existing validation
seed for deterministic candidate generation, fixed source order and batch8.
This is a new DEV audit panel, not an untouched TEST split.

Compare model-only predictions at retained step0 and a hard-linked snapshot
of the latest completed checkpoint at launch; no checkpoint selection,
optimizer step, normalization change or production-panel replacement. First
reproduce the original step0 selection score within1e-5. Cache at most256MiB
CPU inputs, use one remote GPU and a300s deadline; the snapshot is a hard link
within the same NAS, retaining its inode without allocating a second full
checkpoint. If the novel panel retains C improvement with flat trajectories,
keep the C-learning interpretation and focus next on per-source trajectory
conditioning. If it contradicts the original trend, diagnose panel sensitivity
before changing losses or extending scientific claims. This is cheaper than
another training run and runs separately from the local GT-tau experiment.

The audit completed in147.30s on remote `graspenv`/GPU0, pinned training
dependencies `2183eb46` and audit tool `756c11c2`. It retained step13000,
used1,024 unique windows (128/source,8 represented sources), CPU cache
176,241,664 bytes and PyTorch peak allocation469.3MiB. The original selection
baseline reproduced exactly:2.0684724698694046. The production panel and
selection aliases were not changed; the audit process exited.

| Disjoint-panel source-macro metric | Step0 | Step13000 |
| --- | ---: | ---: |
| Objective | 2.087452 | 1.873956 |
| Hand all24 error (mm) | 36.9025 | 37.0866 |
| Object all24 error (mm) | 11.1615 | 10.8577 |
| C contact MAE, all valid queries | .15530 | .13141 |
| C positive-contact part accuracy | 61.4235% | 69.1950% |

Objective decreases10.23% on the disjoint panel; all8 source objectives improve.
The arithmetic C-component decrease accounts for98.6% of the total objective
decrease, not a causal decomposition of representation learning. Hand error
is+.50%, object error-2.72%. GRAB/HOT3D hand error increases70.30->72.52 and
56.12->59.61mm, while TACO improves28.71->26.58mm and ARCTIC60.55->59.39mm.
Thus the interpretation is retained: broader C learning is present, aggregate
hand-trajectory improvement is not. Inspect per-source conditioning/coverage
next rather than restarting training or reducing C weight from loss magnitude.
This panel excludes selection window IDs, not necessarily their sequences or
nearby overlapping windows; it is still DEV, with no independent TEST claim.
Local report/panel/summary:
`outputs/ptv3-shared-wam-v4/heldout-dev-panel-20261018-r1/`.

## 2026-10-19 surface-contact readiness and encoder-grid inspection

The user asked whether CEDex-like surface contact requires more data, forbade
KNN inside the model, and requested consideration of PTv3 voxel size. This was
a read-only engineering inspection, not a new training/evaluation experiment.
The 768-width run is complete at80k, with selection still step30000. The later
1024-width run stopped at a checkpoint-write artifact budget guard; its
[experiment card](../experiments/probes/P-20261018-wam-v4-scale-1024.md)
records the published-checkpoint boundary.

The current remote human training payloads and main3 sequence caches primarily
retain11 hand points, not a full hand mesh. One TRAIN payload header per native
source confirms that UniDex source4 also retains `q`, `dq`, and `hand_root`,
while H2O source9 retains `native_mano_camera` and its availability mask.
TACO/HOT3D/DexYCB samples lack those richer parameters in the training payload;
their original annotations need to be read again. Original TACO hand archives,
HOT3D native files and DexYCB raw labels are present locally. GRAB and ARCTIC
metadata point to existing local read-only raw MANO files. The existing HOI4D
hand intermediate retains native MANO parameters, but the current resampled
training payload only has interpolated11-point positions. Surface labels must
respect its camera transforms, native timing and interpolation provenance.
This is not a full per-sequence surface-recoverability audit.

Thus surface-derived contact supervision requires a new derived data layer,
not necessarily new datasets or more episodes. Reconstruct available hands
from original pose/shape or robot FK, retain normals/part definitions, and
generate contact/part/validity per frame and fixed object-query identity.
Cache per-frame labels rather than repeating overlapping24-step windows or
materializing every full hand mesh. Missing geometry must not become negative
contact labels. Richer labels alone do not remove ambiguity in an11-point
model input; adding observed surface/part features is a separate input choice.

There are two independent1cm grids today: `config.voxel_m` controls PTv3
encoding, while `V4SamplingPolicy.voxel_m` controls scaffold label generation.
Changing the encoder config does not change the latter. The active geometry
and PTv3 forward paths use grid grouping, serialization, and inverse pooling
indices, with no KNN/cdist. Future surface labels can be prepared offline
using geometry/grid queries and loaded by training; neither a model KNN nor
per-iteration MANO/contact reconstruction is required.

A deterministic TRAIN panel of four windows/source (36 total, seed20261019)
was inspected with the frozen real reader, CPU geometry only, no neural
forward/backward. Remote environment: Python3.8.20, PyTorch2.0.1+cu118.
The final inspection took10.78s. At fixed512 object surface points:

| Encoder cell width | Mean occupied object cells | Mean object+hand cells | 1 - occupied object cells/512 |
| --- | ---: | ---: | ---: |
| 10mm | 306.44 | 324.58 | 40.15% |
| 5mm | 439.25 | 457.81 | 14.21% |
| 3mm | 484.28 | 502.92 | 5.41% |

These are window-sampled geometry statistics, not unique-object population
estimates, contact-label accuracy, GPU throughput, or model-quality results.
They support5mm as a candidate for a later bounded comparison, not a validated
optimum. Finer grids preserve more initial local features but also change
sparse-convolution physical support and all four stride2 pooling scales.
At current point counts, the initial occupied cells are bounded by512+22;
halving cell width does not imply a dense-grid8x compute increase. Shared
scene-token count remains256, and raw query xyz/normals remain continuous,
so1cm is not a hard quantization limit on C output positions. The128 contact
query coverage is another independent limit. Keep label precision and encoder
grid separate; a5mm encoder can consume finer offline geometric labels.

Local evidence and the exact read-only remote script:
`outputs/ptv3-shared-wam-v4/contact-data-voxel-audit-20261019-r1/`.
No production weights, readers, raw datasets, training config, or processes
were changed, and no restart or new training was launched.

Documentation checks: changed Markdown links, the updated1024 experiment-card
schema/seed pool and the generated index pass. The full `tools/verify.py
--changed` gate remains blocked by missing v2 metadata in six existing
`trajectory-policy` cards; those unrelated cards were not rewritten.
