---
schema: ref2dex.probe.v2
probe_id: P-20261008-hocap-frame-generalization
experiment_id: P-20261008-hocap-frame-generalization
date: 2026-10-08
task: cm-pointflow-effect-pretrain
branch: main
git_commit: 8f857ae6585354380882709587a1c599c9f9572c
claim_id: C3
hypothesis_family: HF-hocap-frame-generalization
probe_index_in_family: 1
seed_pool: probe
seeds: [228]
decision_changed_if_positive: resolve the original clock and freeze a broader external test evaluation
decision_changed_if_negative: inspect distribution and adapter correctness before claiming external dynamics generalization
status: UNCLEAR
run_id: hocap-frame-eval-20261008-r1
---

# Can the fixed main-three model predict HOCap beyond a static baseline?

Result: Fixed mixed endpoint improves moving h24 EPE22.421→17.805mm versus Oak-only (20.59%), but near-static worsens8.039→22.160mm. Same192windows; clock unverified.
Decision: Preserve moving-window endpoint benefit and near-static regression separately; no causal data-mixture claim due to extra training/loss changes. Keep HOCap frozen and resolve clock before formal testing.

## Motivation and frozen comparison

User asks to attempt evaluation of HOCap and inspect prediction quality.
This tests the physical-pretraining subgoal, not robot/Cm policy utility.
Compare the already chosen main-three `latest.pt` step50000 with identity
SE(3) persistence on identical samples and all512corresponding surface points.
The checkpoint, embedded training-only normalization and model implementation
stay fixed. No optimizer, fitting, adaptation or best-checkpoint sweep.

Raw HOCap9subjects/64sequences/72944frames stay outside training. Use original
array indices directly: H4 and K24 mean four history and24future frames.
Original FPS/timestamps are unverified. Input velocity/acceleration retain
the model's nominal30/900scaling, explicitly an assumption. No invented
measured timestamps, resampling or claimed0.8second test. This new schema is
not a training-eligible WM30 native manifest.

Decode with the HOCap author's declared MANO dependency, commit
`f66a5020d40afbf0b3b5b7ebedd3154aa859cf30`, PCA45/nonflat mean, real subject
beta, meter translation. Fixed right/left and11semantic indices
`[0,1,5,9,13,17,4,8,12,16,20]`; absent hands are masked. Before inference,
validate all21decoded joints against32released camera labels transformed
to world, maximum tolerance10micrometres. Finger-tip identities matter:
generic manopth `4f1dcad` uses index-tip vertex317 whereas the declared fork
uses319. The generic fork differed5.56mm at that point only and is excluded;
the author fork reproduces all21labels with max8.20e-8m. Broken external
TACO import and generic-fork mismatch are preserved as engineering failures,
not prediction errors. No installed/external package is modified.

Canonical512points use the existing native-source deterministic surface
sampler and normals, with its geometry-center/radius features. Current-only
near-hand anchor requires <=5cm surface distance; local objects use0.5m
geometry centers. Require28frame hand-presence stability, all selected
object poses valid, and the existing per-frame jump exclusions (.15m/.6rad).
Candidate anchors every8array frames. Moving0 requires >2mm translation or
>.02rad rotation at any of the24future frames; static-near1 otherwise.
No fabricated program/background labels. Conditioning uses released future
human hand positions, as in current observational pretraining, not executable
robot actions.

Freeze up to2moving windows and1remaining natural eligible window per sequence
without replacement, seed228, max192windows; report missing coverage rather
than filling it from another sequence. The natural panel is sequence-balanced
within current near-hand eligibility, not a uniform census of all raw frames.
Panels are saved before prediction. Report model/static point EPE at horizons
1/4/8/12/24, moving/natural separately; moving anchor h24 is primary.
Frame mismatch prevents a formal scientific conclusion; Probe status stays
UNCLEAR while the clock is unverified, irrespective of apparent model gain.

## Bounds, identity and next decision

One empty GPU0, <=1800seconds preparation+evaluation, <=1GiB new output,
microbatch2BF16, DataLoader workers0, reserve20GiB; shared4GPU/300GB limits
remain. Stop on deadline/owned signal, foreign GPU use, frozen input drift,
nonfinite results or output/disk limit. Hash raw arrays/metadata/meshes/labels,
MANO models/implementation, checkpoint, model/vendor source and frozen panel;
recheck full hashes at completion. Actual runtime git commit is in manifest.

Output: `outputs/cm-pointflow-effect-pretrain/hocap-frame-eval-20261008-r1/`.
Engineering checks: `outputs/cm-pointflow-effect-pretrain/hocap-eval-engineering-20261008-r1/`.

If a sizable moving-window gain is visible, prioritize resolving the capture
clock and freezing a proper external evaluation. If the model underperforms
persistence, verify this adapter and distribution first; do not tune on HOCap,
silently change the checkpoint, or call all external generalization refuted.
Cross-domain mesh/subject overlap remains unaudited; inherited pretraining
included ContactPose. Multi-seed/policy utility are future evidence.

## Actual execution and results

### Frozen Oak-only endpoint comparison (registered before execution)

User requests checking whether mixed pretraining helps relative to the earlier
OakInk2-only version. Reuse this experiment, seed228, the exact existing
processed tensors/canonical surfaces and128moving/64natural panel indices;
do not regenerate or select windows. Run both fixed endpoints on the same
empty GPU1, microbatch2/BF16/TF32 and identical per-panel random seeds.
Oak-only endpoint is `pointworld-action-ddp-20261007/train-action/latest.pt`,
step10000, trained at06933b3. Mixed endpoint remains the originally fixed
`pointworld-main3-20261007/train-action/latest.pt`, step50000. Both use the
same temporal architecture, model/data/evaluation/vendor sources and embedded
normalization SHA6190e0e9. Oak's DDP training-entry hash differs from current
code; its exact recorded hash is verified against Git06933b3 and this
training-only file is not used for inference. Other source drift fails closed.

Decision Probe: does the current mixed endpoint improve moving-window external
prediction, and does it reduce or increase near-static false motion? If the
current endpoint wins only moving windows, preserve that partial benefit and
keep false-motion calibration as a limitation. A regression motivates checking
the training recipe and distribution before further scaling. The cheapest
discriminating experiment is192existing windows with no fitting or new data.

This is a historical endpoint comparison, not a matched data-mixture causal
ablation: mixed training starts from this Oak-only endpoint, adds14250four-source
updates and50000main-three updates, changes global batch192→128 and later
uses physical loss normalization. Oak endpoint itself inherited the earlier
Oak temporal checkpoint step12163. Preserve each counter's stage meaning.
Report all frozen1/4/8/12/24horizons, natural moving/near-static strata and
persistence. Replayed mixed metrics must match the prior run within1e-6
metres for distances and1e-6radians for angular metrics; static controls must match between checkpoints within
1e-12. Same unknown source clock and observed future hand limitations apply.

Run ID: `hocap-oak-vs-mixed-20261008-r1`. Entry:
`tools/run/compare_hocap_checkpoints.py`. Fresh output:
`outputs/cm-pointflow-effect-pretrain/hocap-oak-vs-mixed-20261008-r1/`.
Bounds: one empty GPU1,600seconds,100MiB new output,20GiB free reserve;
stop on source/panel/checkpoint drift, foreign GPU use or deadline. Verify
the prior run's1318input hashes and new checkpoint/script before and after
inference; no optimizer, parameter/normalization update or checkpoint selection.

Engineering r1 at6f7f41f completed both endpoints' inference in36.9seconds but
failed the initial uniform1e-7replay threshold: rotation maximum differs
2.76e-7rad (~0.000016degrees), center/translation distances differ <=6.24e-8m
and surface metrics <=6.21e-8m; primary anchor h24 point metrics match exactly.
Static controls between endpoints match exactly. This is a replay acceptance
bug that compared metres/radians with one tolerance, not a prediction failure.
Keep r1 FAILED and its raw metrics unchanged. Separate angular1e-6rad and
distance1e-7m tolerances before retry; do not relax the point comparison.
Add regression for small angular differences versus unacceptable point drift
and missing coverage. Bounded retry r2 reuses both exact endpoints/panels and
the original r1deadline, without renewing600seconds or selecting checkpoints.

Engineering r2 at34d2e6c again completes inference,36.7seconds; primary h24
point results reproduce, but sub-micrometre CUDA variation in center_error
(maximum1.46e-7m) trips the tightened distance gate. All surface deviations
<=9.29e-8m, angular <=6.05e-7rad and static controls remain exact. Preserve
r2 FAILED. A1micrometre/1microradian replay tolerance is appropriate for
this millimetre-scale endpoint comparison; freeze it for r3, retaining the
original deadline. This numerical gate correction does not change datasets,
weights, reported metrics or the moving-versus-static interpretation.

### Completed Oak-only versus mixed comparison

Run `hocap-oak-vs-mixed-20261008-r3` at code
`a3139fb864e9c0b9e876b38ef6141457409775ab`, GPU1/PID2822122, completes
in40.990seconds within the original r1deadline. Both checkpoints strictly
load; same inference configuration and training-only normalization. All1323
input/source hashes pass final verification; static controls match exactly.
Mixed replay maximum distance difference9.70e-8m and angular2.09e-7rad,
well within the1micrometre/1microradian check. Primary moving and natural
h24 point means match the first HOCap evaluation exactly. Five schema/replay
tests and changed-file verification pass. Process exits/GPU1 is released.

Moving128windows, same64sequences/9subjects, anchor512surface EPE in mm:

| Future frame | Oak-only latest10000 | Mixed latest50000 | Error reduction |
| --- | ---: | ---: | ---: |
| 1 | 1.161 | 1.052 | 9.44% |
| 4 | 4.025 | 3.134 | 22.14% |
| 8 | 6.922 | 5.696 | 17.72% |
| 12 | 10.500 | 8.484 | 19.20% |
| 24 | 22.421 | 17.805 | 20.59% |

Natural64windows, frozen near-hand sequence-balanced sample:

| Stratum, h24 | Windows | Oak-only mm | Mixed mm | Persistence mm |
| --- | ---: | ---: | ---: | ---: |
| All | 64 | 24.675 | 20.346 | 54.985 |
| Moving | 47 | 30.692 | 19.690 | 74.744 |
| Near-static | 17 | 8.039 | 22.160 | 0.356 |

Overall natural error falls17.55%; its moving subset falls35.85%, but the
near-static error rises175.65% (2.756times). Near-static is worse at every
reported horizon, not just the endpoint. Both models underperform persistence
there. Moving endpoint center error17.488→15.462mm, rotation16.084→10.672deg;
the point improvement combines translation and rotation gains.

Interpretation: the current mixed-trained endpoint is PROMISING for moving
external hand-object prediction, with a clear near-static false-motion
regression. Overall experiment stays UNCLEAR because the original clock is
unverified. The historical contrast cannot separate adding GRAB/ARCTIC/earlier
ContactPose from64250additional updates, changed effective batch and physical
loss normalization. It provides no robot-policy/Cm utility claim. No fitting,
hyperparameter/threshold selection for model inference or test-based checkpoint
choice took place. The numerical replay gate was corrected transparently;
all failed runs and raw metrics remain available.

Observed GPU1 sampled driver memory peak713MiB and utilization15–26% during
inference, returning to2MiB/0% after exit. Microbatch2, workers0 and guarded
CPU loading make this small fixed-panel inference low-utilization; this is not
a training-throughput benchmark. CUDA allocator peak378.80MiB/reserved390MiB.
Full metrics, identities and GPU samples:
`outputs/cm-pointflow-effect-pretrain/hocap-oak-vs-mixed-20261008-r3/`.

Run `hocap-frame-eval-20261008-r1` used code `8f857ae`, GPU0, PID2769220,
seed228 and the original latest50000 checkpoint trained at `9019fd4`.
COMPLETED in49.425seconds, process exited normally and GPU0 is empty.
No weights, normalization or checkpoint choice changed. All1318frozen
input/source identities passed final hashes. Output43MiB; CUDA allocator
peak244.85MiB/reserved274MiB, excluding driver/context memory. This is
inference, not training; preparation includes CPU geometry work.

All64sequences have candidates:6275moving/2626near-static at fixed8frame
anchor cadence. Panels have128moving windows (2per sequence) and64remaining
natural eligible windows (1per sequence),9subjects each, no duplicate window
between panels. Natural has47moving/17near-static. Runtime MANO/camera check
passes32frames with mean6.54e-6m/max9.83e-6m under configured CUDA numerics;
earlier engineering under default numerics has max8.20e-8m. Both satisfy the
predeclared1e-5m gate; these are internal consistency errors, not annotation
accuracy. Schema/coverage regression tests pass3/3.

Moving-anchor surface EPE, millimetres:

| Future frame | Fixed model | Static persistence |
| --- | ---: | ---: |
| 1 | 1.052 | 3.470 |
| 4 | 3.134 | 13.151 |
| 8 | 5.696 | 25.289 |
| 12 | 8.484 | 36.155 |
| 24 | 17.805 | 63.704 |

Moving endpoint reduction72.05%; center error15.462mm and rotation
error10.672degrees. Natural endpoint20.346mm vs54.985mm (63.00%lower),
but pooling conceals a failure:

| Natural category | Windows | Model h24 EPE mm | Static h24 EPE mm |
| --- | ---: | ---: | ---: |
| moving | 47 | 19.690 | 74.744 |
| near-static | 17 | 22.160 | 0.356 |

The model predicts substantial motion for objects whose released trajectory
is nearly stationary. Do not describe it as uniformly accurate. This is an
external frame-index observation, not proof of correct source clock, unseen
objects, formally established generalization or robot utility. No follow-up
tuning, extra checkpoints or repeated test selection occurred.

Evidence: run `result.json`, `moving_metrics.json`, `natural_metrics.json`,
`moving_panel.npy`, `natural_panel.npy`, `processed/index_test.npy` and
`input_manifest.json`; launch/stdout/stderr in
`tmp/hocap-frame-eval-20261008-r1.log`. Physical metrics are in JSON. Input
derivatives scale nominally by30/900; horizons remain original array frame
indices with `source_fps_verified=false` and `test_ready=false`.
