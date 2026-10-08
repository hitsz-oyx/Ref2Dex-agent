---
schema: ref2dex.probe.v2
probe_id: P-20261008-hocap-frame-generalization
experiment_id: P-20261008-hocap-frame-generalization
date: 2026-10-08
task: cm-pointflow-effect-pretrain
branch: main
git_commit: 1239e52
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

Result: Acquisition/geometry checks pass; model evaluation is pending under an explicitly unverified source clock.
Decision: Run one bounded no-fit frame-index Probe; preserve HOCap as test-only and do not select hyperparameters or checkpoints from its errors.

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
