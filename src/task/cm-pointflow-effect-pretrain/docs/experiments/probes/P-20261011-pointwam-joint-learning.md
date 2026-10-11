---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-joint-learning
experiment_id: P-20261011-pointwam-joint-learning
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: c9727b88265c1f80b94d52503e79bc58ebbcc28b
claim_id: C1
hypothesis_family: HF-pointwam-joint-learning
probe_index_in_family: 1
seed_pool: probe
seeds: [247]
decision_changed_if_positive: retain the joint target contract and invest next in faithful pretrained encoders
decision_changed_if_negative: distinguish fitting failure from held-record generalization before expanding data or architecture
status: UNCLEAR
run_id: pointwam-joint-learning-20261011-r1
---

# Bounded scene/hand joint learning

Class: Decision. User instructed us to continue after all12 official-TRAIN
clips qualified. This tests the Task's human-video representation subgoal:
can the joint masked teacher targets be learned from current scene/hand/text,
and does a small learner predict held-record hands better than cheap controls?
The cheapest discriminating run reuses six fit/six dev clips, without further
acquisition or native model/policy training. No global claim changes.

## Frozen protocol

- Input cohort: qualified12clip manifest from
  `pointwam-train-qualification-20261011-r1`. Hash-verify every pack and reject
  TEST sources. An admission ledger authorizes only the six fit clips for
  this Probe; original `training_allowed=false` flags remain unchanged.
  Six dev clips are evaluation-only; their original records are disjoint
  from fit records. Overlapping clips are averaged within each original
  record, then the three records receive equal weights.
- Inputs: current scene XYZ/RGB,20 semantic hand points and validity, language.
  Center on the valid current hand mean; coordinate scale.5m. Randomly sample
  1024 initially selected scene identities per training example without
  consulting future masks/motion. Evaluate all scene points. Missing hand
  inputs have zero numerical padding with false masks; labels retain NaN
  and are masked. No future coordinates, teacher tokens, velocity or labels
  enter the model. Teacher point reconstruction itself uses full clips.
- Explicit architecture substitute:128wide set MLP, masked scene/hand/text
  pooling, unique20hand-ID embeddings, learned fit-only word table, two
 30step displacement heads. Mosaic3D, Recap-CLIP and Volt are not reproduced
  here; this is a contract/learnability Probe, not official PointWAM training.
- Only fit labels determine coordinate-wise means/std separately for each
  scene/hand horizon. Scene supervision clips coordinates to[-.5,.5]m;
  std floor1mm is a declared numerical safeguard. Decode predictions to meters
  and evaluate against original unclipped teacher displacements.
- Loss follows [paper Appendix B.5](https://arxiv.org/html/2610.02840v1#A2.SS5):
  scene normalized-coordinate MSE with motion weights
  `sigmoid(5*(adjacent_motion/.005-1))`, normalized over valid weighted
  point-steps; hand MSE averages valid point-steps per example, then examples;
  equal scene/hand coefficients. Adjacent missing scene observations have
  zero motion weight; recovered valid hand targets remain supervised.
- AdamW lr1e-4, decay.01, batch2, gradient norm cap1, seed247, exactly600steps.
  Save only the final checkpoint. No dev-selected epochs/checkpoints or tuning.
  A preceding16step/256point debug seed7 smoke only checks engineering.
- Controls: zero displacement (static) and fit-only per-horizon mean
  displacement. Zero-initialized final heads start at that same fit mean.
  Metrics: masked 3D EPE mm across all horizons, endpoint EPE, and scene
  identities moving>=2cm at the valid final horizon. Motion classification
  is evaluation-only. Macro average is per original record, not per point.
- Fit gate: final full-scene fit loss falls>=50% separately for scene/hand
  versus fixed initial predictions. Dev gate: hand EPE<=95% of the better
  static/fit-mean control; moving-scene EPE<=105% of static; at least2/3 dev
  records improve hand EPE versus static. Both gates => PROMISING. Fit gate
  passes but dev gate fails => UNPROMISING for this recipe. Fit gate failure
  => UNCLEAR and targeted implementation/optimization diagnosis. No gate
  establishes paper/model/policy performance.

## Resources and stop conditions

Reuse existing read-only environment and qualified packs; one free GPU7,
main run<=15min and output<=1GiB; debug smoke<=2min. Shared output use about
162GB remains within300GB. Stop on identity/split drift, non-finite values,
absent gradient to either head, checkpoint replay mismatch, resource conflict,
output collision, or time/disk cap. Preserve evidence of failed runs.
Work on the requested worktree/branch; preserve user drafts and native models.

Before launch, pin code with a local Git commit. Store Git identity, commands,
PID/device, normalization/vocabulary, admission, progress, GPU samples and
final metrics under `outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/`.

Result: UNCLEAR. The fixed main run completes and hand fitting improves, but
scene fitting and the held-record benefit screen fail.
Decision: retain the qualified data and verified joint learning contract;
do not expand data or infer PointWAM/model/policy failure from this substitute.

Debug smoke1 at576b9dc7 reached16 finite updates but failed while saving dev
predictions: original episode IDs contain directory separators. Preserve its
FAILED manifest/log; no scientific result follows. The filename is now escaped
into a single basename; rerun the same debug smoke before main training.

Smoke2 at16912856 completed in5.20s with59.98MiB peak GPU allocation;
both heads receive finite gradients and checkpoint replay is bitwise equal.
It is ENGINEERING_SMOKE regardless of its fit/dev screen numbers.

## Targeted diagnostic decision

The frozen main run atc9727b88 completed600steps: fit scene loss6.02475→4.62520
and hand1.00000→.41528. Only hand passes its fit threshold, so the original
result is UNCLEAR; dev also fails and is not used for tuning. Before attributing
this to data/model failure, perform the cheapest optimization diagnosis: from
scratch on the first fit clip, fixed512initial-only point IDs, debug seed7,
same model/contract/loss,600steps atlr1e-3,<=2min/one GPU/<10MiB artifacts.
Record initial/final loss and finite gradients. This removes cross-clip
interference and changes the optimizer scale purely to check fitting capacity.
It neither replaces the main run nor changes its gates; no dev is evaluated.
Stop on non-finite values/input drift/budget. Strong single-clip fitting would
shift diagnosis toward multi-clip optimization/capacity; persistent failure
requires closer scene-target/weight/model diagnosis. No method is closed here.

## Main observations

Training completes600updates at the pinned main commit above with195764
parameters. All12pack hashes remain unchanged; both heads receive gradients,
predictions/gradients are finite, and checkpoint replay is bitwise equal.
The initial model is exactly the fit-mean control in the reported EPE metrics.
Input-isolation, fit-only statistics/vocabulary, masked hand gradients, scene
permutation and TEST exclusion tests pass:14total data/learning contract tests.

| Full-scene fit loss | Initial | Final | Reduction | >=50% gate |
| --- | ---: | ---: | ---: | --- |
| Motion-weighted scene | 6.02475 | 4.62520 | 23.23% | fail |
| Hand | 1.00000 | .41528 | 58.47% | pass |

Loss uses normalized coordinate MSE; the following metrics decode to original
teacher-scale meters and report EPE in mm, macro-averaged over source records.

| Partition / predictor | All scene | Moving scene | Hand | Hand endpoint |
| --- | ---: | ---: | ---: | ---: |
| fit / static | 10.969 | 34.423 | 78.911 | 134.529 |
| fit / fit mean | 10.863 | 34.166 | 84.763 | 139.622 |
| fit / joint model | 17.948 | 33.908 | 49.040 | 50.379 |
| dev / static | 15.856 | 32.652 | 94.112 | 197.803 |
| dev / fit mean | 16.059 | 32.692 | 94.715 | 193.288 |
| dev / joint model | 36.701 | 55.228 | 96.243 | 177.382 |

| Dev record | Hand static / model | Moving scene static / model |
| --- | ---: | ---: |
| basic_pick_place/1 | 136.900 /117.655 | 27.312 /32.348 |
| basic_fold/1 | 96.238 /115.194 | 42.919 /103.958 |
| fold_unfold_paper_basic/1 | 49.199 /55.880 | 27.723 /29.379 |

Only1/3dev records improves hand EPE over static. Hand macro does not beat
either control, and moving-scene macro exceeds the static ceiling. Dev endpoint
hand EPE improves, but endpoint alone was not the predeclared gate. Do not
replace the failed screen with that favorable metric. The result remains
UNCLEAR because the joint fit threshold was not reached, rather than a negative
conclusion about PointWAM.

Target audit finds only24/1001812fit and931/1155747dev valid scene targets
require coordinate clipping (.00240%/.08055%); this does not explain the broad
fit failure. Motion-weighted loss emphasizes moving point-steps, so its
improvement need not lower unweighted all-scene EPE. This is the intended loss,
not evidence that the full scene objective is solved.

## Diagnostic outcome and next decision

The fixed first-fit-clip diagnostic at8c65728c completed in7.00s. Same loss and
architecture,512fixed initial-only scene IDs,600steps, debug seed7, lr1e-3:
scene7.24403→.65109 (91.01% decrease), hand.499817→.000009402 (99.998% decrease).
No dev evaluation/checkpoint selection occurred. This supports working
supervision/optimizer wiring and fitting capacity on one clip. Changing both
clip population and learning rate prevents attribution to either factor alone;
it is not a matched causal diagnosis or evidence of generalization.

Root inspected the fixed dev paper-record index-tip plot. Predictions are
finite and have the expected horizon/semantic slots, but right-hand X and
several other directions visibly disagree with teacher trajectories. The plot
is consistent with the failed aggregate screen; it is not independent GT.

The small-data pipeline now reaches acquisition→teachers→packs→joint
training→controls→checkpoint replay→trajectory inspection. Retain this contract.
Before scaling, distinguish fit-only optimization limits from architecture
limits without selecting on dev; faithful pretrained scene/language encoders
remain an explicit missing reproduction component. No native model/policy
training or large acquisition is justified by this run.

## Runtime and artifacts

Main elapsed15.67s, training11.27s, about53.2updates/s; GPU7 process memory
about395MiB, PyTorch peak60.77MiB. Sampled utilization5–7% reflects the tiny
model and host batch/gradient checks; the run completes well below its15min
cap, so optimizing throughput would not change this decision. No additional
GPU or external process was modified. GPU7 returns to2MiB idle. Training
outputs18.35MB before the small plot/audit additions; smoke2 outputs16.55MB,
single-clip diagnostic uses21.79MiB peak model memory and negligible output.

- [Main identity, command, PID and gates](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/manifest.json)
- [Admission ledger](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/admission.json) and [fit-only normalization/vocabulary](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/contract.json)
- [Initial controls](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/initial_metrics.json) and [final clip/record/macro metrics](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/final_metrics.json)
- [Final checkpoint](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/final.pt), [GPU samples](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/gpu_samples.json), [target audit](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/target_audit.json)
- [Fixed dev trajectory plot](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-r1/dev-index-trajectories.png)
- [Failed smoke1](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-smoke1/manifest.json), [successful smoke2](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-smoke2/manifest.json), [fit-only diagnosis](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-20261011-fitdiag1/manifest.json)

Entries: `tools/run/train_pointwam_joint_probe.py`,
`tools/audit/diagnose_pointwam_single_clip.py` and
`tools/audit/plot_pointwam_joint_prediction.py` under this Task. Commands are
captured in run manifests; use fresh output paths and the existing pinned
environment, with CUDA_VISIBLE_DEVICES=7 and project-local TMPDIR/cache.

## Limitations / future evidence

Three fit and three dev TRAIN records; no subject/session isolation or official
benchmark. Full-clip approximate teachers, interpolated targets and palm
convention inherit qualification limits. Tiny from-scratch architecture/text
encoder cannot test pretrained PointWAM representations or task-conditioned
policy benefit. Future faithful encoders/data scaling require this Probe's
decision first; broader negative claims require validation, not this run.
