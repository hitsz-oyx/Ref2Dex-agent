---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-joint-learning
experiment_id: P-20261011-pointwam-joint-learning
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: PENDING
claim_id: C1
hypothesis_family: HF-pointwam-joint-learning
probe_index_in_family: 1
seed_pool: probe
seeds: [247]
decision_changed_if_positive: retain the joint target contract and invest next in faithful pretrained encoders
decision_changed_if_negative: distinguish fitting failure from held-record generalization before expanding data or architecture
status: RUNNING
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

Result: pending.
Decision: pending fixed-budget training.

Debug smoke1 at576b9dc7 reached16 finite updates but failed while saving dev
predictions: original episode IDs contain directory separators. Preserve its
FAILED manifest/log; no scientific result follows. The filename is now escaped
into a single basename; rerun the same debug smoke before main training.

## Limitations / future evidence

Three fit and three dev TRAIN records; no subject/session isolation or official
benchmark. Full-clip approximate teachers, interpolated targets and palm
convention inherit qualification limits. Tiny from-scratch architecture/text
encoder cannot test pretrained PointWAM representations or task-conditioned
policy benefit. Future faithful encoders/data scaling require this Probe's
decision first; broader negative claims require validation, not this run.
