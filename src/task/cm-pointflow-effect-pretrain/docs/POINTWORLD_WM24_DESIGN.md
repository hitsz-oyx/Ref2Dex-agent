# PointWorld-small WM24 (user ref3)

On the existing `cm-pointflow-effect-pretrain` branch, implement the user-requested
replacement for the near-static WM30 model. Reuse local PointWorld commit
`05484826dfef74cbe278a3974179a5a16705d35d` PTv3 code and its YAML small blueprint;
do not redesign the backbone. The dependency is the user's clone at
`third_party/PointWorld`, with upstream licenses retained there. Checkpoint
manifests pin its commit and actual PTv3 source hashes.

Keep the completed 627-sequence corpus, 30Hz/K24, four history frames,
program anchor plus current 0.5m neighborhood, one current anchor coordinate
frame, same sequence splits and right/left semantic masks. Scene inputs contain
current geometry and history features. Uniform deterministic subsampling gives
every object encoder points within a total 4096-point scene budget, retaining
current valid hand points. Labels still use all 512 points of EVERY local object.

Project scene18 and action9 features to128. Add timestep, hand and keypoint
embeddings to future action points. Concatenate valid scene and up to528 future
hand points before PTv3, grid1cm/patch128, encoder2/2/2/6/2 and four decoder
stages2/2/2/2. Use the released spatial orders and drop-path0.3. Sparse backbone
runs FP32 (FlashAttention internally BF16), projections/head can use BF16.
After inverse unpooling, scene skip and global max-pooled hand summary use the
same affine FiLM additions as PointWorld. Pool scene features by object, add
current object geometry and24 horizon embeddings, and use a small MLP producing
translation and continuous6D rigid rotations. No action Transformer or dynamics
Transformer. H excludes future hands from geometry AND summary.

Freeze normalization from4096 balanced TRAIN windows, seed216, GPU, before any
training. Per-horizon per-axis means/stds for translation and analytical point
flow; std floors1mm. Rotation uses train RMS angle with floor0.02rad. Scene/action
features use train mean/std (floor0.01). Save indices hash, source manifest hash,
stats in every checkpoint. Physical translation is decoded from normalized
head output;6D rotation is an identity-centered residual with ordinary0.01
head initialization. Loss is sum of normalized point-flow Huber, translation
Huber and scaled rotation-angle Huber (delta1). Motion weight is the released
PointWorld sigmoid(5*(per-frame point displacement-0.005)/0.005), gamma1,
normalized over valid supervision. It only affects loss. No uncertainty head.

Three independent arms H/H+A/H+shuffle(A) have identical initial weights,
sample schedules and effective batch16. Same-hand donor chunks/masks are
shuffled intact, including singleton fallback from the corresponding split.
H+A also receives evaluation shuffle. Fixed balanced/natural validation and
test panels and physical metrics retain the previous protocol. Equal-update
final checkpoints are primary; test data cannot choose checkpoint/statistics.

Implementation: add Task-local model/normalization/trainer/evaluator/launcher,
preserving the old code and raw run. Verify point budget, mask/no-label-leak,
valid SE(3), hand gradients, exact checkpoint restore, real-data updates and
bounded learning smoke. Then start three GPU arms under one24h deadline,
40000 updates maximum; save bounded best/latest/final checkpoints. Stop on
nonfinite, source drift, insufficient disk or resource conflicts. Preserve old
run checkpoints; if required for GPUs, request its verified own workers to save
and stop at an update boundary. No new branch, PPO or global claim change.

This is an observational prediction Probe, not causal action utility. Success
only motivates further pretraining/adaptation. Formal architecture comparison
and multi-seed validation remain deferred; simultaneous objective changes mean
differences from old WM30 cannot be attributed solely to PTv3.
