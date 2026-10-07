---
schema: ref2dex.probe.v2
probe_id: P-20261007-pointworld-small-wm24
experiment_id: P-20261007-pointworld-small-wm24
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: ce6fcb4
claim_id: C3
hypothesis_family: HF-pointworld-unified-action-effect
probe_index_in_family: 1
seed_pool: probe
seeds: [210, 212, 213, 214, 215, 216, 217]
decision_changed_if_positive: continue unified spatial effect pretraining and design robot adaptation
decision_changed_if_negative: inspect learning and action use before additional training
status: UNCLEAR
run_id: pointworld-small-wm24-20261007
---

# Unified spatial hand/action-to-rigid-effect Probe

Decision experiment serving Mission short-term physical prediction: distinguish
whether the ref3 unified PointWorld architecture and normalized motion losses
learn useful observed-hand information rather than static predictions. Cheapest
next gate is train-only stats plus real-data engineering/learning smoke, followed
by bounded three-arm training on the already prepared corpus.

## Decision Note

Old WM30 outputs near-zero despite moving labels and action gradients; this
does not refute Cm. User explicitly requests ref3 replacement. Choose the
specified PTv3-small backbone, delete separate action/dynamics Transformers,
retain exact rigid GT and implement released normalization/soft weighting.
Existing branch; corpus reused without downloads; at most4 GPUs overall,
24h group wall budget/40000updates, new outputs only. Preserve old runs and,
when needed, save/stop identified old workers. Stop on nonfinite, code/input
drift, disk conflict, deadline. Positive screen leads to adaptation design;
negative/uncertain signal leads to attribution rather than extra seeds/epochs.
No external authorization boundary is crossed and Mission claim stays unchanged.

## Frozen design and implementation

[User ref3](../../user/ref/ref3.md) and
[concrete design/implementation sequence](../../POINTWORLD_WM24_DESIGN.md).
Run output: `outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/`.
Input: completed `outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/`.
Actual runtime commit and source/stat/config hashes are authoritative in manifests;
frontmatter commit is the pre-implementation checkpoint until reconciled.

## Evaluation / decision

Same H/H+A/H+shuffle(A), initialization, draw schedule,40000updates and effective
batch16; balanced/natural val256 and held-out test256, sequence isolation. B also
evaluated with shuffled A. Report physical pointEPE/translation/rotation/center
error at h1/4/8/12/24 for anchor,moving objects,scene and each stratum, with static
baseline. PROMISING requires moving-anchor h24 test EPE>=10% lower than BOTH
controls, beats static, and inference shuffle degrades>=5%. Otherwise valid
equal-update failure is UNPROMISING, execution/information insufficiency UNCLEAR.
Engineering smoke is only interface/learnability evidence, not the test screen.

## Current status

Implementation underway; no new predictive or policy conclusion.

## Limitations / future evidence

Observed future hands are post-treatment geometry, not counterfactual commands.
One-seed observational accuracy cannot establish causal robot/PPO utility or
global C3. Later multi-seed matched validation, robot adaptation and trained
Cm-on/off comparison remain necessary. PTv3 and loss change together; old/new
differences cannot separately identify architectural contribution. Sample-based
train normalization is approximate; held-out statistics are never fitted.
