---
schema: ref2dex.probe.v2
probe_id: P-20261007-pointworld-temporal-wm24
experiment_id: P-20261007-pointworld-temporal-wm24
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 26c669c
claim_id: C3
hypothesis_family: HF-pointworld-unified-action-effect
probe_index_in_family: 3
seed_pool: probe
seeds: [210, 212, 213, 214, 215, 216, 217, 218]
decision_changed_if_positive: retain temporal action identity and continue observed-action effect pretraining
decision_changed_if_negative: diagnose representation and optimization before further training
status: UNCLEAR
run_id: pointworld-temporal-wm24-20261007
---

# Time-preserving PointWorld action/effect Probe

Decision: evaluate observed-action predictive benefit after the user-requested ref4 correction.

Result: corrected three-arm training is running with verified common initialization and finite updates.

## Decision Note and motivation

The user's latest request explicitly replaces the running V1 three-arm training
with the corrected architecture and then requests DDP implementation. The
[ref4 diagnosis](P-20261007-pointworld-ref4-input-loss-audit.md) found substantial
action timestamp merging; the cumulative-weight recipe is a declared adaptation,
not a proven explanation of error. This serves the Task's short-term physical
prediction subgoal and preserves global C3/policy-utility boundaries.

Choose the already GPU-checked temporal adapter, per-horizon action summary,
and cumulative surface-displacement/rotation weighting with static floor.
The cheapest relevant decision is the existing matched three-arm Probe on the
same prepared corpus, frozen seed/draw/stats, without new acquisition or seeds.
Positive results motivate further adaptation; negative/unclear results require
attribution rather than interpreting future hands as useless.

Resource boundary: GPUs0/1/2, effective batch16, maximum40000updates each.
Retain the original authorized absolute deadline1791424717.7631629 rather than
extend its24h wall budget; remaining time at launch is about22h. Stop on user
request, nonfinite values, source/input drift, worker failure, disk conflict or
deadline. At most4 task GPUs overall and outputs<=300GB. GPU3 belongs to parallel
work and is not selected. No external authorization boundary is crossed.

Old V1 launcher and all verified workers received save-and-stop on explicit
user request. All exited successfully and preserved final checkpoints at
H4730/H+A4541/shuffle4751. These unequal partial checkpoints cannot give a matched
final comparison; old results remain UNCLEAR, not negative method evidence.
The stop record is `pointworld-small-wm24-20261007/user-stop-for-temporal.json`.

## Frozen execution contract

Model and recipe: [temporal correction](../../POINTWORLD_TEMPORAL_INTERFACE.md).
Input: `outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/`,
627 completed sequences, the original sequence-isolated split and window indices.
Normalization: original train-only4096-window seed216 statistics, frozen
SHA2566190e0e9c51089d2e19a51056b54dd94d6c9265577aad39d16cd857c0b681085.
No held-out statistics fitting. Identical seed217 initialization in H/H+A/shuffle,
global draw seed218, microbatch2/accumulation8,50,495,881parameters. Motion losses
and schedule are the same across arms. Shuffle uses complete same-hand donor
chunks, with H+A receiving an additional inference-shuffle intervention.

Run: `outputs/cm-pointflow-effect-pretrain/pointworld-temporal-wm24-20261007/`.
Launcher: `tools/run/launch_pointworld_temporal_group.py`, explicit GPUs0,1,2.
Actual runtime commit and source/stat/data/config hashes are authoritative in
`group_status.json` and per-arm `input_manifest.json`. The card commit above is
the verified model/entry implementation; launch evidence will record its exact
new launcher commit. Running model, trainer, evaluator, reused source, config,
stats and upstream files are immutable. DDP uses new isolated files after launch
and does not change or migrate these live workers.

## Evaluation and decision

Equal40000-update final checkpoints are primary, never separately chosen best.
Balanced/natural validation256 every1000updates; balanced/natural sequence-held-out
test256 only after all arms complete matched updates and within deadline.
Report physical point EPE/translation/rotation at h1/4/8/12/24 over anchor, scene,
moving objects and sampling strata, alongside static prediction. PROMISING
requires moving-anchor h24 test EPE>=10%lower than BOTH H and shuffled-action
controls, beats static, and inference shuffle of H+A degrades>=5%. A valid
matched failure is UNPROMISING; incomplete/insufficient evidence is UNCLEAR.
Training loss and engineering smoke do not establish quality or global utility.

## Engineering evidence and limitations

Before launch:13GPU semantic checks passed/1historical optional test skipped;
6CPU regressions passed;3update corrected-entry smoke, exact model and optimizer
checkpoint load, independent val-only evaluation within explicit numerical
tolerance. Parameter initialization matches V1. Artifacts are in
`pointworld-temporal-smoke-20261007/verification-r2.json`.

Temporal identity, per-horizon summary and cumulative weighting change together;
this Probe cannot attribute improvement to one component. The released-selector
option permits a later decision-relevant ablation, which is deferred here.
Single-seed observational predictive benefit is not causal robot command evidence
or trained-policy Cm-on/off utility. Future matched multi-seed validation and
robot adaptation remain necessary. Interrupted V1 is not a full-run control.

Startup at runtime26c669c is verified in `startup_verified.json`: launcher4156044,
workers4156111/4156112/4156113 on GPUs0/1/2. Data/config/stats/initial parameter
hashes match and all arms perform finite updates; original monitored source
hashes remain frozen. Isolated DDP implementation starts only after this check.
