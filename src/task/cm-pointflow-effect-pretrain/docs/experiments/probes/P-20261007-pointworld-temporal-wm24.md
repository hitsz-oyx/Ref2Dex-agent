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

Result: at matched6000-update validation, H+A moving-anchor h24 point EPE is15.63mm versus H21.81mm and shuffled-action21.90mm; training and final TEST are incomplete, so status remains UNCLEAR.

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

## 阶段性验证记录（2026-10-07）

以下为同一训练步数、固定 balanced256 验证窗口上的对比。指标是
`model/anchor/cat0/h24/point_epe`，即运动锚物体未来24帧（0.8秒）的点位置
误差，表中单位为mm；不是训练loss，也不是TEST结果。

| 验证步数 | 仅历史 H | 历史＋真实动作 H+A | 历史＋打乱动作 | 静止预测 |
| --- | ---: | ---: | ---: | ---: |
| 3000 | 21.88 | 16.42 | 21.75 | 26.13 |
| 6000 | 21.81 | 15.63 | 21.90 | 26.13 |

第6000步动作组误差比H低28.3%、比打乱动作低28.6%，比静止预测低40.2%。
相对第3000步的16.42mm，动作组误差进一步下降到15.63mm，两个对照变化较小。
但第6000步的运动锚物体旋转误差仍约10.2°，接近静止基线；当前改善主要体现在
位置预测。这里只记录继续完成当前Probe的正向信号，不升级为正式科学结论，
也不提前授予最终PROMISING标签。

14:15（Asia/Shanghai）运行检查：H/H+A/shuffle分别6903/6641/6400步，
最近100步平均训练loss分别1.22/1.01/1.21；三个worker运行中，监控源码hash
无漂移。按当时近期更新耗时，三组完成还需约13h，仅作运行估计。
继续原定40000步训练；当前验证结果不改变数据、模型、预算或最终判定门槛。

追溯产物：运行目录下`stage-snapshots/interim-3000-6000.json`保存两次已读取的
匹配阶段指标摘要、运行commit，以及归档时仍可读取的完整验证JSON和其SHA256。
第3000步和第6000步的数值此前直接读取自各组`validation_latest.json`，
当时未归档完整JSON；该文件会滚动覆盖。补记录时H已进入第7000步验证，故归档
的完整原始文件为H7000/H+A6000/shuffle6000，明确保留各自step，不能作为
匹配比较，也不能冒充第3000/6000步完整指标的重建。历史阶段仅保留上述实测摘要。

结论仍为**UNCLEAR（训练中）**。最终仍需等步数40000的三臂TEST及动作组的
推理时shuffle干预，才能按既定门槛判断本Probe；阶段性验证不证明全局C3或策略收益。

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
