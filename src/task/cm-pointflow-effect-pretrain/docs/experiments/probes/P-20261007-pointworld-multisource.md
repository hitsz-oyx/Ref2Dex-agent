---
schema: ref2dex.probe.v2
probe_id: P-20261007-pointworld-multisource
experiment_id: P-20261007-pointworld-multisource
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: consequence-evaluator
git_commit: 384f860631c1c436129d692d4d81a50c6219f13a
claim_id: C3
hypothesis_family: HF-pointworld-multisource
probe_index_in_family: 1
seed_pool: probe
seeds: [210, 212, 213, 216, 225, 226]
decision_changed_if_positive: retain mixed-source physical pretraining checkpoint for robot adaptation
decision_changed_if_negative: diagnose source contracts and per-source optimization before further data or steps
status: UNCLEAR
run_id: pointworld-multisource-20261007
---

# Four-source physical prediction with an OakInk2 warm start

User pauses the unlaunched two-rank OakInk optimizer continuation and requests
GRAB,ContactPose,ARCTIC and OakInk2 jointly trained from the completed latest
model weights, with a fresh optimizer/schedule. The Task subgoal is observed
future-hand conditioned physical prediction; robot policy utility is untested.
Result: Full four-source corpus and two-rank engineering checks pass; production started, step250 macro moving-anchor h24 EPE52.56→48.10mm, with early OakInk degradation.
Decision: Continue the bounded mixed-source Probe and track each source against its own fixed step0 panel.

No new branch or remote push. ref6 is background design input; this run does
not add ObjectForesight/EgoDex/HOT3D to the four explicitly requested sources.

## Decision and cheapest useful experiment

The question is whether broader genuine3D trajectories improve per-source
held-out physical prediction while preserving useful OakInk performance. First
resolve the Blocker of incompatible category/split/time/frame contracts using
source conversion and actual loader audits. Then run a bounded engineering
check of the mixed model/data/weights before a single-seed exploratory fit.
No matched data-scale causal claim follows; retaining an improved checkpoint
changes the next robot adaptation initialization. Poor results require checking
source contributions and implementation before assuming more steps will help.

Resources: native GRAB/ARCTIC full conversion on emptyGPU0, <=3600s/5GiB;
ContactPose pure file/geometry conversion on CPU, <=3600s/5GiB (no neural model
or MANO needed). Corpus metadata references arrays instead of copying them.
GPU1/2reserved for new fit, per-rank batch64/global128. At most3simultaneous
GPUs including consequence work; hard shared4GPU/300GB cap and20GiBfree remain.
New fitting is <=40000fresh updates, additionally bounded by the original
OakInk deadline1791424717.7631629. No fresh24h allocation. Stop on nonfinite,
OOM, input/source drift, missing valid source strata, foreign jobs or budget.

## Frozen data semantics

Keep current architecture:4history frames,24future steps at30Hz,0.8s horizon,
right/left11semantic points, one current anchor frame,0.5m local object selector,
fixed512canonical points/object, original loss and temporal/spatial identity.
All static/current hand motion and future GT are built from real source labels;
no future contact/reward/progress is given to the model.

- OakInk2: reuse627processed sequences and original501/70/56sequence split.
- GRAB: all1335locally available motion sequences,120→30Hz decimation; preserve
  native hand/object parameters. Official object-based heldouts are used.
- ARCTIC: all301local sequences,30Hz; millimetres converted to metres,
  articulated top/bottom separate rigid parts. Official protocol_p1train/val
  membership is used. Local officialtest is absent; report it as missing.
- ContactPose:1591unique top-level grasps, not1590duplicate nested copies.
  Native object/world and moving-hand relative transforms and ROS timestamps
  restore continuous world motion. Contact mesh geometry is already metres;
  separate canonical ZIP meshes are millimetres and are not directly reused.
  Split by participant, all segments from one grasp stay together. Split at
  invalid poses,>75ms gaps or abnormal speeds before30Hz pose interpolation;
  windows never cross gaps. Fingers remain the source's fixed grasp annotation:
  this source supports grasped-object transport, not dynamic finger/contact GT.

Native index categories1moving/2near-static become common0moving/1selected-static.
Missing program/background strata stay absent. Source-internal .6/.2/.2weights
are renormalized over actual strata. Source probabilities are OakInk.5,GRAB.2,
ARCTIC.2,ContactPose.1, limiting fixed-grasp overrepresentation; these are a
declared exploratory recipe, not a tuned scientific optimum.

## Model initialization and evaluation

Parent: `pointworld-action-ddp-20261007/train-action/latest.pt`,step10000.
Strictly import all model weights/buffers. Fresh AdamW1e-4/wd.01,100warmup,
cosine floor.1, clip1, BF16 forward/FP32 physical loss, seed225/draw226.
Reuse original train-only normalization with its hashes and exact included
OakInk corpus; do not silently overwrite learned scaling buffers with new stats.
New dataset identity is explicitly accepted by a separate mixed initialization
path that still validates frozen model/geometry/normalization/vendor sources.
Optimizer/RNG/step are reset: this is pretraining initialization, not resume.

Validation: fixed64balanced windows from each of4sources, batch2, every250updates;
best selected by equal-source mean moving-anchor h24point EPE. Report source
metrics and static/rotation/shorter horizons separately; a pooled mean alone
cannot show all sources improved. Compare each source against its own imported
step0baseline on the exact new panel. Final natural panel and action shuffle
remain diagnostic. No TEST tuning or checkpoint selection. Old OakInk best/
latest/final and all original curves remain intact. Neither training loss nor
human future-hand dependence establishes robot action causality or policy gain.

## Outputs and current status

- Native full: `outputs/cm-pointflow-effect-pretrain/native-official-full-20261007/`.
- ContactPose full: `outputs/cm-pointflow-effect-pretrain/contactpose-native-full-20261007/`.
- Mixed frozen manifest: `outputs/cm-pointflow-effect-pretrain/mixed-wm30-20261007/`.
- Fit: `outputs/cm-pointflow-effect-pretrain/pointworld-multisource-20261007/`.

Small native official reindex passes26755category/clock/rigid-target checks;
ContactPose9grasp/26segment engineering pack passes the actual loader. These
are engineering readiness checks. Full conversion and mixed fitting subsequently
started; actual full-corpus counts and runtime identities follow below.

Full preparation launched at source commitd6fb20f: native supervisor936433,
ContactPose CPU936434. Each uses the bounds above. The actual mixed-interface
pilot (full OakInk plus official-reindexed40native sequences and9ContactPose
grasps) passes CPU source sampling/clock/category/identity contracts.13root
contract tests pass, including strict model-only scale/source checking and the
existing distributed objective/resume/RNG checks. No future label is replaced
to satisfy the sampler.

Two-rank real-data engineering check at commit00fa494 completes12fresh updates
in25.02s, with initial parameter hash495bdc2aacf5fbbc8188ab7ebdd992f69075ec8f29b6a82c41968d34b8dbc83f,
exactly the original latest model. Both final rank hashes match; serialized
AdamW steps are12, confirming the new optimizer. Median update excludingfirst
is0.615s, peak live allocation13.53GiB/reserve17.46GiB. These short-run values
do not guarantee full-corpus peak VRAM or production speed. Evidence:
`mixed-interface-ddp2-check-20261007/{input_manifest.json,result.json,verification.json,final.pt}`.
The engineering weights are discarded; production imports the original latest.

Queued entry: `tools/run/queue_multisource_training.py`. It waits for both
owned bounded conversions to finish and permit fitting, detects source/input
drift, freezes a new mixed manifest, checks empty GPUs1/2, and repeats12real
engineering updates on the full corpus before production launch. Failure
halts the queue and preserves logs. Queue artifacts:
`outputs/cm-pointflow-effect-pretrain/multisource-queue-20261007/`;
full-corpus engineering artifacts: `pointworld-multisource-full-check-20261007/`.
GPU usage samples are recorded throughout production. The queue was initially
a pending arrangement; the following actual runtime record supersedes that state.

## Full-corpus launch and first validation

Actual production starts2026-10-07 19:57:29Asia/Shanghai at source commit
384f860631c1c436129d692d4d81a50c6219f13a. Queue976793 launches supervisor980674
and torchrun980684 on GPU1/2. Input manifest records50,495,881parameters,
per-rank64/one accumulation/global128 and the original latest's exact parameter
hash495bdc2aacf5fbbc8188ab7ebdd992f69075ec8f29b6a82c41968d34b8dbc83f.
Initialization explicitly records parent_step10000, weights_only and optimizer/
schedule/draw reset. Production starts at step0; full engineering weights are
not imported. Deadline remains2026-10-08 09:58:37Asia/Shanghai.

Full GRAB/ARCTIC conversion finishes in1079.65s:1335/301sequences,619322windows;
audit ENGINEERING_PASS, maximum rigid correspondence error8.95e-8m.
ContactPose audits all1591unique grasps in1010.50s;61lack a valid contiguous
28-frame segment and are excluded. Remaining1530grasps produce5168segments.
Participant counts37/6/5 have disjoint train/val/test membership. This is our
participant split, not an official ContactPose benchmark split.

| Source | Train windows | Val windows | Test windows | Sampling probability |
| --- | ---: | ---: | ---: | ---: |
| OakInk2 | 3860393 | 656368 | 543855 | 0.50 |
| GRAB | 207179 | 18291 | 41916 | 0.20 |
| ARCTIC | 310906 | 41030 | 0, local officialtest missing | 0.20 |
| ContactPose | 415751 | 74193 | 40646 | 0.10 |

Frozen mixed manifest SHA256:
5ebace706378315b2f6cfe1a8b46d8ca7d84f6c5d5d953269015d8859622ac6d.
All source/index identities are in the runtime input manifest. Test counts are
metadata audits only; test tensors are not used for training or model selection.
Full-corpus two-rank engineering check completes12fresh updates in21.80s;
rank parameter hashes identical, all logged losses/gradients finite. Median
update excludingfirst0.601s, peak live13.45GiB/reserved18.38GiB. This proves
wiring/readiness, not predictive quality.

Production fixed64windows/source validation, moving-anchor h24point EPE inmm:

| Source | Imported model, step0 | Step250 |
| --- | ---: | ---: |
| OakInk2 | 9.7872 | 10.622 |
| GRAB | 84.3269 | 72.637 |
| ARCTIC | 76.0066 | 73.394 |
| ContactPose | 40.1089 | 35.750 |
| Equal-source macro | 52.5574 | 48.101 |

These early results suggest new-domain adaptation with possible OakInk forgetting;
250updates do not settle the tradeoff. The new OakInk panel differs from the
previous256-window OakInk panel, so9.7872mm is not a direct comparison with
that run's11.1342mm. Keep this run UNCLEAR until later per-source evidence.
At step250 median recorded update time is0.632s; peak live13.98GiB/reserved
23.13GiB. First250updates take205.64s before the first~26s validation. A rough
40000-update estimate is8–9h including periodic validation, subject to corpus
I/O, variable object count and resource guards; the original deadline wins.
GPU snapshots during the first update block show40–42%utilization, so this
run does not claim to have solved the earlier GPU utilization issue.

Evidence: `pointworld-multisource-20261007/{group_status.json,console.log}`,
`train-action/{input_manifest.json,validation_initial.json,validation.jsonl,train.jsonl,best.pt}`;
`native-official-full-20261007/audit.json` and both source processed manifests.
Supervisor saves latest periodically and on graceful stop; old run checkpoints
remain intact. GPU0 is free after conversion; consequence r4 remains unlaunched.

## Limitations and future evidence

ContactPose global motion is reconstructed from native pose plus fixed hand
annotation; MANO source hand validity is finite-label validity, not visibility.
Human measured future motion is observational, not commanded robot action.
Official splits differ across datasets; report their scope, not generic unseen
object/subject generalization. Missing ARCTICtest must not be backfilled fromval.
Matched controls, independent tests, robot domain adaptation and trained-policy
Cm-on/off utility remain future work; this Probe does not settle those claims.
