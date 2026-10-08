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

# Source-aware physical prediction with an OakInk2 warm start

User pauses the unlaunched two-rank OakInk optimizer continuation and requests
GRAB,ContactPose,ARCTIC and OakInk2 jointly trained from the completed latest
model weights, with a fresh optimizer/schedule. The Task subgoal is observed
future-hand conditioned physical prediction; robot policy utility is untested.
Result: Four-source14250 preserved; main-three COMPLETED50000, fixed-panel macro moving-anchor h24 EPE34.480→27.126mm; latest/final50000 and best46000 saved.
Decision: Preserve the completed main checkpoint; run the separate bounded history-only ContactPose auxiliary ablation, keeping EPIC non-training.

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

| Source | Imported model, step0 | Step250 | Step500 |
| --- | ---: | ---: | ---: |
| OakInk2 | 9.7872 | 10.622 | 10.487 |
| GRAB | 84.3269 | 72.637 | 68.384 |
| ARCTIC | 76.0066 | 73.394 | 69.343 |
| ContactPose | 40.1089 | 35.750 | 34.756 |
| Equal-source macro | 52.5574 | 48.101 | 45.742 |

These early results suggest new-domain adaptation with possible OakInk forgetting;
500updates do not settle the tradeoff. OakInk partially recovers between250/500
but remains7.15%above its imported-model error; keep source sampling unchanged
while collecting later validation. Step500 macro improves12.97%vsstep0.
The new OakInk panel differs from the
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
Stage500 evidence is independently summarized in
`pointworld-multisource-20261007/stage-00500.json`, with runtime identity,
per-source moving errors/static baselines/rotation and recent loss/speed.
latest/best checkpoint files are both present after the first validation;
checkpoint interval250updates. Supervisor saves latest periodically and on graceful stop; old run checkpoints
remain intact. GPU0 is free after conversion; consequence r4 remains unlaunched.

## Stage 9000, observed 2026-10-07 21:56 Asia/Shanghai

The production run remains RUNNING at approximately 9100/40000 fresh updates.
On the same fixed64windows/source moving-anchor h24 panel (0.8s, mm):

| Source | Imported model, step0 | Step9000 | Error change |
| --- | ---: | ---: | ---: |
| OakInk2 | 9.7872 | 11.3297 | +15.76% |
| GRAB | 84.3269 | 42.3978 | -49.72% |
| ARCTIC | 76.0066 | 57.2700 | -24.65% |
| ContactPose | 40.1089 | 29.6910 | -25.97% |
| Equal-source macro | 52.5574 | 35.1721 | -33.08% |

Step9000 is the best observed macro checkpoint so far. The macro curve is
noisy: step7000/8000/8750/9000 gives36.035/38.180/36.686/35.172mm;
these observations do not establish a plateau. Mean train loss falls from
2.382 over the first500updates to approximately1.65 over the latest500.
OakInk remains a retention concern: step8250 nearly recovers its initialization
(9.815mm), while step9000 worsens again. Do not interpret the macro improvement
as improvement on every domain. Continue the existing bounded recipe; retain
the UNCLEAR classification for joint adaptation/retention until later evidence.

Recent1000updates take approximately0.75s/update including validation and
checkpoint overhead, estimating40000updates around2026-10-08 04:23 if this
speed holds. The hard deadline remains09:58:37. GPU1/2 are still the training
devices, per-rank64/global128, with approximately21.0/20.5GiB device memory.
Two spot samples show utilization96/97% and71/96%; these are instantaneous
observations, not a sustained average. No performance conclusion follows from
them. Both latest.pt and best.pt exist at the step9000 save. Runtime evidence:
`outputs/cm-pointflow-effect-pretrain/pointworld-multisource-20261007/stage-09000.json`
plus the original progress, validation and training logs. Live code, inputs and
sampling probabilities are unchanged.

## Limitations and future evidence

ContactPose global motion is reconstructed from native pose plus fixed hand
annotation; MANO source hand validity is finite-label validity, not visibility.
Human measured future motion is observational, not commanded robot action.
Official splits differ across datasets; report their scope, not generic unseen
object/subject generalization. Missing ARCTICtest must not be backfilled fromval.
Matched controls, independent tests, robot domain adaptation and trained-policy
Cm-on/off utility remain future work; this Probe does not settle those claims.

## 2026-10-07 ref3 Decision Note：停止四源主训练，准备三源主监督

用户要求同时推进 consequence-evaluator ref2/ref3。ContactPose future hand 由
固定手形及逐帧物体/手刚体变换构造，属于刚性抓持运输，不能与测量动态手形的
OakInk2/GRAB/ARCTIC 等价解释。root 选择先保存并停止当前四源运行，保留其
latest/best及验证记录，然后新运行仅抽样 OakInk2/GRAB/ARCTIC；ContactPose
保留为独立辅助/历史诊断，不删除数据。三源以停止后的 latest 模型权重初始化，
重新建立优化器和抽样状态。修订训练源统计/损失尺度并验证 warm-start 的物理
输出合同，禁止只改统计量却让已有权重解释发生不透明变化。

最多继续使用 GPU1/2、每卡64，原截止时间2026-10-08 09:58:37保持不变，
不重新领取24h；现有和新增产物合计仍受300GB限制。数值不有限、数据/源码
漂移、外来GPU进程或deadline触发保存退出。三源近期 moving-anchor 物理EPE
决定是否值得下一轮投入；当前没有 ContactPose 因果影响或 normalization 原因
的正式结论，不立即运行 auxiliary ablation。EPIC保持候选审计，未授权突破外部
只读/GPU权限边界。

## Ref3 三源实现、训练预算及候选审计

四源run于23:00左右保存正常退出，step14250，rank参数哈希一致。
latest与best保留；四源macro moving-anchor h24为31.63mm，末次Oak9.2367/
GRAB38.8630/ARCTIC51.2378/ContactPose27.1500mm。此次结果包含ContactPose
刚性运输，不能解释为四源等价的动作条件因果监督；监督差异是确定的数据合同，
但未实验证明它造成Oak遗忘。

新run_id为pointworld-main3-20261007，数据manifest main-wm30-20261007：
Oak/GRAB/ARCTIC=5/9,2/9,2/9（保留旧主三源相对比例），ContactPose无抽样、
无主loss，不删除其产物。categories不伪造program/background；native .6/.2/.2
在真实moving/static strata归一化为.75/.25。主要指标仍moving-anchor，
新固定验证panel每源64/共192，与旧256panel不同，必须从新step0对比。

停止后的latest14250仅初始化模型；新AdamW、scheduler、抽样和步数重置。
forward input/output保持checkpoint已学习的Oak训练统计；另外冻结2048个
三源训练窗口、seed216/sourceweighted的flow/translation/rotation loss尺度，
按物理残差计算Huber，不改模型buffer或前向物理输出。归一化统计来自train
而非val/test；保留shared source-weighted目标，不能把sampling比例当作严格
梯度贡献比例，也不能声称input domain mismatch已经全部消失。

用户追加授权50000新更新，GPU1/2每卡64/global128，明早2026-10-08 10:00
硬截止（替代原09:58:37）；按旧含val/save的.76s/update约10.5h，启动后重估。
最多2GPU/现有总300GB/free20GiB；stats<=600s/1MiB，新双卡12update
engineering smoke<=300s，不复用smoke优化器。deadline/source drift/非有限/
外来GPU进程触发保存退出。初始化严格核对旧四源到三源的manifest/index子集、
不变模型源码/vendor和统计身份，不单纯放开dataset hash。

EPIC ref3独立CPU文件审计产物：
outputs/cm-pointflow-effect-pretrain/epic-contact-ref3-audit-20261007-r3/audit.json。
读取ObjectForesight官方逐clip convention，w2c取逆；按非连续frame_ids
匹配独立物体pose，保留双手及源side/joint/high-confidence/gap掩码。
P03_03样本源右手有效20/30，30Hz右17/31、物体27/31，4个跨gap点无效；
完整4+24有效窗口从旧未屏蔽4降为0。OF匹配30/30，但共同world中
anchor-relative运动差median1.544m、rotation45.04°，单scale诊断残差
RMS50.68mm；scale未注入数据。手物相对表面gap median3.35mm，不足以
证明动态可信：官方非中央帧由固定手物相对关系传播，绝对位置近似。
仍CANDIDATE_ONLY/training_allowed=false，不新增弱源训练。
官方依据：[ObjectForesight](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC#camera--pose-conventions)、
[EPIC datasheet](https://huggingface.co/datasets/Sid2697/epic-contact/blob/0df7796dba1acdc4d0260b69662524916e9f7079/DATASET.md)。
新audit6tests+原ref5共22通过；三源训练/scale相关18tests通过，均为工程检查。

三源manifest已完成，train4,378,478窗口：Oak3,860,393/GRAB207,179/
ARCTIC310,906。2048个train统计样本1124/476/448，GPU2实际9.595s，
无neural fit；h24 flow std约40.6/43.5/40.9mm，translation std42.4/46.1/
42.8mm，rotation RMS.35524rad。固定loss_stats.json绑定新manifest/hash。
所有41项ref3相关合同测试通过，仓库changed验证通过；代码提交cdb0d68。

正式启动前两卡12update工程检查完成：
pointworld-main3-smoke-20261007，9e1da03，20.87s update阶段；
初始参数哈希1bfa9a70c10bf1c33dd1fc6576182fb1e929898f98f28f8fe3d1199aa107069d
与停止14250的两个rank完全一致，最新模型成功导入；optimizer/schedule/draw
明确重置。末步loss.48334/grad1.91334有限，rank最终参数一致；峰值allocated
14,406MiB/reserved20,636MiB。首步12.12s含编译，后11步平均.6242s/update，
正式val/save还需额外时间；不把20秒总时长线性当作训练ETA。smoke权重不进入
production，production重新从四源latest14250导入模型。

## 三源正式运行已启动

run pointworld-main3-20261007，commit9019fd467c2803888e031b43eda5c505b14e4021，
supervisor1411132/torchrun1411144/workers1411207,1411208，
GPU1/2、perrank64/global128；2026-10-07 23:20:24启动，50000updates，
harddeadline1791424800（2026-10-08 10:00）。初始化parent_step14250、
weights_only=true，optimizer/schedule/draw reset=true；新初始参数与parent
rankhash一致。生产未读取smoke final权重。首步loss.63287/grad2.55545有限，
第2步.6219s；首步含编译，不能当作稳定ETA。
新step0固定moving-anchor h24 panel：Oak15.9166/GRAB36.1329/ARCTIC
51.3909mm，三源macro34.4801mm。新panel与旧四源末次不同，不因Oak
新step0较旧末值大就判断切换退化；后续仅与本panel自己的step0比较。
每250updates val并保存latest/best；deadline自动保存退出。源hash守卫运行中
不编辑实际训练源码、配置、stats或manifest。

首个250update阶段验证及保存已完成；23:24进度305/50000，RUNNING。
新固定moving-anchor h24：Oak15.9166→16.5042mm，GRAB36.1329→
42.7453mm，ARCTIC51.3909→52.2425mm；macro34.4801→37.1640mm。
首轮验证退化，不能以loss或checkpoint继承断言新方案已改善；保持预设配方
继续观察，尚不足以定平台或关闭路线。新loss尺度与旧loss不同，不横比绝对loss。
step50..305含一次val/save实测.74171s/update，预计50000约2026-10-08
09:39，初期估计有不确定性，10:00硬截止保持。最新weight checkpoint250，
best为新panel的初始化step0；两者均保留。peakallocated15,128MiB/
reserved23,392MiB，perrank64未OOM。磁盘审计outputs101G/artifacts.9G/
tmp125G，仍在300GB产物限制内，未删除其他agent产物。

### Main-three step1500 follow-up

At23:41 the production run is healthy onGPU1/2, per-rank64/global128.
Fixed same-panel moving-anchor h24 EPE at1500: Oak11.647/GRAB33.089/
ARCTIC44.896mm; source macro29.877mm versus initial34.480mm.
The early250 degradation has reversed on this panel, still a Probe observation.
Includes validation/save, step50→1595 averages.7444s/update;50000newupdates
ETA2026-10-08~09:41, with user-authorized10:00hard deadline unchanged.
Latest checkpoints are saved every250updates; best is selected from this
three-source panel only. No test evaluation or Cm utility claim.

### Ref3 completion audit and transient validation

Independent read-only review confirms the production main source exclusion,
train-only2048 loss statistics, actual trainer physical_loss branch and
all frozen dependency hashes. Keep the model's forward Oak normalization
to preserve imported weights; do not claim full domain scale correction.
Because initialization is the former four-source14250 model, previous
ContactPose knowledge persists. Exclusion in this run is not a clean
ContactPose-free ablation; separate auxiliary training remains future
evidence until it changes the next decision. EPIC r3 still has0valid
4+24windows and substantial world-motion mismatch, so OF+EPIC usable
training pairs are not delivered and the candidate guard remains closed.

Step2500 same panel moving-anchor h24: Oak17.742/GRAB39.673/ARCTIC50.179mm,
macro35.865 versus initialization34.480. The1500 positive observation
was transient; do not report stable improvement or convergence yet.
Training continues under the user's50000update/10:00deadline protocol.

### Recover adopted monitoring (2026-10-08)

The original launcher PID 1411132 is no longer live and its group summary is
stale at step 4702. Torchrun 1411144 and ranks 1411207/1411208 are confirmed
live, still write the original console/progress logs and carry the original
absolute deadline; worker-side deadline and implementation-drift guards remain
active. All 38 launch-frozen source/input hashes match. Do not restart training
or reset optimizer/sampling state.

The new `tools/run/adopt_pointworld_monitor.py` adopts only those verified owned
process identities, including process start ticks to reject PID reuse. It
preserves the old group record before updating it, keeps the same deadline and
GPU set, restores source/foreign-process checks and requests checkpoint saving
only from verified ranks when a stop condition occurs. It does not create a
model process or add updates. Adoption is bounded by the existing deadline
plus the original 180-second shutdown grace. An adopted process is not a child,
so terminal state comes from trainer output and process exit, not an invented
exit code. Two CPU process-identity/signaling regression tests pass.

### Fixed-panel late-training snapshot (2026-10-08 09:39)

The production run remains live on GPU 1/2, with the original 50000-update and
10:00 absolute deadline. Same-panel moving-anchor h24 EPE at step 49000 is
Oak 14.265 / GRAB 28.214 / ARCTIC 38.002 mm; equal-source macro 26.827 mm,
versus initialization 34.480 mm. Best through this snapshot is step 46000,
26.786 mm. Mean macro over updates (35000,40000], (40000,45000] and
(45000,49000] is 27.904, 27.581 and 27.410 mm respectively. Gains have slowed
but these logs do not establish a complete plateau. This is a Probe panel
observation, not test performance or Cm utility. The bounded log-only summary
is `outputs/cm-pointflow-effect-pretrain/main3-phase-summary-20261008-r1/report.json`;
no extra model evaluation or optimizer change was introduced.

### Main-three terminal result (2026-10-08)

`pointworld-main3-20261007/train-action/result.json` reports **COMPLETED**,
50000 updates, 37705.41 s through final saves, original runtime commit
`9019fd467c2803888e031b43eda5c505b14e4021`. Both rank parameter hashes are
`bbf2fe180f29895a5aafa127e3c6a4c1718ae7f00998dc3b55aedd41cedb7a95`.
The recovered group monitor also reports COMPLETED; torchrun, both workers
and monitor have exited. GPU 1/2 are empty; GPU 0's continuous diagnostic
has also exited. No extra updates or deadline extension occurred.

Final equal-source balanced moving-anchor h24 EPE is Oak 13.778 / GRAB
27.795 / ARCTIC 39.804 mm, macro **27.126 mm**, versus initialization
34.480 mm (21.3% lower on this fixed Probe panel). Natural-panel source
macro is 28.308 mm. Same balanced-panel action-shuffle diagnostic is
125.442 mm; this tests dependence on action conditioning, not online policy
benefit or a matched Cm-on/off trained-policy claim. No test split evaluation
was added. Late block means above show slower remaining improvement; no
claim of a fully reached plateau.

`latest.pt` and `final.pt` deserialize at step 50000; `best.pt` at step 46000
(fixed-panel macro 26.786 mm). All are complete two-rank DDP checkpoints,
578.5 MiB each, retaining model, optimizer, identity and both rank RNG states.
CPU deserialization verifies file integrity/metadata only, with no model
evaluation. Original full checkpoints are preserved; no continuation is
automatically scheduled.
