## 2026-10-22 并行执行线程：GT-τ 执行资格未通过；三臂首批准备完成

本线程独立推进机器人闭环，不重复另线程的可辨识性/goal契约诊断。
固定已适配τ的Scheduled686 step50000，同16个机器人DEV初态（相对其TRAIN名称/源路径
交集0），原动作回放抓取抬升持握15/16=93.75%，标准机器人GT-τ0/16，健康4371预测0/16，
固定FM8500预测1/16。标准GT与预测均前3控制zero/invalid；额外live-centered GT也0/16，
但其旧startup不同，单列oracle。五臂初态7数组逐项exact；每臂native300步，PD误差/episode错配0。
平均最长连续持握分别2.090/0.0542/0.0438/0.0625s（原动作/标准GT/健康/FM）。完整原路径任务
均0/14未删失，另2例截断；不要当16例全任务失败。自然失持恢复原动作0/2、FM0/1，其余无
机会记UNKNOWN；强制扰动与mesh穿透未测。首FM整批越界失败已保留，修复逐环境边界后完成；
末FM完整300步workspace2/16均在各自准入前缀后，主指标0/16。

结论：当前执行桥连可执行机器人GT-τ都不能稳定执行，资格门未过，不能断言τ解码质量是
主瓶颈。范围仅这个v4-τ训练的执行器/v6适配桥，未训练的新direct executor仍不能伪报完成；
WAM祖先终身未见也无法认证。数字/失败类型/配对CI/精确命令见
`src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-executor-qualification.md`，
产物`outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/summary.json`。

P1已先写设计，单Inspire、cubemedium/cylindermedium/mug，统一真实CAD2048、相同BC、奖励、
显式position-only抬升8cm/HOLD1s与共同H4；A无τ、B几何优化参考、C健康4371预测。
3独立seed191/192/193，每臂每seed目标1,048,576在线控制；共9,437,184，eval共同seed194，
14字段初态exact校验。B是受限几何优化候选，未达到完整碰撞约束IK强基线，不能据C>B宣称
打败所有几何重定向。256环境真实工程容量A/B/C均2更新40optimizer/4096交互，错配/PD/
logprob0；C batch64保留24帧、跳独立未用C头后实测26.921s/更新，改统一6h/run硬界。
科学首批尚待启动，工程smoke不是学习曲线或成功率证据。P2仅公平训练设计，0运行。

本线程GPU2/3完成P0；原诊断后来使用0/2继而0/3，未干预。P1将用实测空闲1/2；
GPU4/5/6长训、7、凭据、GitHub push均未触碰。仅本地提交。整库verify受6张既有旧卡
schema缺字段阻断；本线程3卡schema/seed检查PASS。

## 2026-10-22 大规模 goal 契约诊断进行中

本轮仅用既有健康4371与FM8500，未启动训练/扩容/加数据。全DEV3356、TRAIN5241/1064序列、
本地未覆盖前缀905；FM K8，GPU95–99%。GPU2被并发作业占用后仅迁移本轮自己的诊断到3，
另一worker用0；4/5/6长训与7及其他作业未动。调度器是spatial_path_waypoint且endpoint_time=None，
训练是实际t24终点，语义不同。健康全DEV前8手/物EPE Oracle15.80/14.72→Planner21.12/25.55mm，
C F1 .686→.620；FM全量尚待完成，不拿部分面板混比。DEV约69.85% Oracle目标超调度器单次边界。
事件对齐方差扩大候选与CAD占用复核正在做；executor三条件只写设计，不执行。
卡片：`src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-goal-chain-audit.md`。
产物：`outputs/ptv3-shared-wam-v6/goal-chain-audit-20261022-r1/`。

## 2026-10-22 第一阶段现有checkpoint诊断完成，无新训练/扩容/加数据

固定健康4371与FM8500；GRAB三面板各96窗口：TRAIN95录制、序列DEV66、本地未覆盖
前缀93。种子178，统计180，flow噪声172；H4+K24完整28帧支持排重，原生TRAIN1040/
DEV92录制交集0。祖先远端窗口清单缺失，本地未覆盖前缀不能宣称终身未见。

可辨识性待定：4096TRAIN库，50mm严格组仅5组/30录制，匹配事件时间后1组。
组内手8方差1609→接触区域1927→手指2190→交互类型2663mm²，无可靠收缩。
配对原始3031→2034的约33%下降，被未来是否接触控制（188对2029）基本解释；
同区域66对2034未再降。100mm宽组全24手方差接触区域降约42%，只能算粗条件信号。

前8手/物EPE：TRAIN10.42/4.55→9.90/4.97mm；序列DEV20.28/9.36→21.78/10.97mm
（配对增量95%CI手[0.89,2.07]、物[1.08,2.12]）；本地未覆盖40.64/2.80→42.34/5.55。
DEV C头MAE0.05546→0.04934却几何GT接触MAE0.06826→0.06942，C改善未保证τ改善。

DEV前8预测手接触距离GT物/预测物：健康15.025/10.477、FM15.534/10.630mm，
差值−4.548/−4.904，未见大的正分离缺口；不能凭相对距离更小认证绝对预测正确。
16窗口/15录制的t8/t24闭合物体mesh复核，每锚点/时刻1024个手皮肤点均未检出>1mm
穿透；手仍是观测皮肤运输代理，不能外推全表面。物体刚体拟合残差t8=4.60→7.83、
t24=5.71→15.35mm，非刚体性上升；GT刚体控制最大0.0000313mm。

GPU0等待他人临时占用后续算，GPU1仅几何smoke；未信号其他作业或GPU4/5/6长训，
未碰7/凭据、未push。原长训仍RUNNING9700/max43200不变；本轮输出1.4GiB，余149.46GiB。
4测试、实数据平移/归一化/立方体/CAD内部控制通过。数字与完整60分层/命令见
[实验卡](experiments/probes/P-20261022-wam-v6-identifiability-audit.md)，
数据根`outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/`，入口`summary.md/summary.json`。
物理可控性未由离线预测误差验证，等待机主对数字的下一步决策。

# 2026-10-22 FM trial complete; sustained three-rank training RUNNING

Frozen0917f47e;supervisor1926227 PPID1/SID1926227,torchrun1947491 separateSID.
Only GPUs4/5/6,local96/global288,strictFP32,latestGT-object2048 contact0.05,
goal/events retained. No other jobs/cards/credentials touched.47 tests and real
preflight/capacity8/DDP8+strict2/trial128/ODE audit PASS:617 finite Adam states,
RNG3,exact scales,equal rank signatures. Fixed four-TRAIN teacher MSE ratio0.883
to current zero field; fixed GRAB DEV prefix-object17.800→18.699mm and hand
55.222→56.571mm,no held-out benefit yet. Owner-directed longer exploration.

At01:33 main160 loss0.8639/norm1.2235/shared-context RMS3.77/5.84 finite,
0.310updates/s,GPUs89/93/98%,~20GiB/card,free153.79GiB. Bound30k or cumulative
12h;estimated~12k–14k/13:20–13:30+08,08:00~7000–7500,provisional. DEV/save500.
Local207579 TRAIN/3356 DEV,~99.8%GRAB;no new data. Output
`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/`:training.log,
training/,process.json boundedETA,startup-health.json. Launcher receipt and
supervisor.log in matching tmp/run directory. Disk20GiB floor;owned rootSTOP
requests saving/stopping. No rollout/effectiveness claim.
[Full record and exact commands](experiments/probes/P-20261022-wam-v6-flowmatching.md).

# 2026-10-22 Joint conditional residual flow trial RUNNING

Owner requests runnable FM trial then longer three-card training onGPU4/5/6.
Latest GT-object2048 contact retained,FP32,goal/events,healthy4371 plus new flow.
Bound30000 updates or12h cumulative training,detached,no credentials. Frozen0917f47e/supervisor1926227 PPID1.47 tests,real-input
gradients,capacity96/DDP8+2 PASS:617 finite Adam states,RNG3,exact normalizers
and equal rank signatures. Trial60 loss0.8784/norm0.923,all GPUs100%;
continue through128/sampling audit before automatic long stage. Local207579 TRAIN/3356 DEV,~99.8%GRAB.
08:00 is interim review, no efficacy claim.
[Experiment](experiments/probes/P-20261022-wam-v6-flowmatching.md).

# 2026-10-22 GT2048 contact arm saved/stopped3238; no held-out benefit

Execution STOPPED normally at the original4h boundary, final complete
training/step003238.pt (1,186,351,949bytes). CPU audit PASS:568 finite optimizer
states all3238, model finite, RNG3/world3/global288, all4 normalizers exact.
Supervisor1572421 and torchrun1583025 exited; no extension/restart.
Checkpoint cumulative elapsed14402.59s; stage-only result14286.84s excludes
readiness/recovery time. final-stop-audit.json records the evidence.

Latest fixed DEV is3000, not3238: GRAB hand52.588/object-prefix8=18.231/
object-h24=14.660mm, previous matched FP32=52.229/18.131/14.150mm.
GT geometry contact MAE0.14122→0.14177, no gain; macro hand32.647→33.383mm.
Do not extend this package based on these results. No statistical universal
harm, sole-cause or rolling-success claim. Retain20/3000/3238 and separate
readiness/recovery artifacts for inspection; original FP32 source3337 preserved.
[Final experiment record](experiments/probes/P-20261021-wam-v6-gt-object-contact.md).

# 2026-10-21 GT2048 contact probability training RUNNING

User-chosen GT2048 object plus predicted-hand contact-probability supervision
actually running, frozen39d66051/supervisor1572421, GPUs4/5/6/local96/global288.
Same healthy4371 initializer; original tau/scene/C/goals/events/strictFP32 retained.
GT future labels are loss-only. Original FP32 saved/stopped3337 with full audit.
44 tests and real preflight/capacity96/readiness8/strict2 recovery all PASS;
568 states, RNG3, exact scales and3 equal rank signatures. Calibrated weight0.05,
ramp100. New initializer physical DEV metrics match the prior arm (<0.0001mm).
Sustained startup at60, loss0.7292/norm0.916, finite activations;
warmups not yet complete. Bound10000 updates or4h new-objective time; likely
binds time first at initial~0.32updates/s. Free178.56GiB,20GiB floor.
No efficacy or rollout claim; main comparison is prefix1–8 hand/object errors,
new GT geometry contact metrics and unchanged predicted-both geometry diagnostic.
[Experiment](experiments/probes/P-20261021-wam-v6-gt-object-contact.md).

# 2026-10-21 Fixed TRAIN probe supports a generalization gap

32 fixed GRAB TRAIN examples improve from4371 to FP32 new3000: object prefix8
6.509→5.784mm, hand23.809→21.120mm, normalized scene loss0.0634→0.0465.
Fixed DEV object prefix8 worsens17.018→18.131mm. Optimization still progresses;
capacity saturation and gradient conflict are not established. Offline object
t+24 endpoint is directly conditioned; relative geometry permits common absolute
drift. Neither endpoint nor contact improvement certifies useful prefix gains.
GPU7 read-only probe20.1s/<0.453GiB; active three-card training unchanged under
original4h bound. Next work should separate held-out data slices and prefix
selection before extending training or expanding the FFN.
[Experiment evidence](experiments/probes/P-20261021-wam-v6-strict-fp32.md).

# 2026-10-21 FP32 stable through3000; short-horizon gain unresolved

Actual strict-FP32/backend arm has passed the BF16 instability interval.
At3040 gradient0.761 versus BF16=83.95; shared/context RMS3.63/5.64.
GRAB fixed DEV3000 hand52.229mm, object-h24=14.150mm versus BF16=39.462mm
and own init15.829mm. Object-h08=33.398mm still worse than init30.934mm.
Available-source macro hand32.647→33.244mm and DEV objective1.1253→1.1743
worsen; no overall accuracy or rollout-drift claim. Continue unchanged4h bound,
about50min remaining; GPUs99–100%, free184GiB. No new rolling video evaluation.
[Precision experiment](experiments/probes/P-20261021-wam-v6-strict-fp32.md).

# 2026-10-21 Healthy4371 strict FP32 training RUNNING

User-requested healthy-start full FP32 arm actually launched, frozen1b57a487,
supervisor1234454, GPUs4/5/6, local96/global288. Capacity8, real three-rank8-save
and strict2-recovery PASS:568 finite states/correct counts,3 RNG, exact scales,
equal signatures. Actual PTv3×16 and dense attention×13 use FP32 efficient SDPA;
AMP and matmul/cuDNN/spconv TF32 off.38 tests pass. No vendor/architecture edits.
Same source4371/geometry weight/goals/events/local207579 TRAIN/3356 DEV/seeds/LR;
fresh optimizer only at initialization, then strict recovery/continuation.

Sustained startup accepted at100, loss0.7583, norm0.851,
shared/context RMS3.87/6.06; LR and geometry warmups both complete.
Recent0.303updates/s, GPU average~96–97%, physical20–22GiB/card,
free187.08GiB.10000 updates or4h cumulative; at FP32 speed time bound likely
binds first. Root STOP saves/stops. No later-stability/drift benefit established.
Initial GRAB hand52.6872/object-h24=15.8287mm; step20=52.5363/15.0312mm.
Short prefix only; compare500/1000+ and frozen rollouts before any efficacy claim.

Previous BF16 geometry run COMPLETED10000, final GRAB hand56.4268/object-h24
30.3106mm, final checkpoint audit PASS, GPUs released before this arm. The
queued no-geometry control is not launched during this user-prioritized FP32 run.
[Precision experiment](experiments/probes/P-20261021-wam-v6-strict-fp32.md).
Live: outputs/ptv3-shared-wam-v6/strict-fp32-pretrain-20261021-r1/process.json
and training/progress.json; startup-acceptance.json records checks and telemetry.

# WAM v6 state — 2026-10-21

## 2026-10-21 Shared FFN scale drift localized; partial natural recovery at9000

Read-only four-TRAIN-example audits reproduce degraded scene gradients without
geometric loss. PTv3 token RMS1.62→1.86, but shared Transformer last-layer RMS
3.58→423.15 and scene_context5.58→3408.50 (source4371 versus geometry4500).
FP32/FP64 reduce numerical amplification but retain large scene-head gradients.
LayerNorm counterfactual lowers gradients yet worsens existing checkpoint loss,
so it is not a validated hot repair. Six disposable optimizer steps do not show
that resetting Adam resolves the issue. No source checkpoint or active run
changed; original trigger and matched-control causality remain open.
FFN-only4371 rollback in4500 copies lowers shared RMS423→14.45, while attention-
only rollback leaves355.62; partial hybrids worsen accuracy, so no hot repair.
All4 normalizer tensors unchanged;568 finite optimizer states all4500, no
negative second moments. Active run naturally recovers at9000: shared RMS5.12,
scene-head gradient6.40; fixed GRAB DEV hand55.45mm/object-h24=37.26mm versus
initial52.68/15.82mm. Not permanent monotone collapse, still no net object gain.
PointWorld source comparison: GELU/4x/pre-norm FFN core matches exactly in a
same-weight forward/backward smoke; release LayerScale disabled. Shared-stack
DropPath and layer initialization differ from PointWorld. Missing final norm
alone is not a demonstrated explanation. See paper/pointworld/ffn-comparison.md.
[Detailed diagnostic evidence](experiments/probes/P-20261021-wam-v6-geometry-consistency.md).

## 2026-10-21 Geometry rollout measured; matched control awaits resource plan

Frozen parent4371 versus geometry+4500 evaluated on GPU7 (436.53MiB allocated)
without touching active4/5/6 or unrelated users. Same four old GRAB TRAIN
anchors/H4/true t+24 goals, forecast24/consume8/predicted-state feedback, common11
scoring. Old v4/parent4371/geometry4500 means in mm: eyeglasses67.07/90.36/88.40,
bowl220.00/292.50/302.08, stamp52.19/179.84/195.66, bunny43.79/178.22/228.01.
Thus no consistent GRAB drift improvement; three worsen relative to parent.
All eight generated-history/raw-coordinate audits pass, no stop/truncated tail.
V6 native21/observed skin differs from v4 inputs; not a matched-loss causal test.

Apple same sparse11 initial observations (missing geometry explicitly masked):
geometry fixed replay improves old-v4 mean674.87→228.32mm, but endpoint1272.47mm.
TaskController unknown-event mean664.59→330.26mm, worse than parent260.58mm;
planned-event variant318.39mm, parent179.88mm. Both stop acquire_timeout at15s,
447 common frames, zero established grasp ticks. No physics/generalization claim.
Seven four-panel videos are complete under the evaluation output.
Additional recorded-plan GRAB arm: geometry181.38/286.39/188.94/259.43mm,
three cases worse than parent; planned instructions do not eliminate drift.
Four separately labelled planned videos are complete;13 MP4s including two
combined panels pass ffprobe and selected-frame inspection.16 raw audits pass.

Control entrypoint/test/queue prepared,31 v6 tests pass. Same source4371 and
fresh optimizer/draw streams/world3/local96/global288; only geometry training
graph disabled, DEV diagnostics retained. All new control checkpoints preserved.
Existing geometry500/4500 pinned independently before source rotation. Proposed
automatic three-card launch after current run naturally releases4/5/6 requires
the pending user resource-plan response; no watcher/control training started.
Current geometry training continued untouched to5620 at14:14 (loss9.44425).
[AB experiment](experiments/probes/P-20261021-wam-v6-geometry-ab.md).
Artifacts: outputs/ptv3-shared-wam-v6/geometry-ab-rollout-20261021-r1/evaluation/
(result.json/comparison.json/data-audit.json) and control-queue-plan.json at root.

## 2026-10-21 Geometry consistency training RUNNING from latest local4371

User-requested predicted hand/object geometry objective implemented and launched
from the baseline's normally saved final4371, frozen b852f39c. SharedWAMV6 goal
and event conditioning remain active. Observed skin follows predicted joint21;
predicted object flow moves surface queries. Positive contacts supervise the
predicted relative vector against its GT-proxy counterpart; local noncontact
clearance adds separation gradients. Approximate geometry, not exact mesh/SDF.

Old C-only loss has no direct coordinate-readout gradients; actual source4371
GPU preflight confirms new gradients to both readouts/decoded outputs and100%
eligible coverage on four TRAIN examples. TRAIN-only gradient calibration gives
weight0.02292055275, ramp100. All model weights/normalizers copied exactly;
fresh optimizer/RNG/count/selection for the changed objective.30 v6 tests pass.
Three-card8-update save→2 strict recovery passes full568-state/RNG/normalizer
checks and identical rank signatures. Sustained stage has reached100 with finite
loss and nonzero geometry terms. Capacity96 passes13.59GiB allocated/15.34GiB
reserved; training allocated peak14.12GiB, sampled utilization100/95/99%,
physical18–21GiB/card. Free197.71GiB. Initial compute~240samples/s; periodic
DEV/checkpoint overhead affects sustained ETA.10000 updates or new-objective
cumulative4h boundary,20GiB disk floor. No accuracy or drift benefit established.

GPU4/5/6, global288, existing TRAIN207579/DEV3356 recipe and normalizers retained.
Exact4371 baseline GRAB handEPE52.6763mm/geometry relative22.5379mm on fixed32
DEV examples; compare future metrics with this before point. Old expanded run
is STOPPED4371; older RUNNING sections below are historical. Live supervisor
943417, outputs/ptv3-shared-wam-v6/geometry-consistency-20261021-r1/process.json
and training/progress.json; root STOP saves/stops.
[Geometry experiment](experiments/probes/P-20261021-wam-v6-geometry-consistency.md).

## 2026-10-21 Historical three-GPU expanded training (now STOPPED4371)

Latest user requests multi-GPU, utilization and scale. Original single-GPU r2
normal STOP saved1269. Explicit local data/topology transition retains model,
568 optimizer states, normalizers, elapsed, selection and rank0 RNG;3-rank
readiness1269→1277 and strict resume→1279 passed finite checkpoint audits and
identical rank signatures. Full v6 suite23 passed.

Current local-expanded-ddp-20261021-r1, frozen b1821d9c, GPUs4/5/6, local96/
global288,8 workers/rank. This supersedes the previous two-idle-card arrangement
for this user-requested run; no unrelated GPU0–3/7 processes affected. Capacity128
adds only0.8% throughput versus96 but reserved21.90GiB versus17.02GiB, so96
selected. Readiness compute248samples/s; sustained1300→1360 interval242.37samples/s
versus the old single-GPU1020→1180 interval85.59samples/s, a measured2.83x
increase. Different window pool/batch: practical throughput, not identical-load
DDP scaling efficiency. Active at1360; peak allocated14.05GiB, physical~18GiB
per card, free206.70GiB. Recent four10s GPU samples average76/98/96%, with
startup/loading/synchronization fluctuations; do not equate NCCL spin with
productive throughput. Initial ETA~3h to10000, within remaining4h cumulative
budget. Acceptance snapshot contains the exact intervals and telemetry.

TRAIN33672→207579 (all inherited GRAB TRAIN windows, sixfold), DEV3356 unchanged,
1177 existing sequence caches reused, no new raw data copies. Still a GRAB-
dominant subset, not the remote nine-source frozen recipe. Target cumulative
10000 updates or14400s including earlier single-GPU training, LR1e-5/warmup50
unchanged, DEV/checkpoint every500. Source checkpoint/readiness retained.
Live outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/process.json and
training/progress.json. Root STOP saves/stops. [Experiment card](experiments/probes/P-20261021-wam-v6-local-subset.md).


## 2026-10-21 Independent local subset training RUNNING

User authorized completing preparation, saving/stopping PPO and starting WAM.
1177 local sequences/33672 TRAIN/3356 DEV windows admitted under an explicit
local-only subset scope, disjoint sequences; original engineering data unchanged.
Initialize model-only from local step23758, exact inherited normalizer buffers,
new optimizer/RNG/count. GRAB~98.81% TRAIN sampling; no new coupling loss or
claim of strict full-nine-source continuation. All original full-recipe gates
remain strict.21 v6 tests pass; real2351 reader endpoints pass.

PPO saved/stopped normally at1575, complete61,158,850byte checkpoint verified;
all3 ranks/six workers/supervisor exited. WAM8-step save→2-step strict resume
passed:568 optimizer states all at8 then10, finite and scales unchanged. Frozen
d20b565c; current local-subset-pretrain-20261021-r2 training step80 at startup,
GPU4 single/local64/global64, LR1e-5/warmup50, target2000 or cumulative4h.
Measured1.36updates/s, allocated peak9.54GiB/physical12–13GiB, GPU76–100%;
GPU5/6 idle, free216.92GiB. Initial ETA~30min including approximate overhead,
not a promised completion time. First r1 attempt had no update, failed on long
IPC socket path; fixed with project-local short tmp path, preserved evidence.

Latest sustained checkpoint/DEV200 published. GRAB hand EPE49.9303→51.5141mm,
macro loss1.077912→1.065935; no accuracy benefit established. Continue the bounded
pilot, preserving initial/readiness/best checkpoints and reporting metrics apart.
[Local training card](experiments/probes/P-20261021-wam-v6-local-subset.md).
Live: outputs/ptv3-shared-wam-v6/local-subset-pretrain-20261021-r2/process.json;
startup-acceptance.json and training/progress.json; supervisor806802/worker810310.
Root STOP requests saved termination. Existing remote strict resume remains
blocked on its recipe/stats/index inputs; local adaptation does not replace it.


## 2026-10-21 Local data subset prepared; no training

Fresh source-header inventory covers39526 sequences. Target /home2 (/dev/md0)
started228.10GiB free; current about222.35GiB. Prepared and independently verified
1177 complete sequences (GRAB1132 plus45 across the other8 sources),5.69GiB,
33672 TRAIN/3356 DEV bounded engineering windows. No deletion or training;
training_allowed=false. Original frozen recipe/window indices/strata/stats remain
unavailable, so strict step23758 resume is not ready. Details and artifacts:
[local preparation record](research/2026-10-21-local-data-preparation.md).


Post-reboot resume requested, currently blocked on SSH connectivity. Three
attempts to127.0.0.1:22023 timed out during SSH banner exchange; the local port
is listening but remote boot/NAS/GPU status cannot yet be verified. No training
has been launched. Explicit world3→4/local128/global512 restoration is prepared
in13612688 and8 topology/actual-entrypoint tests pass. Reconnect and inspect
four-GPU health before choosing it; otherwise retain strict world3 recovery.
The local full checkpoint backup is now VERIFIED (1186363152bytes,23758,568
optimizer states,3 RNG payloads). User was asked whether the rebooted node's
SSH tunnel/address has been restored; code snapshot13612688.tar is ready locally.

Current status: STOPPED normally at23758 for the user's planned remote-server
reboot. Complete step023758.pt (1186363152bytes) was CPU-loaded and checked:
all model tensors finite,568 optimizer states (564 at23758,4 event states at8638),
3 RNG payloads, world3/local64/global192, elapsed20075.13s. File fsync completed.
Supervisor/torchrun/new workers3114372/3114380/3114387–9 all exited. Old faulted
kernel-exit contexts still remain for the reboot to clear; no system reboot
was executed by the agent. NAS is an NFS mount on /mnt/ugreen_nas.

Original checkpoint is under the threegpu-r2 run, with an additional hard link
pinned at outputs/reboot-preparation-20261020-r1/step023758.pt (same NAS file,
not an independent storage backup). Checkpoint receipt and prepared strict
three-GPU post-reboot launch script are in that directory. The optional local
1.19GB download is slow and was still IN_PROGRESS at handoff; it must not be
claimed verified. Local backup-status.json records completion/interruption.
The verified/pinned NAS checkpoint is sufficient for this compute-node reboot.

After reboot: reconnect SSH, confirm NAS mounted and GPU health/free memory,
then resume into a fresh output. Prepared resume_three_gpu_after_reboot.py
keeps world3/global192 and ordinary free-GPU checks; no old retained-PID
exception is reused. Returning to four GPUs/bigger batch requires an explicit
new topology transition and capacity check, not ordinary resume. Resume has
not been launched before the reboot.

Historical three-GPU operation follows:

Historical status: RUNNING on three healthy remote GPUs0/1/2, as explicitly
requested by the user who cannot resetGPU3. New run:
pretrain-nine-v6-events-threegpu-20261020-r2, frozen d43b662d, supervisor3114372.
Explicit world4→3 transition from original complete17000 preserved model,
568 optimizer states, elapsed/warmup/selection, RNG ranks0–2 and unchanged
24-step normalizers. Batch changed local128→64/global512→192; LR remains1e-4.
Original80k/cumulative604800s ceiling remains. Lower per-update exposure means
this is not equivalent to the old global512 training schedule.

Single-GPU CUDA checks and3-rank NCCL all-reduce passed.8-update recovery
17001→17008 passed DEV, full saves and3 equal parameter signatures. CPU checkpoint
audit:564 old states at17008,4 event states at1888,3 RNG payloads, strict
world3/global192 binding and exact horizon-normalizer buffers. Long run resumed
that actual checkpoint; verified17120 at2.034updates/s (about0.49s/update),
nine sources seen, no new failure. Recovery allocated peak9.95GiB; long first
window peak10.14GiB. Physical memory about38–40GiB includes old25–27GiB retained
contexts; it is not all new-process memory. Selected GPUs83–100% utilization.
Remaining80k initially estimated10–12h including DEV/saves; step-only ETA8.6h
omits periodic overhead. No event-efficacy/80k reliability claim.

Original GPU3 still reports requires reset and is excluded from CUDA/NCCL and
supervisor health decisions. Guarded sharing verifies only the failed run's
recorded owned worker PIDs, UID/start time, orphan/empty exiting command lines,
pending termination, zero utilization and>=20GiB free capacity before launch;
unknown/live work is rejected. No reset/system operation performed.56 v5/v6
tests passed in17.96s. Frozen original checkpoints and data remain untouched.

Historical four-GPU failure: around00:35 Beijing after last logged17300,
supervisor failed parsing[N/A] and model reported nonfinite valid observation.
The stale old progress.json is not active training. Original17000 retained.

At17000 DEV objective1.281051, handEPE32.784425mm, trajectory score16.892473mm;
prior16129 values1.286467/33.167013/17.007955. Event-on minus unknown handEPE
is−0.051441mm; same-model diagnostic only. Replayed next20 draws17301–17320 on
all4 ranks:10240 CPU samples, checked observed scene/hand/history/dense/features
and goal inputs all finite. Actual failed rank/step is unlogged, so this does
not prove device-memory or every intermediate valid. Hardware fault is a leading
explanation, not an established sole cause of the nonfinite observation error.

Local supervisor now handles missing telemetry as None, records reset faults,
refuses unhealthy/unconfirmed-free GPUs, and requests controlled STOP on reset.
3 targeted regression tests pass; this patch cannot repair a failed GPU and
has not hot-edited the frozen failed run.

The following startup observations are historical:
User requested continuing v6. v5 after-v6-probe-r3 saved/stopped normally at
15723; its complete checkpoint remains available. Strict v6 continuation uses
the learned event model's own readiness checkpoint16136, frozen d37e2320,
not a new v5 migration. Current four-GPU run:
pretrain-nine-v6-events-20261020-r1, supervisor3083547/torchrun3083564.
Verified RUNNING at16520, global512, loss0.77588,1.114 updates/s, peak allocated
18.57GiB; GPUs99–100% and about23–27GiB physical memory. Remaining80k updates
estimated17–19h including scheduled DEV/saves; progress ETA itself omits some
periodic overhead. Neither full completion nor event efficacy is established.

Model architecture and data recipe remain unchanged:768/12heads/12layers,joint21/H4/K24/30Hz,
surface778/object2048,voxel5mm, original TRAIN-only horizon normalization,
AdamW LR1e-4, existing warmup history and optimizer/RNG/elapsed. Restore moved
568 fused AdamW step tensors to parameter devices; strict local128/global512
and world4 binding passed. Original80000/cumulative604800s bounds retained.
Event encoder50304 parameters/dropout15%, no KNN or new tokens.

## Target outlier diagnosis

Actual reader index3150491/sample[29384,0,35], Shadow s4-34534, produces8.92525m
maximum scene-point displacement. Original local rollout and remote compact
payload object/time/hand-root/reset/success arrays match exactly; compact object
translations also match the frozen cache exactly. At frame36 the object starts flying away; maximum speed14.087m/s,
h24 center displacement8.85587m, eventual y17.023m. Clock is regular30Hz, no
reset, rotation orthogonality error1.19e-7, query radius43.65mm. This is an
inherited simulator-runaway segment, not a v6 event/scale/frame-transform bug.
Actual current parent recipe lacks strict motion-admission metadata; historical
cleaning validates finite poses/clocks and observed workspace, not future
runaway motion. It admits ticks3,11,19,27,35. The last3 windows cross onset36.

Read-only audit7a817817 scanned all39526 sequences and actually admitted
TRAIN3218740/DEV545459 windows in67.28s. Object-center displacement>2m occurs
in29 TRAIN/5 DEV windows, all Shadow. A Shadow-only candidate prefix cutoff at
first known observed center speed>5m/s identifies60 TRAIN/10 DEV windows in25
sequences. Weighted TRAIN exposure is0.00275% of draws (about1.4% of global512
batches under independent draws), so rare samples can still cause visible spikes.
Candidate exclusions are prepared but NOT applied; no running recipe/cache was
modified. A clean data continuation needs a separately admitted recipe and
explicit checkpoint data-contract transition. No scratch retrain is implied by
these sparse findings. Large net displacement alone is not a bad human label.
HOT3D s6-00372 has2 additional TRAIN windows crossing known observed center
transitions12.10/12.54m/s; retained as tracking candidates, not classified as a
confirmed simulator defect. The scan covers center translation, not full hand,
angular or surface-point motion quality.

## Prior readiness evidence

The8+1000+8 probe completed15120→16136, checkpoints/recovery/signatures passed.
CPU audit verified564 original optimizer states at16136,4 event states at1016,
4 RNG payloads and exactly unchanged horizon-normalizer buffers. Latest probe
DEV handEPE33.167mm versus unknown-event33.285mm: a small same-model diagnostic,
not a matched-training causal gain. Combined v5/v6 suite48 tests passed;
2 new motion-audit tests cover future-only runaway, rigid frame invariance and
observed-frame clock/masks. No model/training code changed for this switch.

Remote root:/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v6-20261019-r1/.
Historical four-GPU run:outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/.
Switch:outputs/switch-to-v6-20261020-r1/report.json.
Motion scan:outputs/target-motion-audit-20261020-r1/summary.json.
Forensics and candidate exclusions:outputs/target-motion-forensics-20261020-r1/.
Small report/plot copies:outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/ in local repository.

[Current training and diagnosis](experiments/probes/P-20261020-wam-v6-mixed-pretrain.md)
· [completed readiness check](experiments/probes/P-20261019-wam-v6-event-continuation.md).


Current remote three-GPU artifacts:
outputs/three-gpu-continuation-controller-20261020-r1/report.json (transition and checkpoint audit),
outputs/ptv3-shared-wam-v6/three-gpu-recovery-20261020-r1/ (completed8updates),
outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-threegpu-20261020-r2/ (active).
Local small receipts:outputs/ptv3-shared-wam-v6/three-gpu-recovery-20261020-r1/.
