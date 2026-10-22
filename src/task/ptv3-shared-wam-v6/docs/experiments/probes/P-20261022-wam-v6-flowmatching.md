---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-flowmatching
experiment_id: P-20261022-wam-v6-flowmatching
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: 0917f47ebb74173a1f43b10f3e3a21c5689ae0b6
claim_id: C1
hypothesis_family: HF-v6-joint-trajectory-flow
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171,172]
decision_changed_if_positive: retain residual conditional flow if fixed held-out short-prefix errors improve stably
decision_changed_if_negative: retain checkpoints and inspect generalization before scaling the flow branch
status: STOPPED
run_id: flowmatching-overnight-20261022-r2
---

# Joint hand/object conditional residual flow with GT-object contact

Result: 128-step trial/sampling/recovery PASS; teacher field improves11.7% versus current zero field, DEV prefix not improved; longer DDP training started.
Decision: proceed with owner-requested longer training because the flow is technically trainable; no held-out gain established at128.

## Question and minimal decision probe

Does a joint hand/object conditional vector field learn useful trajectory residuals
around healthy WAM-v6 predictions, retaining the latest GT-object contact loss?
This Decision Probe first tests real-input learning, noise-only ODE sampling and
strict distributed recovery; failure blocks the long run. Success permits the
requested longer exploration, not a claim that FM solves rolling drift.
Compare fixed DEV prefix1–8 hand/object errors, horizon24 object error, contact
and feature-stability diagnostics with initialization.

## Method and retained objective

Healthy no-geometry v6 source:
`outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt`.
Copy all old weights and four normalizer buffers exactly; new flow parameters,
Adam/RNG/counters start fresh. Keep goal/next-event conditioning,24 future ticks,
joint21 per hand,32 object points and original coordinate/C/part supervision.
This expands the architecture, not strict resume of GT-contact3238; later
trial-to-long transitions restore optimizer/RNG/config strictly.

One future-tick token packs both hands and object (222 coordinates). New width256,
two-layer temporal Transformer decoder attends to observed shared context.
Linear training path `x_t=(1-t)*noise+t*(GT_normalized-base.detach())`, velocity
`GT_normalized-base.detach()-noise`, residual noise sigma0.1. Generate actual
coordinates with four differentiable Euler steps from noise, add to pretrained
base. Coordinate/contact losses see this target-free ODE sample, never the
teacher interpolation. Backbone encoded once. Fixed sample-ID/seed172 DEV noise;
alternate seed/eight-step audits. Future GT labels remain inference-inaccessible.

Strict FP32 PTv3/attention/flow, no AMP/TF32. Shared/geometry LR1e-5, new flow
LR1e-4, warmup50, clip1. Keep normalized tau and motion-weighted scene MSE,
learned C BCE/part CE; add masked hand/scene velocity MSE weight1. This hybrid
retains pointwise MSE and cannot establish multimodality by itself. Learned C
head is still observed-feature conditioned, not resampled-trajectory conditioned.
TRAIN total includes velocity terms; DEV total excludes teacher-only terms.

Contact unchanged: GT future ordered2048 object points plus predicted-tau hand
skin, proximity probability vs GT labels; ticks1/4/8/12/16/20/24,chunk64,
weight0.05/ramp100,clearance0.25. GT object is loss-only. This term directly
constrains generated hand, not predicted object; scene/flow losses supervise it.

## Inputs, resources and execution plan

Recipe `outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json`
and sibling stats:207579 TRAIN/3356 DEV windows,1177 caches,~99.8% GRAB draws.
Longer training does not recover the full remote nine-source data. Matched parent
seeds170/171, flow172, not independent replication. Runtime
`/home2/wyy/miniconda3/envs/graspenv/bin/python`,Python3.8/Torch2.4.1+cu121.

Only physical RTX3090 GPU4/5/6. Capacity8 onGPU4 at local96; fallback80/64 only
for OOM or reserved>21.5GiB. DDP8/save+strict2/recovery, continuation to128,
then real ODE/teacher-field audit. Automatically continue same optimizer/RNG to
30000 updates or43200s cumulative new-objective training time, whichever first.
DEV0/20/128/every500, checkpoint every500, two recent plus selected within8GiB
per stage. Actual throughput sets final steps.08:00 is an interim review, not
stop time; likely finish ~12h after main stages begin. Absolute supervisor wall
bound47000s,stall20min,disk20GiB floor. Free~175GiB,projected extra<24GiB.
No deletion/migration required. Owner exception recorded in root CAMPAIGN.

Supervisor and children use separate sessions, stdin closed, nohup launch.
Environment allowlist; no credential reads/writes. Frozen tracked code;
third_party symlink read-only. Exact launch/PID/SID/telemetry to be appended.
STOP file requests saving/stopping only this run; no other jobs/cards touched.

## Sources and limitations

Linear conditional flow formulation: primary [Flow Matching paper](https://arxiv.org/abs/2210.02747).
Old `src/task/PointWorldWAM/train_wam_v1.py` has a linear velocity-target example;
its action/MANO contracts are not transplanted. Prior
`P-20261015-mse-multimodality-audit` did not train FM or establish multimodality
as this model's failure cause. No universal FM superiority or drift-fix claim.

Implementation validation:47 v6 tests PASS in80.70s, including target isolation, fixed sampler RNG, differentiable Euler and old-weight-preserving expansion.

## Initial admission correction

Run r1 supervisor1924337 detached successfully (PPID1,SID/PGID1924337), but
preflight stopped before training: audit differentiated physical decoded scene,
while retained scene MSE uses normalized scene and GT-object geometry has no
direct predicted-scene gradient. Correct audit to use normalized hand/scene
for total-loss gradients and physical outputs only for geometry gradient
contracts. This is an audit correction, not a change in the training objective.
Preserve r1 logs; subsequent committed/frozen retry uses run_id r2.

Repository-wide verify remains blocked by pre-existing missing v2 metadata in
six trajectory-policy experiment cards; no new-card error was reported. Those
unrelated cards are not modified. New v6 suite47 PASS.

## Detached launch receipt

Frozen code: `0917f47ebb74173a1f43b10f3e3a21c5689ae0b6`. SupervisorPID `1926227`; launcher receipt
`tmp/flowmatching-overnight-20261022-r2/launch.json`. Full child stage commands
are persisted in output `process.json`. Exact top-level invocation below; launched
with stdin=DEVNULL/start_new_session=True/close_fds=True and environment allowlist.

```sh
CUDA_VISIBLE_DEVICES=4,5,6 nohup /home2/wyy/miniconda3/envs/graspenv/bin/python -u /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/flowmatching-overnight-20261022-r2/frozen/src/task/ptv3-shared-wam-v6/tools/run/launch_flow_overnight.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt --baseline-config /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/gt-object-contact-20261021-r1/config-b96.json --recipe /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2 --run-root /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent --git-commit 0917f47ebb74173a1f43b10f3e3a21c5689ae0b6
```

Logs: `tmp/flowmatching-overnight-20261022-r2/supervisor.log`; stage logs,
config, checkpoints, DEV and telemetry under
`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/`.
Corrected real-input preflight on four TRAIN examples PASS:617 parameter tensors,
all strictFP32, total flow/readout gradients finite/nonzero, geometry hand
gradient0.8163 and predicted-scene gradient absent as intended. Initial zero
field teacher/zero ratio1.0, fixed-sample repeat exact, alternate seed differs.

## Admission at01:22 +08

R2 supervisor1926227 PPID1/SID/PGID1926227 confirmed. Real preflight PASS.
Capacity8 local96:peak allocated15.62/reserved17.58GiB,0.365updates/s singleGPU.
DDP8/save then strict2 recovery through10 PASS:617 finite Adam states/counts
match,3 RNG,world3/global288,four exact normalizers and equal three-rank
parameter signatures. Checkpoint size1,217,083,169bytes. Trial128 nowRUNNING;
step13 loss0.9350/norm0.8504,three GPUs100%,memory~18.6–20.1GiB/free156GiB.
No prediction-quality improvement claim yet.

## Trial128 result and decision,01:31 +08

DDP trial COMPLETED128.617 finite optimizer states all128,3 RNG/global288,
normalizers exact and three equal parameter signatures; saved checkpoint
`flow-trial/step000128.pt` (1,217,116,001bytes,cumulative606.64s). Real four-TRAIN
sampling/gradient audit PASS: fixed draws repeat bitwise, alternative seed gives
RMS0.14795 normalized variation; Euler4vs8 RMS0.005593. Teacher velocity MSE
0.049497 vs current zero-field0.056058,ratio0.88296 (11.7% lower). This comparison
uses the checkpoint's current base, not a fixed original residual target. No
multimodal-coverage or rollout success is established. All attention FP32
(serialized16,dense17),AMP/TF32 off; geometry hand gradient finite/nonzero and
direct predicted-scene geometry gradient still absent.

Fixed32 GRAB DEV, fixed seed172; physical errors inmm:

|metric|FM init0|FM128|
|---|---:|---:|
|all-horizon hand EPE|55.222|56.571|
|hand prefix1–8 mean|16.908|17.151|
|object prefix1–8 mean|17.800|18.699|
|object horizon24|21.881|21.071|
|GT-object geometric contact MAE|0.146995|0.149882|

No held-out prefix/contact improvement. The sampler initially adds noise to
healthy predictions, so compare this arm against its own fixed-noise init and
retain previous deterministic baselines for later comparison. This short probe
passes technical viability, not efficacy. Per explicit owner instruction continue
the same optimizer/RNG/config for longer training, without treating a small
TRAIN field gain as evidence that FM resolves the generalization gap.

## Actual longer-training command

Torchrun PID `1947491`, separate session from supervisor and TUI:

```sh
CUDA_VISIBLE_DEVICES=4,5,6 /home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/flowmatching-overnight-20261022-r2/frozen/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py --config /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/config-b96.json --manifest /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --stats /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training --workers 8 --git-commit 0917f47ebb74173a1f43b10f3e3a21c5689ae0b6 --resume /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/flow-trial/step000128.pt
```

Supervised child receives closed stdin, redirected stdout/stderr to
`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training.log`,
`start_new_session=True`. Output/checkpoints/DEV:
`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/`.
`process.json` includes the exact stage commands and updated ETA/step forecast;
`gpu.jsonl` logs only authorized4/5/6 telemetry every10s.

Trial throughput0.2641updates/s. Bound30000updates or43200s cumulative, likely
time-limited to~12000–14000updates; estimated end2026-10-22~13:20–13:30+08, with
~6000–7500 updates by08:00 if steady. These are provisional, validation/checkpoint
pauses and host contention affect them. No new data is being downloaded or
migrated, no other jobs stopped, no cleanup and no credentials read/written.

## Sustained startup acceptance,01:33 +08

Main stage reached160 (32 updates after strict resume128).loss0.86393,
pre-clip gradient1.2235,shared/context RMS3.773/5.842,finite; geometry weight0.05.
Throughput0.3100updates/s/global288 (~89examples/s aggregate). GPU4/5/6
utilization89/93/98%,memory20925/20121/20707MiB; peak tensor allocation16.09GiB.
Actual free space153.79GiB (>20GiB floor). Runtime confirms the actual FP32
three-rank training, not merely a queued launch. Supervisor1926227 PPID1/SID
1926227; main torchrun1947491 SID/PGID1947491. Closing TUI does not own either
process session. Resource telemetry remains limited to GPUs4/5/6.

Current forecast~13326 final updates/2026-10-22T13:20:55+08 (12h cumulative
bound),allow~12000–14000 and13:20–13:30 for validation/checkpoint pauses.
08:00 forecast~7000–7500updates,not a completion claim.30k is the step ceiling.
Use `process.json.estimated_end_unix/expected_final_step` for bounded ETA; the
legacy trainer `progress.json.eta_seconds` counts toward30k only and excludes
the12h stop condition. Next fixed DEV/checkpoint at500; then every500.
`startup-health.json` captures admission,128 audits,main health and ETA.
No rollout re-evaluation or statistical performance conclusion has been made.

Operator stop: create root output `STOP`; supervisor asks only its active stage
to save and exit. For normal completion it audits final model/optimizer/RNG
and exact normalization, then records final status/checkpoint in `process.json`.
No existing source checkpoints or user files were modified.

## 2026-10-22 owner SIGINT stop：权重与曲线独立审计

机主通过Hermes在12:29发SIGINT停止本轮；训练日志独立记录12:29:11.355672收到SIGINT，
三个rank KeyboardInterrupt，torchrun退出1。此卡标STOPPED表示owner已停止探索；原
`process.json`仍为FAILED、`training/progress.json`仍残留RUNNING，均保持原样，
不能假称正常final-save退出。信号发送者身份来自owner转达，不是训练日志可认证的内容。

CPU审计observed12:37:46+08：原supervisor/torchrun/三个main workers等11个已知PID均不存在，
无该run的目标训练进程；GPU4/5/6各6MiB、0%，无对应compute app。未信号/重启其他作业。

**主阶段确有权重，最新为11000，不是旧trial128。** 实际payload逐一CPU读取：
`training/step010500.pt`于11:31:29、`training/step011000.pt`于12:04:02写成，均1,217,116,001字节；
11000的schema v6、model626项/flow49项、optimizer617项、world3/RNG3。
3个真实CPU sentinel均finite且不同于trial128（flow output weight RMS .002313→.018866），
但本轮没有做全optimizer数值或恢复审计。`best.pt`指trial20、`trajectory-best.pt`指strict9，
都是继承的selection，不能拿软链mtime或best别名充作latest权重。

owner观察点11020有日志、无已发现权重；**最后日志实际到11380**（12:28:49，
累计11.132684h、loss .66100058）。最新保留11000到最后确认更新至少差380updates，
即global288下109440次样本曝光（不是环境交互或独立数据量）、1488.15s累计elapsed。
raw SIGINT绕过冻结trainer第366行正常loop后的final-save/result路径，故没有11380 final权重。
周期保存500/保留最近2份正常，不能据此说“主阶段从未保存”；后续信号安全保存修复只作建议，
本轮没有修改训练源码或启动训练。

曲线原观察可复核：step9000累计8.711117h/loss .68901015 → step11000累计10.713360h/
loss .66127519；这两个batch的contact .243203→.252238、geometry_contact .289527→.294095。
固定±15min窗口总loss均值 .684049→.656991，而contact .251234→.249398（约降.73%）、
geometry_contact .300649→.293192（约降2.48%）；后期接触改善弱，不是严格完全flat。
固定32-window GRAB DEV的geometry contact MAE在9000/11000为.148111/.148999，
object prefix8 EPE19.441/20.013mm；未显示持续DEV改善，不能当作成功率或架构因果结论。

完整八个原run普通权重、六个symlink、诊断保留8500 copy的payload/真实mtime、扫描边界、
停止receipt、566条主训练日志/22次DEV曲线与保存路径定位见
[停止与权重审计](../../audit/flowmatching-stop-checkpoint-audit-20261022.md)。
产物 `outputs/ptv3-shared-wam-v6/flowmatching-stop-audit-20261022-r2/`：
`stop-receipt.json`、`checkpoint-inventory.json`、`summary.json`、`train-curve.csv`、
`dev-curve.csv`、`training-curves.png/svg`。关联扫描覆盖本项目outputs2396个权重路径，
没有发现11020 payload；未知位置/任意改名的外部export仍不能排除，不宣称全世界不存在。
本次只CPU审计、不建CUDA context、不改原权重/训练manifest，不写STATE、不做Git操作。
