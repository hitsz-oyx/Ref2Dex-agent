# WAM v6 state — 2026-10-21

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
