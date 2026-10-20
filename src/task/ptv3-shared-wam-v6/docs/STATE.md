# WAM v6 state — 2026-10-21

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
