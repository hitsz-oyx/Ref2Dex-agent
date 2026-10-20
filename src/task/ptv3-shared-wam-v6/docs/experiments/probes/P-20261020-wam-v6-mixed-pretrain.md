---
schema: ref2dex.probe.v2
probe_id: P-20261020-wam-v6-mixed-pretrain
experiment_id: P-20261020-wam-v6-mixed-pretrain
date: 2026-10-20
task: ptv3-shared-wam-v6
branch: main
git_commit: d43b662d
claim_id: C1
hypothesis_family: HF-v6-event-pretraining
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain event-conditioned checkpoints for downstream execution and a matched efficacy comparison
decision_changed_if_negative: revisit event conditioning or upstream label quality before downstream use
status: STOPPED
run_id: pretrain-nine-v6-events-threegpu-20261020-r2
---

# Continue WAM v6 event-conditioned pretraining

Stopped r2 used three healthy GPUs0–2, local64/global192, frozen d43b662d.
The original four-GPU r1 failed; its protocol/results remain below for provenance.

Classification: Decision Probe, requested by the user after the completed
1016-update readiness check. Determine whether continued event-conditioned
training improves fixed DEV trajectory/contact metrics and remains operational.
This is an exploratory long run, not a causal event-benefit comparison. The
cheapest next information uses the existing complete recipe and latest strict
v6 checkpoint, with scheduled DEV and unknown-event diagnostics; no new model,
data rebuild, or hyperparameter search is required to start.

## Protocol and identity

Original four-GPU r1 resumed the completed readiness run's resume-recovery/step016136.pt, frozen
training code d37e2320, original model weights/optimizer/RNG/elapsed. Preserve
768/12heads/12layers,joint21/H4/K24/30Hz,surface778/object2048,PTv3 voxel5mm,
TRAIN-only24-step normalization and original九源 sampling. Event encoder50304
parameters, dropout15%, same object endpoint goal, no KNN/new tokens. Data:
39526 sequences, TRAIN3218740/DEV545459; DEV8 sources (no DexYCB DEV).

Four remote4090 GPUs0–3, local128/global512,8 spawn workers/rank/120s timeout,
bf16/TF32,AdamW LR1e-4, existing warmup history, seed170/DEV171. Retain original
80000-update and cumulative604800s limits; artifact ceiling20GiB per run,
free space>=20GiB. Keep objective-best and independent trajectory-best.
Remote graspenv Python3.8.20/Torch2.0.1+cu118; original frozen v4 dependencies.

User explicitly changes the earlier restore-v5 choice to continuing v6.
Gracefully STOP only confirmed owned v5 after-v6-probe-r3, preserve its complete
final checkpoint, wait for all GPUs released, then strict-resume v6 into a fresh
directory. No migration from current v5 and no reset of the learned event branch.
If v6 startup fails, attempt restoration of the saved v5 checkpoint. v5 code,
config, frozen data, all previous checkpoints remain preserved.

Remote root: /mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v6-20261019-r1/.
Code: code-d37e2320/. Recipe: data/event-recipe-20261019-r1/.
Run: outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/.
Switch receipt: outputs/switch-to-v6-20261020-r1/.

## Anomalous-target investigation

Independently reproduce TRAIN index3150491/sample[29384,0,35] through the actual
reader, compare frozen object poses and raw upstream poses, validate rigid
transforms/CAD scale, and inspect neighbouring frames. Distinguish inherited
tracking outliers, transform errors, mixed object identity and time/index errors.
This read-only CPU label inspection needs no GPU/model inference. Preserve
frozen recipe while diagnosing; any later correction requires a new admitted
data identity and explicit continuation handling, not a live cache edit.

## Results

Switch completed. v5 normal STOP saved step015723.pt in its own after-v6-probe-r3
output. Strict v6 parent remains readiness/resume-recovery/step016136.pt; all568
optimizer step tensors restored to GPU, effective local128/global512/world4
checks passed. Supervisor3083547/torchrun3083564 launched the unbounded-to80k
production configuration (no recovery or continuation-update override).
Verified16520, loss0.775881,1.114updates/s, allocated peak18.570GiB;4 GPUs
99–100%, physical about23–27GiB. Remaining time initially estimated17–19h
including periodic dual DEV/saves. Code stays frozen d37e2320. Switch helper's
first import-only attempt failed before any STOP and was corrected; all logs
retained, no training updates occurred in that failed helper invocation.

### Reproduction and ranked hypotheses

Actual frozen V6Windows TRAIN index3150491 reproduces scene-target norm8.92525m
and sample[29384,0,35], key s4-34534. A deterministic assertion target<2m fails.
Ranked hypotheses: inherited raw motion outlier (predict identical upstream
positions), coordinate/CAD scale error (predict raw is normal), identity/clock
mixup (predict mapping or timing discontinuity). Inspection supports the first:

- Original file: outputs/3d-data-preparation/unidexgrasp-expansion-20261014-r1/batch077/replay/trajectories/episode-00-env-130.npz.
- Remote compact: ptv3-shared-wam-v3-nine-source-20261015-r1/data/native/payloads/38103.npz.
- Raw local and compact object, time, hand_root, reset and success arrays
  compare exactly; compact object translations equal the frozen cache exactly.
  Frozen matrix rotation conversion differs
  at most2.97e-8 from upstream float64; orthogonality error1.19e-7.
- Actual regular observed30Hz, no reset within the file, one fixed object;
  CAD query radius43.65mm. At frame36 first center speed>5m/s, maximum14.087m/s;
  object center35→59 moves8.85587m and eventually reaches y17.023m.
- The actual base recipe-auditbound-20261018-r1 has no quality_admission entry.
  Its cleaning covers pose/clock validity and observed workspace; it does not
  impose a future simulator-motion gate. Native compact decoder marks finite
  simulator poses known; v5 inherits them unchanged. Admitted ticks3/11/19/27/35
  include3 windows whose future overlaps runaway onset36.

This identifies an inherited simulator-runaway admission gap. It does not prove
the underlying PhysX cause or exactly attribute the recorded loss26.098 to a
specific model term (no extra model replay was required for the data diagnosis).
Neither v6 event encoding nor21-point reconstruction introduced the object jump.

### Full actual-window scan and prepared candidate

Read-only tools/audit/audit_target_motion.py, commit7a817817, scanned all39526
sequences and all3218740 TRAIN/545459 DEV windows in67.28s,4 CPU I/O workers.
No model inference or GPU contention. Clock uses observed flags on the30Hz grid,
so HOI4D observed transitions use actual2-tick intervals. Counts below are
object-center motion only, not every surface point or hand joint.

| Diagnostic | Shadow TRAIN | Shadow DEV |
| --- | ---: | ---: |
| Maximum future center displacement>2m | 29 | 5 |
| Maximum future center displacement>5m | 4 | 0 |
| Overlaps known center transition>5m/s | 42 | 6 |
| Tail cutoff after first center transition>5m/s | 60 | 10 |

No other source has center displacement>2m. HOT3D s6-00372 has2 TRAIN windows
crossing two known observed transitions12.10/12.54m/s, frames2849/2850; these
are separate tracking candidates, not a confirmed simulator defect. Moderate
human long-range displacements alone do not establish label errors.

Candidate exclusions cover25 Shadow sequences. Uniform native source sampling
and source4 probability0.120625887 imply60/263106 within-source windows and
2.7508e-5 overall TRAIN draw probability (0.00275%); global512 draws imply about
1.4% of batches contain at least one candidate. Thus sparse samples can produce
visible loss spikes. No source-success filter was used.

Remote outputs/target-motion-forensics-20261020-r1/candidate-exclusions.npz and
.json contain concrete index/sample identities with training_allowed=false.
They are NOT applied. Frozen stats/recipe/caches remain unchanged. Next cleanup
would admit a new recipe and explicitly bind a data-contract continuation;
there is no evidence that all learned weights must be discarded. This audit
is not complete hand/angular/surface-motion certification.

Reports: outputs/switch-to-v6-20261020-r1/report.json,
outputs/target-motion-audit-20261020-r1/{summary.json,sequences.jsonl},
outputs/target-motion-forensics-20261020-r1/{focal-sequence.json,candidate-exclusions.json,hot3d-candidate.json}.
Small local copies/plot live in outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/.
Two diagnostic regression tests pass: future-only runaway caught with clean
prefix preserved; rigid coordinate invariance; observed clock and masks.
At the failure audit the proposed next step was GPU3 repair; the user subsequently authorized three-GPU recovery below.

Bounded readiness evidence is in
[P-20261019-wam-v6-event-continuation](P-20261019-wam-v6-event-continuation.md).
No matched training arm, independent replication, robot success or event net
benefit is claimed. Event annotations are proximity teachers, not force labels.


## 2026-10-20 failure and recovery boundary

At about00:35 Beijing (2026-10-19 16:35UTC), the supervisor recorded FAILED:
ValueError converting utilization [N/A] to int. Training's final log reports
nonfinite valid observation at clean(scene_points,point_valid); last regular
log17300, loss0.771962. Latest complete checkpoint17000 is retained; uncheckpointed
updates after it cannot be assumed recoverable. No result.json was published.

At09:36 Beijing, nvidia-smi explicitly reported GPU3 requires reset and GPU
Recovery Action Reset. All GPU utilizations were idle or unavailable despite
stale allocated contexts. Old supervisor/torchrun exited; recorded worker PIDs
remain in exiting/zombie states. No unknown tasks killed, no system-level reset
attempted. Administrator recovery of the node/GPU is required before restoring
four-GPU training. Running status/ETA in stale progress.json is not live evidence.

DEV17000: source-macro objective1.2810514953; handEPE32.784425mm;
trajectory score16.892473mm (short6.228554/root-relative17.506836/contact26.942029).
Compared with16129, modest continued improvement. Conditioned minus unknown
handEPE−0.051441mm; no matched training efficacy claim.

CPU replay of actual next20 TRAIN draws17301–17320,4 ranks,10240 samples:
scene_points,xyz,start_hand/history,dense_hand/history,features,goal_object_points
all finite. This narrows the data-input hypothesis but cannot identify actual
failed rank/step or certify CUDA memory/intermediates. The GPU reset and model
error may be related; sole causality is not established. The earlier inherited
Shadow extreme finite targets are a separate recorded issue.

Supervisor reproduction with mocked actual [N/A] telemetry failed2 tests before
fix. Fix records unavailable numbers as None, recognizes explicit reset status,
rejects unhealthy/unconfirmed-free launch, and issues controlled STOP on reset
while retaining fault reason. Unknown utilization alone does not trigger reset.
3 targeted tests pass. The frozen failed d37e2320 snapshot is not modified.
No automatic4→3GPU/world change or scratch restart was made.

Remote CPU/hardware receipts: outputs/gpu-fault-audit-20261020-r1/.
Recovery parent: outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/step017000.pt.
At that audit the original world4 run remained stopped; subsequent user-authorized world3 recovery is below.


## Authorized three-GPU recovery protocol

The user cannot reset the node and explicitly requests three-GPU continuation.
GPU0/1/2 passed isolated CUDA arithmetic and real3-rank NCCL all-reduce while
GPU3 remains excluded. Exiting old owned contexts retain about25–27GiB on each
card; available CUDA memory about20.6–21.9GiB. No system reset is required for
these successful checks. A bounded sharing exception verifies only the failed
run's recorded worker PIDs, owner UID, original process-start time, empty exiting
command lines, orphan parent and pending termination. Reject unknown/live work,
selected reset GPUs, non-idle devices or less than20GiB NVML free capacity.
This is not permission to share arbitrary allocated GPUs.

Use separate explicit transition_three_gpu.py from original complete step17000:
world4→3, local128→64, global512→192. Preserve all model/normalizer buffers,
568 optimizer states/moments/counters, elapsed/warmup/selection and RNG ranks0–2;
omit only rank3 RNG and record that future draws/gradient noise change. Ordinary
resume still requires exact config/data/world binding. LR remains1e-4. Same
80000-update/cumulative604800s bounds and frozen event recipe remain in force.
No anomaly-exclusion candidate is silently applied. This protocol sees fewer
examples per update than the originalglobal512 run; it is not equivalent
training exposure or bitwise replay. Planned run_id:
pretrain-nine-v6-events-threegpu-20261020-r2.

Engineering acceptance:8 real three-rank recovery updates, finite gradients,
nonzero event gradient, real DEV/checkpoints, three identical signatures;
strict re-open of the resulting three-rank checkpoint before continuing the
long run. Per-phase fresh outputs, original checkpoints untouched. Observe
new-process allocated memory separately from old retained-context memory;
only selectedGPU0–2 health controls the new supervisor. Fixed GPU3 status must
not abort otherwise healthy selected devices.


Three-GPU recovery8 passed:17001→17008, three identical parameter signatures,
full DEV and complete checkpoints. Peak allocated9.9484GiB; GPU forward+loss
136.619ms/backward311.065ms/optimizer29.056ms (sum476.74ms). Full loop includes
startup/dual DEV/save and must not be equated with steady update time. Final
checkpoint CPU audit verified original564 optimizer states at17008, event4 at1888,
3 RNG payloads, exact old horizon-normalization buffers and retained topology
provenance. Combined56 v5/v6 tests passed in17.96s; scoped code/config checks pass.

Remote controller outputs/three-gpu-continuation-controller-20261020-r1/ includes
transition/, recovery-launch.log, long-launch.log, progress.json/report.json.
Short recovery outputs/ptv3-shared-wam-v6/three-gpu-recovery-20261020-r1/.
Long run outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-threegpu-20261020-r2/.
All under the existing remote v6 root. Local small receipts:
outputs/ptv3-shared-wam-v6/three-gpu-recovery-20261020-r1/.


Three-GPU long continuation confirmed RUNNING at17120, about2.034updates/s,
all9 source IDs present; global192/world3/LR1e-4. Supervisor3114372. First-window
allocated peak10.14GiB, physical about38–40GiB/card including retained old
contexts; selected utilization83–100%. Initial remaining80k ETA10–12h including
scheduled DEV and saves (step-only estimate8.6h). Original frozen data and sparse
exclusion candidates remain unchanged. Actual checkpoint/provenance audit is
retained in controller report; recovery transitioned forward through17008 and
its updates are consumed by the long run. GPU3 is still faulted but excluded.
This is a changed batch/world training regime and not equal sample exposure
or a bitwise continuation of world4. Keep fixed DEV reporting and original
80000/cumulative604800s bounds.
Historical decision: continued three-GPU r2 under the guarded capacity contract.


### Owned residual-worker escalation after user retry request

Verified every old residual TID, including3083624 in zombie-leader3083586.
All residual live threads have PF_EXITING set and empty process command lines.
Targeted tgkill(SIGCONT) then tgkill(SIGKILL) returned success for each verified
old live thread. GPU allocations did not change. Across about4.24s, each of
four residual live threads accumulated about4.24s kernel CPU time and0 user
CPU time. This supports a stuck kernel exit-cleanup path; specific driver stack
cannot be confirmed because /proc/PID/stack and syscall reads are denied.
The old contexts remain in NVML while read-only fuser lists only new training
workers, so broad fuser -k would hit active training rather than selectively
release the residual allocations. No MPS service was running. UID1006 has zero
effective/permitted capabilities; no system reset or driver operation attempted.

Receipt: remote outputs/gpu-fault-audit-20261020-r1/thread-wake-kill-and-cpu-audit.json;
small copy under the matching local output directory. New three-GPU training
was untouched. For administrator recovery prerequisites see
[NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).


## Controlled stop before user-planned server reboot

User plans a reboot; controlled STOP applied only to confirmed active owned r2.
Normal STOPPED at23758, final step023758.pt published and CPU-loaded, all model
values finite,568 optimizer states and3 RNG payloads retained. State counters:
564 original23758 and4 event8638. Effective world3/local64/global192 and
elapsed20075.1297s retained. Checkpoint1186363152bytes was fsynced. New training
supervisor/torchrun/all3 workers exited; original hung contexts remain for the
node restart. No reboot/reset/system operation executed by the agent.

NAS outputs/reboot-preparation-20261020-r1/ holds checkpoint-report.json,
additional hard-link pin step023758.pt (not a separate storage copy), and
resume_three_gpu_after_reboot.py. The prepared script checks NFS inputs and
requires a fresh after-reboot-r3 output; same world3/global192/code d43b662d,
no retained-context exception. It has syntax validation and is NOT launched.
A return to world4/batch128 needs explicit new topology/RNG/config handling
and healthy-device/memory verification after boot. Existing strict resume is
not weakened. Original checkpoints/data/code remain on NAS.

Optional full local checkpoint copy is slow and was still in progress at this
handoff. Local output directory outputs/ptv3-shared-wam-v6/reboot-preparation-20261020-r1/
has bounded backup observer and backup-status.json; do not treat a partial
file as a completed backup. The verified NAS file is the recovery source and
is sufficient to reboot this compute node. No need to wait for optional local
transfer. SSH forwarding and NAS mount must be available after boot.
Decision: reboot preparation complete; resume after node/GPU/NAS checks.
