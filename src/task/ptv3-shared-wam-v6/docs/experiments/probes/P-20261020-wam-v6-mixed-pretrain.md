---
schema: ref2dex.probe.v2
probe_id: P-20261020-wam-v6-mixed-pretrain
experiment_id: P-20261020-wam-v6-mixed-pretrain
date: 2026-10-20
task: ptv3-shared-wam-v6
branch: main
git_commit: d37e2320
claim_id: C1
hypothesis_family: HF-v6-event-pretraining
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain event-conditioned checkpoints for downstream execution and a matched efficacy comparison
decision_changed_if_negative: revisit event conditioning or upstream label quality before downstream use
status: RUNNING
run_id: pretrain-nine-v6-events-20261020-r1
---

# Continue four-GPU WAM v6 event-conditioned pretraining

Classification: Decision Probe, requested by the user after the completed
1016-update readiness check. Determine whether continued event-conditioned
training improves fixed DEV trajectory/contact metrics and remains operational.
This is an exploratory long run, not a causal event-benefit comparison. The
cheapest next information uses the existing complete recipe and latest strict
v6 checkpoint, with scheduled DEV and unknown-event diagnostics; no new model,
data rebuild, or hyperparameter search is required to start.

## Protocol and identity

Resume the completed readiness run's resume-recovery/step016136.pt, frozen
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

Preparation and controlled switch pending. Bounded readiness evidence is in
[P-20261019-wam-v6-event-continuation](P-20261019-wam-v6-event-continuation.md).
No matched training arm, independent replication, robot success or event net
benefit is claimed. Event annotations are proximity teachers, not force labels.
