---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v5-mixed-pretrain
experiment_id: P-20261019-wam-v5-mixed-pretrain
date: 2026-10-19
task: ptv3-shared-wam-v5
branch: main
git_commit: 395ab290
claim_id: C1
hypothesis_family: HF-v5-nine-source-learnability
probe_index_in_family: 1
seed_pool: probe
seeds: [170, 171]
decision_changed_if_positive: use learned joint21 and local surface contact forecasts for subsequent executor adaptation
decision_changed_if_negative: diagnose source and fine-grained forecast failures before extending training or porting the executor
status: RUNNING
run_id: pretrain-nine-v5-20261019-r1
---

# Four-GPU nine-source WAM-v5 pretraining

Classification: Decision. The user explicitly requested stopping the v4 scale
run and starting v5 on four GPUs, with hyperparameters based on v4. Test whether
the repaired joint21/dense-surface/local-C representation learns source-specific
motion and proximity structure at practical mixed-training cost. A full-data
run is required for this learning question; the earlier8-step engineering
snapshot cannot answer it. No PPO or final manipulation improvement is claimed.

## Inputs and prior-run transition

Remote root:
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/`.
Use all39526 repaired/inherited hand sequences and2747 CAD assets. Contact
completion is a Blocker: do not silently train the current arrival-biased subset.
Replace the stopped single-worker preparation with16 bounded fork workers
calling the identical `prepare_surface_contact.prepare` function. New labels
must retain the same cache contract and upstream geometry identities; no KNN
or contact recomputation enters the model/reader. Completion updates the same
cache's progress, preserving original STOP and serial artifacts for provenance.
Parallel completion is bounded7200s, NAS free>=20GiB, no other writer active.

The scale1024 supervisor2975536/torchrun2975737 was explicitly stopped through
its own STOP file. Preserve full `step39000.pt`, latest/best and historical
outputs. The old launcher reports requested STOP as FAILED; the transition
receipt records user intent, not a newly diagnosed model failure. No automatic
restart of the old scale job is intended after this request.

Freeze an independently admitted, complete v5 recipe with the shared H4
predicate, original TRAIN/DEV identities, source probabilities and inherited
main3 motion strata0.6/0.2/0.2 (native uniform). Recompute TRAIN-only24×3
hand/flow moments using128 sampled windows/source. No TEST data. Refuse pending
sequences or an engineering subset at the sustained training entry point.

## Initial training protocol

- One synchronous DDP model on remote GPUs0/1/2/3, local batch128/global512,
  eight loader workers/rank, two intraop threads/rank, bounded reader caches.
- Retain v4's1024 shared width/12 layers/16 heads,128 point width, bf16/TF32,
  activation checkpointing, fused AdamW, both parameter groups LR1e-4,
  weight decay.01, clip1, linear500-update warmup,80000-update limit.
- v5 differences: joint21,778 observed hand samples,2048 object/contact
  queries,5mm geometry grid,16-part C and same-ID local PTv3 readout.
  Keep masked ordinary BCE, positive-neighborhood CE and motion-weighted flow.
  Normal-aware candidate and BCE balancing stay disabled pending evidence.
- Start v5 from scratch; v4 weights/optimizer are not implicitly migrated.
  Exploratory model/draw seed170 and DEV panel seed171 are reused, not an
  independent replication. Data/labels/architecture differ from v4, so this
  is not an isolated causal comparison of one change.
- Fixed source-macro DEV panel256 windows/source, microbatch8; first validation
  at20 then every1000. Save first, every1000, best,40000/80000 milestones and
  the final/controlled-stop update. Keep recent2 plus protected selections.
  No tuning on TEST. Fine-grained metrics remain diagnostics, not extra losses.
- Report globally reduced loss components, draws, throughput/ETA, per-device
  GPU utilization/memory and owned checkpoint state. Verify four-rank startup
  and synchronized parameters on a throwaway real-data engineering smoke first.

## Execution bounds and validation

Environment: remote graspenv Python3.8.20/Torch2.0.1+cu118, RTX4090 cards
reporting49140MiB each. Use frozen code/config snapshots with recorded commits.
Sustained training<=604800s including checkpointed prior elapsed time,
outputs<=20GiB, NAS free>=20GiB; STOP requests final recoverable checkpoint.
The supervisor owns only its isolated torchrun group and does not kill unknown
processes. Preparation and metadata audits are CPU jobs; existing local Inspire
training remains outside this run. Actual throughput/ETA are reported only
after startup, including the limits of small measurement windows.

## Startup evidence

Code395ab290 passed22 local tests, including actual optimizer/DEV/checkpoint
loops on tiny geometry: a STOP after update3 saves that non-periodic update;
resuming into a fresh directory produces bitwise-equal model tensors to an
uninterrupted4-update run, preserving DEV selection and protected snapshots.
This is a CPU engineering check, not bitwise four-GPU resume certification.

Four real4090 ranks completed8 throwaway updates at local128/global512,
8 workers/rank on the repaired9-sequence engineering recipe. All nine sources
appeared. The two descriptive parameter signatures (sum and sum of squares)
were exactly equal across all four ranks after optimization:
`[32779.318755598084,105661.06832150088]`. This checks synchronization, not a
cryptographic tensor identity claim. Rank0 GPU timing315.75ms forward/loss,
781.66ms backward,40.15ms optimizer; peak19.79GiB, loop19.86s including startup.
No checkpoint was written or used to initialize sustained training. The parent
experiment remains a learning Probe, while this startup check is engineering.
Remote evidence: `outputs/ddp-startup-20261019-r1/`.

## Sustained run startup result

All39526 hand/contact sequences completed, with16250070 valid object frames;
16-worker contact completion took879.79s without changing the teacher. The
replacement admitted finalizer completed with pending0, training_allowed=true
and engineering_subset=false. The frozen recipe retains3218740 TRAIN and
545459 DEV windows. Joint21 hand/scene24×3 normalization uses1152 TRAIN
moment windows (128/source); minimum hand standard deviation is7.058mm.
Recipe: `outputs/finalize-training-20261019-r1/recipe/manifest.json`.

The sustained run is RUNNING under supervisor3028892/torchrun3029034, frozen
code395ab290. At the startup report it had reached update280, and all nine
training sources had appeared. Complete `step000001.pt` and `step000020.pt`
are each about1.993GB; the first fixed DEV evaluation completed, source-macro
objective5.1234348847, with `best.pt` published. This is startup evidence, not
a learning-quality or v4 improvement conclusion. The inherited DEV split has
eight sources: DexYCB/source8 has no DEV windows. Its TRAIN draws are present;
the macro score therefore does not assess DexYCB generalization. No split was
changed to manufacture a ninth DEV source. A follow-up liveness read reached
update440 with RUNNING status and finite loss3.22755.

Updates220–280 took1.15426s/update. All four GPUs were100% utilized; reported
memory24571–30099MiB/card. Pure remaining update time is25.56h; allowing
periodic DEV/checkpoint overhead, initial80k estimate is27–30h. The short
window does not establish sustained throughput over the whole data mix.
The progress-file aggregate ETA still includes cold startup and first DEV,
so it is not the same estimator as this recent-window estimate.

Remote run: `outputs/ptv3-shared-wam-v5/pretrain-nine-v5-20261019-r1/`.
Remote receipt: `outputs/pretrain-nine-v5-20261019-r1-startup-report.json`.
Local evidence: `outputs/ptv3-shared-wam-v5/pretrain-nine-v5-20261019-r1/`
contains `startup-report.json` and `ddp-smoke-result.json` (not tracked).

Result: full-data four-GPU mixed pretraining has started, with real updates,
published recoverable checkpoints and a completed first DEV evaluation.

Decision: continue the authorized80k bounded run and use later fixed DEV
panels to assess learning; keep the teacher/loss and v4 scale job unchanged
in their current states (v4 stopped, v5 normal-aware/balanced variants disabled).
