---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v5-mixed-pretrain
experiment_id: P-20261019-wam-v5-mixed-pretrain
date: 2026-10-19
task: ptv3-shared-wam-v5
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-v5-nine-source-learnability
probe_index_in_family: 1
seed_pool: probe
seeds: [170, 171]
decision_changed_if_positive: use learned joint21 and local surface contact forecasts for subsequent executor adaptation
decision_changed_if_negative: diagnose source and fine-grained forecast failures before extending training or porting the executor
status: PLANNED
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

Result: preparing complete contact labels and four-rank startup checks.

Decision: complete the admitted recipe, verify DDP, then start the explicitly requested sustained four-GPU run.
