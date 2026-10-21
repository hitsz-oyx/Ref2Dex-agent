---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-local-subset
experiment_id: P-20261021-wam-v6-local-subset
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-v6-local-adaptation
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: use the bounded local recipe for subsequent contact-consistency and rollout interventions
decision_changed_if_negative: resolve training or data mismatch before investing in coupling changes
status: PREPARING
run_id: local-subset-pretrain-20261021-r1
---

# Local GRAB-dominant WAM-v6 adaptation

Decision Probe authorized by the user, including graceful saving/stopping of
the identified Inspire PPO r3 after preparation. Can the audited local subset
support stable, resumable updates and improve held-out teacher-forced tau/contact
metrics? This determines whether to use this local setup for later geometry
consistency interventions. The cheapest discriminating run keeps the existing
architecture/loss and warm-starts step23758; no contact-coupling loss is added.

## Inputs and initialization

1177 independently verified sequences: GRAB1132 and45 sequences from the other
eight sources. TRAIN33672/DEV3356 engineering windows, at most32 per sequence
and split, with disjoint sequence IDs. Preserve original within-source motion
strata; source weights proportional to the admitted TRAIN-window counts, hence
GRAB33272/33672 (~98.81%). This is a fixed subset, not all nine-source pretraining.
The original engineering preparation remains unchanged and non-training-admitted.
The new recipe explicitly keeps engineering_subset=true and38349 unprepared
sequences visible; its dedicated local admission never weakens full-recipe checks.

Initialize all model weights from the local complete step023758.pt, retaining
its four normalizer buffers exactly. Reconstituted normalization JSON records
checkpoint provenance; it is not the unavailable original frozen stats JSON.
No new TRAIN/DEV moments fitted; no future DEV information in normalization.
Reset optimizer, update count, scheduler, RNG and checkpoint selection. This
is weights-only adaptation, not strict continuation of remote training.

## Budget and protocol

Single local RTX3090 GPU4, batch64/global64, seed170/DEV171, bf16/TF32.
Shared/geometry LR1e-5,50-update warmup, original joint21/H4/K24/surface778/
object2048 architecture and losses. Eight initial actual updates, two strict
recovery updates, then resume the resulting checkpoint to at most2000 cumulative
updates or14400 cumulative training seconds (4h). Single GPU leaves5/6 idle
after the user's PPO stop; never stop unrelated users or touch their GPUs.

Measure fixed DEV32 samples per available source before any update, after8,
then every200 updates; microbatch4. Track GRAB separately from the source-macro
diagnostic because other sources are sparse. Preserve objective-best and
trajectory-best, last2 checkpoints, no milestones;8GiB per-run artifact budget.
Combined readiness/checkpoint overhead reserved separately; require20GiB free
throughout. Check GPU use, CUDA memory, finite updates and measured ETA before
leaving the sustained run active. No guarantee that local adaptation cures
free-feedback drift or improves physical grasping.

Outputs: outputs/ptv3-shared-wam-v6/local-subset-recipe-20261021-r1/ and
outputs/ptv3-shared-wam-v6/local-subset-pretrain-20261021-r1/; readiness and
recovery use separate sibling directories. Logs/code archive and switch receipt
belong to the declared output/tmp directories. Environment: local graspenv,
Python3.8.20/Torch2.4.1+cu121; inspect runtime at startup.

## Results

Preparation in progress. No training result claimed yet.
