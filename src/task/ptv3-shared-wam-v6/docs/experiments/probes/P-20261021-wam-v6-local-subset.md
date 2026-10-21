---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-local-subset
experiment_id: P-20261021-wam-v6-local-subset
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: d20b565c
claim_id: C1
hypothesis_family: HF-v6-local-adaptation
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: use the bounded local recipe for subsequent contact-consistency and rollout interventions
decision_changed_if_negative: resolve training or data mismatch before investing in coupling changes
status: RUNNING
run_id: local-subset-pretrain-20261021-r2
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
outputs/ptv3-shared-wam-v6/local-subset-pretrain-20261021-r2/; readiness and
recovery use separate sibling directories. Logs/code archive and switch receipt
belong to the declared output/tmp directories. Environment: local graspenv,
Python3.8.20/Torch2.4.1+cu121; inspect runtime at startup.

## Results

Preparation PASS:2351 actual V6 reader endpoints,1177 sequences,33672 TRAIN/3356
DEV windows, no shared sequence IDs.7 initial focused tests passed; an additional
full-recipe-versus-subset admission regression makes8 passing tests. Code
preparation d1079f6a; launch fixed d20b565c and archived under
`tmp/local-subset-pretrain-20261021-r2/`. Third-party PointWorld is a read-only
symlink to the installed repository dependency. Runtime versions are written
into each stage run-manifest.

PPO r3 normal STOP at1575: all3 ranks STOPPED,6 workers/supervisor exited,
61,158,850byte final model+optimizer CPU-loaded and finite. No unrelated
processes stopped. First r1 readiness failed before any optimizer update due
to AF_UNIX path length in the long output directory; retained as diagnostic
evidence. Fixed by putting multiprocessing IPC under the project's short
`tmp/v6i-<pid>` path. r2 rechecks the saved PPO checkpoint and reuses its normal
STOP; no redundant training/resume of PPO.

Readiness8 updates and strict recovery8→10 PASS. Complete1,186,322,105byte
checkpoint has568 optimizer states, all at8; recovery has all568 at10. Both
full CPU audits find finite weights/moments and identical inherited normalizer
buffers. Actual entrypoint resume/config/data/effective batch bindings verified.
Sustained training active onGPU4 at startup acceptance step80, loss1.132238,
1.3559updates/s, CUDA allocated peak9.54GiB. Last four15s GPU samples show
99/76/78/100% utilization, physical memory12,075–13,271MiB. GPUs5/6 remain2MiB
and0%; unrelated0–3/7 jobs unchanged. Free disk216.92GiB. Measured remaining
step-only ETA23.6min excludes periodic DEV/checkpoints; initial planning~30min,
with4h upper boundary. These are startup measurements, not completion claims.

Fixed DEV GRAB hand EPE at initialization49.9303mm, after8 updates49.8771mm;
source-macro objective1.077912→1.076177. This tiny startup difference is not
scientific evidence of improvement. No new rollout/physical success measured.
Baseline and current DEV panels are not interchangeable with the remote full
recipe; do not compare their scores as if evaluation admission were identical.

Live receipt: outputs/ptv3-shared-wam-v6/local-subset-pretrain-20261021-r2/process.json.
Startup acceptance: same directory/startup-acceptance.json; per-stage config,
run-manifest, progress, DEV reports and checkpoints under readiness/recovery/training.
Conda history exported; Python3.8.20, Torch2.4.1+cu121/CUDA12.1. Supervisor806802,
training worker810310. STOP at the run root requests a controlled saved exit.

Repository verification:21 tests in the complete v6 test suite passed after the final fix; changed-document/seed/card checks pass. The global `tools/verify.py
--changed` was also run, and fails on pre-existing missing v2 schema fields in
six trajectory-policy experiment cards. Those unrelated cards were left intact;
no new-card or seed-ledger failure was reported.

First scheduled sustained DEV at200 is published. GRAB hand EPE51.5141mm versus
initial49.9303mm; macro objective1.065935 versus1.077912. Thus loss improvement
must not be presented as improved hand trajectory accuracy. Continue the bounded
2000-update adaptation to observe the prescribed trajectory metrics; do not
promote it as a better execution model based on this early result. Original
initialization and readiness checkpoints remain intact. The first sustained
checkpoint CPU audit is recorded in sustained-acceptance.json.
