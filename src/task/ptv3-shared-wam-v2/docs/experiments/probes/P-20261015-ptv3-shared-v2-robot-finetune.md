---
schema: ref2dex.probe.v2
probe_id: P-20261015-ptv3-shared-v2-robot-finetune
experiment_id: P-20261015-ptv3-shared-v2-robot-finetune
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: fd83aebd
claim_id: C1
hypothesis_family: HF-v2-remote-robot-adapter
probe_index_in_family: 1
seed_pool: probe
seeds: [151]
decision_changed_if_positive: continue v2 remote-WAM fine-tuning with the explicit PD-target action adapter
decision_changed_if_negative: stop this adapter and diagnose data/model/wiring before spending a long run
status: RUNNING
run_id: ptv3-shared-v2-robot-finetune-20261015-r5
---

# V2 remote-WAM robot fine-tune

Current execution: r5 restores r2/step3000 under isolated code fd83aebd,
supervisor1544990, GPU4/5/6/7/global512, target12000. Model/optimizer/rank RNG,
config and data preserved; no warmup. r2 stopped at logged3340 when concurrent
src edits triggered the implementation guard. Frozen worktree keeps numeric
model/data/optimizer code from dccd77ee, changing only artifact/IPC root and
code-provenance paths and retaining the same vendor ptv3 code. Current working
tree edits are preserved. r3 dependency-import and r4 long IPC startup failures
are retained and have no valid training-update evidence. Temporary seven-card
exception allows native collection GPU0/1/2 to run simultaneously.
Resume logging has an inherited ETA denominator bug (uses total step rather
than new updates): derive seconds/update from elapsed_s/(step-3000), not the
reported minute-scale ETA. First120 new updates took117.5s (~0.98s/update).

This Probe adds a v2 Chunk-R action adapter on top of the remote v2 shared WAM
checkpoint and trains on the frozen DExplore RL660 corpus. The action loss uses
the feedback-compensated `q_pd` target encoded as
`wrist-geometry-continuous-lift.v1`. Measured `q_real` remains the current-state
input/diagnostic. Geometric `q_ref` is loaded from the read-only source into the
sample contract for provenance and future audits, but is not silently used as
the teacher action.

## Decision Note

### 2026-10-15 10:00 CST — fixed-workspace input admission repair

The first four-GPU/global512 smoke failed before any update on rank0/2:
valid hand grid minima were -345/-385 under the unchanged parent origin.
Retain `robot-adapter-smoke-remote80k-4gpu-20261015-r1` and its error manifest.
CPU replay independently reproduced the exact failing batches. The geometry
transform/masks are correct: measured hands in some otherwise object-tracking
qualified windows are over6m away from the object, near the robot's +/-5m
translation limits. The original corpus only qualified object tracking.

Keep the parent's voxel origin, metric points, serialization and labels
unchanged; integer grid translation would change Hilbert order/pooling. Add
explicit `observed-fixed-grid.v1` admission in the four-GPU config. It checks
only current observed hand/CAD with the float32 fixed grid contract (including
serializer depth); no future labels/outcome select the subset. TRAIN retains
110,469/110,777 windows, excludes308 across6 sequences; dev retains11,908/11,908.
Original corpus and original admitted ticks stay untouched. Every run saves
the full per-sequence retained/excluded ticks and original-to-new index map
in `encoder_workspace_admission.json`; R normalizers use retained TRAIN ticks,
and sampling/validation use rebuilt subset indices. This is encoder input
support, not a guarantee of successful demonstrations or clean future labels.

The full CPU replay matches an independent agent's counts; all four corrected
first batches pass the original `fixed_workspace_grid`. Seventeen contracts
pass, including known-answer real-sample rejection, current-frame/future
independence, sampler/normalizer membership and index mapping. Implementation
commit `dccd77ee`. Retry only the affected engineering gate in fresh
`robot-adapter-smoke-remote80k-4gpu-20261015-r2`, then start the authorized
four-GPU scientific r2 if it passes, with all original stop/budget conditions.

### 2026-10-15 09:52 CST — user requests four GPUs and larger batch

The user changes execution to four GPUs and a larger batch. The owned
two-GPU r1 supervisor (PID1417465) was verified by command line and normally
signalled; it terminated only its own torchrun group. The last saved progress
was step1/global64; its directory and user-requested reconfiguration receipt
are preserved. Its terminal FAILED/signal15 record is an intentional stop,
not method evidence. No checkpoint from r1 will initialize the replacement.

Use `robot_remote_finetune_4gpu.json`: GPUs4/5/6/7, 128/rank/global512,
12,000 fresh updates. All learning rates, weights, seed, data splits, parent
step80000, no-warmup/fresh-R policy and q_PD labels stay fixed. Sample budget
is eight times the earlier global64 plan at the same update count, so this
is not a matched batch-size comparison. First run four real updates at the
actual four-GPU/global512 size (<=600s), then launch independent run r2 with
the owned supervisor and the same 80,000s/8GB/20GiB boundaries. Remote GPU0/2
pretraining is complete, all four assigned local GPUs are free after r1 exits,
and project allocation remains below the global six-GPU limit. Code08f0b439
only adds the four-GPU config and allows engineering smoke up to256/rank;
it does not change model, labels, loss, or scientific optimization behavior.

### 2026-10-15 09:48 CST — remote parent available; user authorizes launch

The user explicitly requests fine-tuning the remote WAM-v2 now, without R
warmup, using `q_PD`. Use the copied scientific `step80000.pt` from
`outputs/ptv3-shared-wam-v2/remote-parent-20261015-r1/`; its receipt identifies
remote pretraining code `1650ffce` and the complete 80k parent. The new R and
optimizer start fresh. Keep predicted-tau action gradients into WAM+R and the
GT-tau anchor into R, with the existing all8 objective and fixed 12k protocol.
This probes whether the remote pretrained forecaster can adapt with a fresh R;
it does not establish a warmup comparison or physical grasp success.

The first copied-parent check rejected local stats because that file contains
extra `relocation` metadata. Field-by-field comparison found no numerical,
source, split, or normalization-contract difference; only `data` path and the
extra relocation annotation differ. Extract the parent's original stats to
`remote-parent-20261015-r1/stats_step80000.json` and use those exact statistics
for smoke and training, preserving the old local stats. Preflight evidence is
`tmp/v2-robot-launch-20261015/parent-stats-preflight.json`.

Run the real remote-parent four-update full-batch DDP engineering check on
GPUs4/6 (32/rank, global64; <=600s), then the owned supervisor for 12,000
updates, <=80,000s, <=8GB artifacts, >=20GiB free disk. No active local or
remote training was found at launch preflight; maximum two project GPUs are
needed. Stop on nonfinite data/gradients, resource conflicts, implementation
drift, missing progress or budget breach. If engineering passes, launch the
fixed Probe directly; if it fails, preserve the failed run and repair only the
implementation blocker before retrying. Native multi-source manifests remain
engineering-only and are not inputs to this RL660 run.

The reference-q replay and branch-aligned follow-up both failed direct
execution gates while preserving the distinction between q_ref, q_pd, and
q_real. The remote v2 checkpoint has only the pretraining forecaster, so the
minimal compatible route is a new v2 adapter under this Task: load the remote
v2 model strictly, attach a fresh 15-D continuous-lift Chunk-R head, and keep
the three q fields explicit. A v1 weight reshape is prohibited.

## Protocol

Before scientific training, run the four-step engineering smoke on GPU 4/5/6/7 and
require finite loss, nonzero gradients through v2 shared/geometry and R, exact
checkpoint load, and no foreign GPU process. Then run the bounded two-GPU Probe
on GPUs 4/5/6/7 with seed 151, frozen RL660 split, batch 128 per GPU, 12,000
updates, validation every 500 updates, and an 8 GB owned-artifact budget. Use
`tools/run/launch_robot_finetune.py` so the owned torchrun group, progress,
GPU, disk, and stall timeout are recorded throughout the run. The parent must
be a non-engineering v2 pretraining checkpoint with matching model schema,
config architecture, horizon statistics, and the pinned remote provenance
commit `1650ffce293fb201fd7d12d5797ed71608dff5c5`. The fine-tune itself is local
on GPUs 4/5/6/7: the remote WAM contributes only the copied parent checkpoint;
the RL660 corpus, geometric q source, and statistics remain the local
read-only inputs whose paths are recorded in the manifest.

## Data contract and interpretation

Each robot window carries `q_ref_next` from geometric
`interaction_hand_inspire.pt[:,373:391]`, `q_pd` from RL660 `pd_targets`,
`q_real` at command time, and `q_real_next` after command execution. Only the
existing physical action target encoded from q_pd contributes to the action
loss. A successful smoke proves wiring only; a short Probe can be PROMISING,
UNPROMISING, or UNCLEAR and cannot establish a grasping claim.

## Limitations / stop conditions

This is offline supervised fine-tuning; it does not execute a learned policy
in Isaac or on hardware. Stop on nonfinite values, checkpoint/schema mismatch,
foreign GPU PIDs, no progress for 30 minutes, less than 20 GiB free disk, or
an adapter audit failure. If the remote SSH tunnel is unavailable, do not
substitute the local v2 checkpoint while claiming a remote-WAM result; use it
only for engineering smoke and record the parent provenance.

## Engineering smoke status

### Four-GPU gate and launch (2026-10-15 10:01 CST)

Corrected `robot-adapter-smoke-remote80k-4gpu-20261015-r2` is COMPLETED:
four ranks,128/rank/global512, four finite updates,256-window dev evaluation
and a complete engineering checkpoint. Step4 loss10.2761, training-loop
elapsed6.46s; rank0 peak allocated5497MiB, physical snapshot6557/6541/6549/6747MiB.
These four updates are engineering only, not convergence evidence. Independent
read-only review confirmed the subset, sampler/normalizer and provenance
implementation;17 tests pass. The scientific run starts fresh from remote
step80000 with a new R/optimizer, not from either smoke or stopped r1.

Run `ptv3-shared-v2-robot-finetune-20261015-r2` was launched under owned
supervisor PID1436540 on GPU4/5/6/7. It uses the four-GPU config, exact parent
stats, q_PD labels and explicit workspace admission described above. The
supervisor polls every60s and retains progress/GPU/storage/status records in
`tmp/ptv3-shared-v2-robot-finetune-20261015-r2/`, mirrored into its output.
First stable throughput/utilization observation is recorded separately in
the run's `startup_observation.json`; no improvement or grasp claim yet.

At10:03:29 CST the30s observation reached step60: stable step20-difference
throughput0.9514s/update,538.1 samples/s, estimated remaining pure training
3.16h (validation/save overhead additional). GPU4/5/6/7 utilization means were
86.97/90.30/87.83/90.90%, physical memory peaks6793/6777/6785/6747MiB.
By10:03:44 progress reached step80 with finite loss5.8672. The owned supervisor
remained live and healthy; no resource anomaly was observed. These are startup
capacity measurements, not evidence of convergence or task competence.

### Remote 80k parent engineering gates (2026-10-15 09:50 CST)

`robot-adapter-smoke-remote80k-ddp-20261015-r1` completed four DDP updates,
32/rank/global64, with the real copied step80000 parent, the parent's exact
stats, two workers/rank, and finite loss/gradient/save/evaluation. Complete
step4 checkpoint is engineering-only and will NOT initialize the scientific
run. Fresh R's zero output head makes action-to-WAM gradient zero before its
first update; this is initialization, not an R warmup stage. The separate
actual-sample post-step4 GPU audit verifies action-only WAM/R gradient norms
`0.0006093/1.39159`, anchor-only WAM/R `0/1.38385`, and exact q_PD target
encoding. Evidence: `tmp/v2-robot-launch-20261015/action-gradient-route.json`.
The 13 v2 contract tests and changed-document verification pass. A read-only
independent agent review found no startup blocker. This validates engineering
only, not convergence, grasping, or scientific benefit.

The v2 adapter passed single-GPU and two-GPU four-step smokes using the local
scientific v2 `step1000.pt` only. The latest smoke records nonzero gradients in
`joint_geometry`, `shared_and_heads`, and `retargeter`, finite loss, explicit q
contract, and immutable checkpoint save. These runs are stored under
`outputs/ptv3-shared-wam-v2/robot-adapter-smoke-*` and are engineering-only.
The 660-sequence q-reference preflight also passed (all files finite, matching
frame counts and raw schema). These early local-parent smokes did not
constitute scientific remote-parent fine-tuning. The remote parent was later
copied and validated; actual four-GPU launch and input-admission repair are
recorded above.

## Early plateau inspection requested by the user

At logged step2220, fixed dev256 has only four evaluations:

| Step | Total objective | Predicted-τ action | GT-τ anchor | Forecast residual |
| --- | ---: | ---: | ---: | ---: |
| 500 | 5.4520 | 0.5406 | 0.5373 | 4.3741 |
| 1000 | 5.2023 | 0.4897 | 0.4838 | 4.2289 |
| 1500 | 5.2487 | 0.4606 | 0.4542 | 4.3338 |
| 2000 | 5.2622 | 0.4361 | 0.4282 | 4.3978 |

All component weights are1, so forecast residual is exactly total minus the
two action losses; hand vs scene cannot be separated from saved dev logs.
Total dev objective is flat/slightly rising since1000, while action and anchor
still improve10.94%/11.48%. The latter are normalized all8 prediction losses,
not physical first-PD error or grasp rate. Rank0 logged train loss means over
successive500-step blocks fall4.241→3.568→3.000; these are varying training
draws and not a matched dev panel. Do not call the whole optimization stalled.
Keep the authorized recipe running and inspect more fixed-dev points; do not
early-stop from these four points. best.pt is selected by total objective and
still points to1000, which does not imply action progress stopped.
Evidence: run plateau-step2220.json/.png, generated without inference or
production edits by tools/audit/audit_robot_plateau.py.

## Updated plateau inspection at logged step8340

User requests the current action-loss trend. Active r5 is still RUNNING on
GPU4/5/6/7, target12000/global512. The latest fixed-dev256 evaluation is8000;
the original r2 and resumed r5 validation index arrays are exactly equal.

| Step | Predicted-τ action | GT-τ anchor | Total dev objective |
| --- | ---: | ---: | ---: |
| 5000 | 0.391091 | 0.376277 | 5.391422 |
| 6000 | 0.384994 | 0.368858 | 5.455194 |
| 7000 | 0.380777 | 0.362222 | 5.529810 |
| 7500 | 0.376566 | 0.357895 | 5.518565 |
| 8000 | 0.376261 | 0.355949 | 5.516710 |

Predicted-τ action improves7.55% since3500,2.27% since6000,1.19% since7000,
but only0.081% in the last500 updates. GT anchor improves3.50% since6000.
Rank0 train action means over successive1000-update logged blocks ending
6000/7000/8000 are0.466486/0.447449/0.443855; the latter improvement is0.80%.
Thus action fitting is still improving with diminishing returns, approaching
a plateau; it is not proven completely stalled. Total dev is already flat
around5.5 and higher than early evaluations; improving action does not imply
improving forecast or physical grasp behavior. Keep the authorized12000 cap,
do not extend from falling training loss alone.

Evidence: r5 `plateau-step8340.json/.png`, CPU-only read/plot of existing
metrics with no model inference or configuration changes. Corrected resume
ETA at8340 is about60min, using elapsed/(step-3000); the inherited logger's
elapsed/total-step formula underestimates time remaining. GPU4/5/6/7 observed
at89/92/100/100% utilization and roughly6.6GiB physical memory per card.

## Stage preservation requested by the user

User wants the different training stages available for later downstream
comparison because action and forecast losses favor different stages.
At this request, complete checkpoints still exist at1000/3000 in r2 and
3500/4000/8000 in r5. Old policy retained best/latest plus4000-step milestones
and one recent recovery file; some intervening saved checkpoints had already
been rotated out. Not every500-step dev evaluation created a checkpoint.
An extracted R-only step500 snapshot is not a complete WAM+R checkpoint and
cannot reconstruct a missing joint stage. Do not claim all stages survived.

Action: immediately hard-link all five existing complete checkpoints into
`outputs/ptv3-shared-wam-v2/robot-finetune-stage-archive-20261015-r1/`, then
watch for all subsequently saved stages through12000. Separate CPU-only
watcher `tools/audit/archive_robot_stages.py` checks every30s, stops within4h
or when the source completes/fails, caps logical checkpoint bytes at16GB
and requires20GiB filesystem headroom. This is bounded bookkeeping, not a
new learning experiment. No training code/configuration/optimizer change;
the existing8GB active-run budget remains in force. Hard links initially
consume negligible new data blocks; they retain blocks that source pruning
would otherwise free. All five link identities verified; the archived8000
checkpoint loads as a complete robot model/optimizer/stats/four-rank RNG.
Engineering smoke confirms source unlink cannot remove the archive, partial
writes are excluded and the archive budget is enforced.

Archive manifest retains source identity and each stage's objective/action/
anchor/forecast-residual metrics. Future downstream qualification must compare
these stages on the same task panel; neither lower action nor lower forecast
alone proves better grasping or adaptation. Stage selection can find the more
useful weight, but causal attribution between the two losses would require
separate matched interventions. That downstream evidence is deferred to the
next decision-relevant qualification Probe; current work preserves options.
