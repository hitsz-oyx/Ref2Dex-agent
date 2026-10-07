---
schema: ref2dex.probe.v2
probe_id: P-20261007-pointworld-action-ddp
experiment_id: P-20261007-pointworld-action-ddp
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 06933b3
claim_id: C3
hypothesis_family: HF-pointworld-unified-action-effect
probe_index_in_family: 4
seed_pool: probe
seeds: [210, 212, 213, 216, 217, 219, 220]
decision_changed_if_positive: retain efficient action-only DDP pretraining and its improved validation checkpoint
decision_changed_if_negative: diagnose larger-batch optimization before extending training
status: PROMISING
run_id: pointworld-action-ddp-20261007
---

# User-directed action-only three-GPU warm start

Decision: retain the improved action checkpoint. The user-directed three-rank
recipe completed all10000fresh updates at global batch192. Fixed balanced
validation improves20.85%from the imported checkpoint; this is a pretraining
Probe signal, with robot policy utility still untested.

## Motivation and Decision Note

This serves the Task's observed-action physical prediction subgoal under C3;
future measured human hands remain predictive supervision, not a demonstrated
causal intervention or robot policy utility. User explicitly requests stopping
three independent arms, then selecting hyperparameters from memory and GPU
efficiency and training the real-action model with three GPUs.

The old corrected Probe stops successfully with H12661/action12163/shuffle11774,
preserving best/latest/final checkpoints and all logs. Its matched40000-update
test gate is canceled; it remains UNCLEAR. See the
[old card](P-20261007-pointworld-temporal-wm24.md) and its run's
`user-stop-for-action-ddp.json`. No negative result or matched final comparison
is inferred from this interruption.

The hypothesis is that larger per-rank microbatches plus verified Hilbert fusion
can increase useful training throughput while continuing to improve the fixed
validation metric from the imported action checkpoint. A positive signal keeps
this recipe; regression motivates checking optimization before more investment.
Cheapest engineering selection:45real training updates each at per-rank
batch16/32/64, three ranks, excluding the first15updates from timing. No science
conclusion or test access occurs in the selection. Choose the smallest batch
within5%of fastest measured throughput with allocator reserve below75%of VRAM.
The engineering selection cap is15minutes; stop on device conflict or failure.

Resource boundary: three explicitly assigned GPUs, <=4task GPUs in total; outputs plus
temporary artifacts remain <=300GB. Retain the original absolute training
deadline1791424717.7631629 (2026-10-08 09:58:37 Asia/Shanghai), without a new24h
allocation. Stop on user request, nonfinite values, source drift, worker failure,
resource conflict or deadline. No new branch, external modification or push.

## Initialization and changed training contract

Parent: `outputs/cm-pointflow-effect-pretrain/pointworld-temporal-wm24-20261007/train-action/latest.pt`,
step12163 after graceful stop. Load all model tensors strictly, including the
original normalization buffers. Preserve data manifest, train-only stats,
temporal architecture, horizon, loss weighting, and vendor/source identities.
Reset optimizer, schedule, step and draws explicitly; this is not exact resume.
Import initialization hashes and parent SHA256 are written to input_manifest.

The new global effective batch is `microbatch_per_rank * global_accumulation`;
global accumulation3 means one microbatch on each of three ranks. DDP averages
these three separately normalized losses. A larger microbatch changes valid-mask
and motion-weight denominators and the batch spatial grid origin, so new loss
values and optimization trajectories cannot be directly compared as an unchanged
continuation of the old2x8recipe. Integer-exact Hilbert fusion preserves all four
serialization orders and time identity; it does not change the encoder structure.

Proposed fresh schedule: AdamW lr1e-4 (no linear batch scaling), weight decay.01,
100update warmup followed by cosine decay to.1of peak, gradient clip1, BF16 forward
and FP32 physical loss. Maximum10000new optimizer updates, further bounded by
the original absolute deadline. Seed219, fresh train draw220; fixed corpus
split210, stats216, balanced validation212 and natural validation213 retained.
Final batch and timing evidence will be recorded after engineering selection.

## Validation and checkpoint record

Keep validation inference microbatch2 to retain the old panel batching and grid
origin. Evaluate the imported checkpoint at new step0 before optimization;
retain full metric rows in `validation.jsonl`, not only an overwritten latest
file. Balanced256 validation every250updates and natural256 final validation;
report motion-anchor h24point EPE as primary and h8/rotation/static as diagnostics.
Final inference shuffle checks observed-action dependence on validation only.
No TEST split is opened for tuning or this action-only exploratory run.

Save best.pt when the primary validation metric improves, latest.pt every250
updates, and final.pt on normal completion or graceful stop. Checkpoints include
optimizer, new update/draw position, source identities and RNG for all three
ranks. Native resume requires the same world size/config/backend/sources.
Weights are checked identical across ranks. Larger-batch stage gains alone are
not a new matched H/action/shuffle scientific claim; controls for the new recipe,
multi-seed validation and robot utility remain future evidence.

Run output: `outputs/cm-pointflow-effect-pretrain/pointworld-action-ddp-20261007/`.
Selection: `outputs/cm-pointflow-effect-pretrain/pointworld-action-ddp-selection-20261007/`.
Entries: [DDP trainer](../../../tools/run/train_oakink2_pointworld_ddp.py),
[batch selection](../../../tools/audit/select_pointworld_ddp_batch.py).
[Launch supervisor](../../../tools/run/launch_pointworld_action_ddp.py) refuses
occupied devices, records runtime sources/PIDs/deadline, detects foreign GPU
processes and requests checkpoint-preserving stop only from its own workers.

## Engineering checks

Ten CPU/Gloo contract tests pass: reference1/2/4and new3-rank draw reconstruction,
three-rank masked objective/gradient averaging, RNG and collective stop, strict
native resume rejection, and model-only initialization permitting recipe changes
while rejecting altered horizon or normalization identity. CPU is used for this
small distributed mathematical contract; real network timing uses the GPUs.
Actual retained step12163checkpoint initialization was checked on GPU1: all model
tensors, including normalization buffers, match the parent exactly. Parameter
SHA256 is6bb8c7dfa7d02bcfe3e19b73006c1d70110cc6e426e0b568222ed00d6aaf3d28;
parent file SHA25658397295bdd4fd2bebae4cd6089046f52369637a4010a572789c92deafb2d6d2.
Evidence: `outputs/cm-pointflow-effect-pretrain/pointworld-action-ddp-init-check-20261007/verification.json`.
Only initialization was checked; no optimization/validation took place in that
check, and its GPU process exited afterward.

Resource checkpoint: after old workers exited, GPU0was newly occupied by a
PointMotus process538081outside this repo, and GPUs3-7had other jobs. User states
they will coordinate the third card and provide its index. GPUs1/2are available;
no foreign process is signaled or shared. Three-rank selection and the new full
run are therefore not launched until the third allocation is confirmed. No
automatic device substitution or two-rank production training is performed.


## Three-card allocation and measured selection (2026-10-07)

User confirms GPUs0/1/2may now be used. All three were checked empty (about2MiB
each, no compute process) before the bounded selector launched. Runtime source
commit4f2d9d5, identical imported step12163parameter hash for all configurations.
All45updates/configuration completed, losses/gradients remained finite, and final
parameter hashes were identical across all three ranks. Excluding the first15
updates from timing gives:

| Microbatch per rank | Global batch | Median update seconds | Windows/second | Peak allocator reserve GiB | Maximum sampled total VRAM GiB | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 48 | 0.277 | 173.5 | 5.80 | 6.24 | yes |
| 32 | 96 | 0.420 | 228.5 | 12.06 | 12.49 | yes; selected |
| 64 | 192 | 0.728 | 263.6 | 23.24 | 23.20 | no; exceeds75%reserve boundary |

The larger64case finishes, but has insufficient headroom. Choose32instead of
chasing utilization or filling VRAM. Measured timing includes real-data loading
and DDP communication; data-fetch median for32is0.51ms after prefetch warmup.
45updates are an engineering sample, not a guarantee of long-run throughput or
all-window peak memory. Larger-batch loss values do not retain the old2x8meaning.

Frozen proposed production config is
[pointworld_action_ddp_wm24.json](../../../configs/pointworld_action_ddp_wm24.json):
per-rank32, global accumulation3/local1, global96, workers4/rank, lr1e-4,
100warmup, cosine floor.1, clip1, BF16 forward/FP32 physical loss, maximum10000
new updates, checkpoint/validation250, fixed evaluation microbatch2. The original
absolute deadline still applies. No test data was accessed. Raw logs, source
identities, utilization samples and selection rule are in selection.json under
the declared selection output. Actual production code commit/PIDs will be
recorded in the launch and input manifests.


## User-directed batch64check

User explicitly requests trying64per GPU despite the conservative allocator-
reserve selection. Decision Note: retain the same three cards and run200actual
training updates at microbatch64/global192, unchanged data/architecture, lr1e-4
and100warmup. The45-update case already finishes; allocator peak live allocation
is16.19GiB while reserved cache reaches23.24GiB. Reserved but unused allocator
cache is reclaimable and is not equivalent to live tensor memory; record both
without asserting a25%headroom guarantee. This check serves the concrete batch
selection decision, bounded by5minutes and the existing storage/time campaign.
On finite completion, validate checkpoint serialization/native3-rank restore and
use64for the requested production run; on OOM keep its evidence and fall back
to32rather than overwrite previous checkpoints or preempt another GPU job.
Output: pointworld-action-ddp-batch64-check-20261007. These engineering updates
are discarded for production initialization, which remains the original12163
latest model. No test split is accessed.


## Final selected batch after extended check

The200-update64per-rank probe finishes in169.33seconds including startup/data
loading and final serialization, with median measured update0.72601seconds
(264.46windows/second), maximum live allocation16626.67MiB (16.24GiB) and
maximum allocator reserve23794MiB (23.24GiB). All three final parameter hashes
match6696713f19230ee0787e0144736c32bfc6e4f372c4b000090453e2f71636d1c1.
All losses/gradients finite, no OOM. The earlier32choice is superseded by the
user-directed64check and its successful outcome; the previous75%allocator-
reserve heuristic is not asserted for this choice. Live-memory reserve is
substantial, but allocator cache/sparse-workspace peaks vary, so future OOM
remains an explicit stopping condition, not a claim ruled out by this short test.

Production config now uses64per rank/global192with one synchronized backward
per rank/update. Other parameters retain the proposed lr1e-4/100warmup/10000
updates/250validation and checkpoint intervals/fixed evaluation batch2. The
engineering200updates are not imported: full training starts again from the
retained original12163model and a new optimizer/draw. The prior32check also
verifies actual three-rank native continuation:1to2optimizer updates, exact
resume starting parameters, changing trained weights, advanced CUDA RNG on all
ranks, and rank-synchronized final weights; its verification.json is under
pointworld-action-ddp-resume-check-20261007.


Batch64native restore also passes at step200: every model and optimizer tensor,
optimizer parameter-group settings and all three Torch/CUDA RNG states are
exactly equal after restore/serialization; unwrapped keys retain the existing
model contract. Twelve held-out validation windows can be read/evaluated at
fixed microbatch2, with480finite metric entries and no source drift. This is
engineering validation readability, not a predictive quality conclusion. Raw
proofs are native-restore-verification.json and validation-read-check.json under
the batch64check output. All short-check GPU processes exit before production.


## Production launch record

Actual source/config runtime commit06933b3b7b773655e433e5c73537dbb6f71ba1b5;
launcher592221, torchrun592237, rank PIDs592274,592275,592276 on assigned0/1/2.
Initial imported parameter hash matches the retained original12163checkpoint
exactly. All workers are alive, sources show no drift. Fixed balanced256
validation at new step0gives moving-anchor h24point EPE14.06723mm; this is the
imported model baseline before new optimization, not an improvement claim.
The initial best.pt is retained at step0and new best/latest/final policies are
active. Full rows are retained in validation.jsonl.

Startup check at new step99 shows finite loss1.050297, gradient norm0.555811,
update0.7138seconds, global192and no OOM. Training remains in progress and
status UNCLEAR. The fixed deadline and maximum10000new updates are unchanged.
Actual identities/config/draw hashes and current process status are in the
production input_manifest.json/group_status.json rather than inferred from
later documentation-only commits.

## Completed production run and convergence check (2026-10-07)

The owned launcher and all three ranks finish normally (exit0), with10000fresh
updates in8305.38seconds (2h18m25s). All three final parameter hashes match;
all runtime source/input hashes still match the launch manifest. No test split
was used. This run retains the627-sequence OakInk2 corpus and original split;
it does not train on the separate in-progress dataset expansion.

Primary metric is moving-anchor h24 point EPE on the same256-window balanced
validation panel throughout (30Hz,24steps=0.8s):

| Fresh update | Primary EPE (mm) |
| --- | ---: |
| 0 (imported old action latest, step12163) | 14.0672 |
| 2500 | 12.3408 |
| 5000 | 12.0379 |
| 7500 | 11.5370 |
| 9500 (best) | 11.1026 |
| 10000 (final/latest) | 11.1342 |

Final improves20.85%over initialization; best improves21.07%. The last2500
updates still improve3.49%, although per-update gains have slowed. The last
two1000-update mean training losses are0.93067and0.93668; this small increase
alone does not establish convergence. Assessment: diminishing returns and
training-loss flattening, but the fixed validation curve has not established
a complete plateau. This cosine schedule also ends at lr1e-5, so a slow final
segment does not prove that a different continuation cannot improve.

Final balanced moving-anchor h8 EPE is3.6783mm; h24 rotation error is0.15862rad.
The natural256-window panel gives moving-anchor h24 EPE12.7213mm (static
baseline39.4330mm), versus balanced11.1342mm (static26.1252mm). These panels
have different composition and should not be compared as a training trend.
Final validation-only action shuffle raises balanced h24 EPE to32.7580mm:
the model depends on the observed future-hand input. This does not establish
causal action effects or robot policy usefulness.

Checkpoint metadata was actually loaded on CPU for inspection (no model
inference): best.pt is step9500, latest.pt/final.pt are step10000, each about
0.565GiB including optimizer/RNG state. Retain best for subsequent adaptation;
do not automatically extend the expired10000-update recipe or reset its budget.
Future evidence: independent test evaluation, matched controls/multiple seeds,
robot transfer and eventual matched trained-policy Cm-on/off utility.

Evidence in the declared run's train-action directory: result.json,
validation.jsonl (41fixed-panel measurements), completion_summary.json
(recomputed trends and inspected checkpoint identities), convergence.png,
best.pt/latest.pt/final.pt. Existing initial/raw records are preserved.
