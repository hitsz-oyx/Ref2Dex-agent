---
schema: ref2dex.probe.v2
probe_id: P-20261017-wam-v4-interaction-head
experiment_id: P-20261017-wam-v4-interaction-head
date: 2026-10-17
task: ptv3-shared-wam-v4
branch: main
git_commit: b65872f8
claim_id: C1
hypothesis_family: HF-wam-interaction-head
probe_index_in_family: 1
seed_pool: probe
seeds: [170]
decision_changed_if_positive: keep the dynamic graph interface and run a matched tau-only versus tau-plus-C executor probe
decision_changed_if_negative: remove or redesign C before any long pretraining budget
status: RUNNING
run_id: pretrain-nine-v4-fastinput-warm80k-20261018-r1
---

# WAM-v4 target-object interaction field

This is an architecture probe, not a robot-benefit result. It keeps the v3
shared encoder and hand/scene heads and adds a target-object interaction field:

```text
G_task -> WAM -> (tau_hand[24], tau_scene[24], C[24,128,7]) -> executor
```

`C` has shape `B,O,24,128,7`: contact confidence and six semantic hand-part
probabilities. The 128 object queries carry position and unit normal as inputs;
they are not regression targets. Labels use a deterministic 1cm voxelized
semantic hand scaffold. All nine sources use the same 512-surface/32-flow/128-
interaction contract. Underfilled native geometries are rebuilt from original
CAD triangles with distinct positions and unit normals; cyclic repeats are
rejected. The original 32 flow query identities remain fixed.

The trainer supports a model-only v3 warm start. Compatible trunk parameters
transfer and the graph relation modules initialize fresh. The cheapest
qualification is a forward/backward smoke and a masked-label audit. Only after
those pass should a matched `state + tau` versus `state + tau + C` executor
probe decide whether sustained pretraining is justified.

The local contract suite passes (`63 passed` in the prior full run). Remote
v4 interaction/data tests pass (`8 passed`), and a real four-GPU main3/native
mixed batch passed the gradient audit, finite loss, backward update, and
checkpoint write. The NAS filter artifact is
`data/v4-window-filter-train-20261018-r4.json`; it uses conservative bounding-
box contact candidates for thinning only, while exact C labels remain the
voxel-scaffold training path. The scaffold query lookup is vectorized to keep
the online labels practical; the nine-source raw v3 data remains untouched.

## 2026-10-18 contract-fixed warm start from the existing v4 checkpoint

The audited input/label fixes (unified target/static 18-D features, current-
observation C origin, moving-object non-contact class, realtime metrics, v4
online tau contract) are in code commit `812c26aa` and its follow-ups. The old
remote run was stopped after its retained best at step 13000; its checkpoints,
logs and filter artifacts remain untouched.

The new run is **not a continuation of the old curve**:

- `run_id`: `pretrain-nine-v4-contractfix-warm13000-20261018-r1`
- remote code: `training-code-contract-20261018-r1`, commit `2183eb46`
- initialization: `--init-v4 <old step13000.pt>`; model weights only, with
  optimizer, step, per-rank RNG and historical best reset
- data identity: `input_semantics=v4.target-static-h4-2mm.current-hand.v1`,
  `interaction_semantics=v4.current-interaction-origin-minus5cm.v1`
- recipe: existing `recipe-auditbound-20261018-r1` and its frozen
  `window-filter-{train,dev}.json`. The new `moving_object_noncontact`
  classifier has **not** yet regenerated those filter manifests, so window
  sampling coverage is still the old one; features and C labels are the fixed
  ones.
- supervisor: `2638598`; four remote GPUs, global batch 512, per-rank loader
  workers 4.

A `--stop-after 1` smoke first verified strict checkpoint load, baseline
selection, finite step-1 loss and checkpoint write. The full run reached step
40 at loss `1.3867`, `1.674` updates/s, four GPUs at 100% and about 7.7--7.9
GiB each; the launcher ETA is about 47.8k seconds (13.3 h). DEV evaluation is
scheduled at the configured interval; no model-benefit claim follows from the
startup loss.

## 2026-10-18 corrected C supervision and distributed startup

Classification: Blocker. Missing future hand observations and shifted MCP part
IDs invalidate C supervision, and a rank0-only validation collective prevents
any training update. The cheapest discriminator is the label/DDP contract suite
followed by a two-update four-GPU run with the full real-data DEV panel.

The derived target-object contract is now `ref2dex.wam-v4-target-object.v2`:
incomplete active-hand frames and windows with no observed hand are masked;
MCP and tip pairs share their intended finger semantic ID. C BCE and positive
part CE use separate globally reduced denominators during DDP training only.
Rank0 evaluation and gradient audits use local masked objectives. Additional
metrics separate all-query contact MAE, positive-part accuracy and part CE.
The real-batch C gradient audit checks representative parameters in the C
head/query, shared trunk, geometry, hand encoder and goal paths.

The prior uncorrected long run stopped at step 18,300 and remains available as
historical evidence, not a corrected-v4 result. Smoke r4 passed the C gradient
audit but stalled before step 1 because evaluation entered training collectives
while other ranks waited at a barrier; it was stopped through its own STOP
file. The local regression reproduced that call and passes after honoring the
`distributed=False` flag. The focused local suite passes 13 tests.

Next execution: `interface-cfix-warm-smoke-20261018-r5`, four remote GPUs,
global batch 512, two updates, v3 step80000 model-only initialization. After a
successful smoke, start `pretrain-nine-v4-cfix-warm80k-20261018-r1`, same recipe
and hardware, 80,000 new updates, seven-day/8GB run budget and 20GiB free-space
floor. Use a dedicated v4 code checkout and NAS output directory; preserve all
v3 data/checkpoints. Runtime: remote `graspenv`, Python 3.8.20,
PyTorch 2.0.1+cu118. All observations remain offline Probe evidence.

Limitations: v4 inherits frozen v3 TRAIN normalization rather than refitting it
after thinning. At this stage native objects stored with fewer than 512 unique
surface samples still used cyclic repeats; this is superseded by the dense
geometry correction below. The 80k run is explicitly user-authorized, but does not establish
that C benefits robot control. New metrics must be compared on identical
masked panels; historical uncorrected C losses are not matched baselines.

## 2026-10-18 actual distinct512 CAD surface correction

The user asked whether reprocessing really provided 512 surface points. A
complete canonical-file audit found that all 252 main3 geometries already have
512 distinct points and unit normals, but native target geometries had 1,078
deficiencies: UniDex 1,074/2,066, TACO 2/129, HOT3D 1/19, DexYCB 1/20;
H2O 8/8 and HOI4D 286/286 were already qualified. Counts cover native target
geometry identities and all main3 canonical artifacts, not training windows.

Classification: Blocker before the authorized dense-v4 80k run. The cheapest
repair samples only deficient canonical geometries, leaving temporal payloads,
v3 snapshots, original 32 flow queries and normalization values untouched.
`prepare_v4_dense_surfaces.py` sampled 1,078 derived CADs in 52.92s on local
graspenv/CPU (geometry-only processing, no neural workload), producing 12MiB
under `outputs/3d-data-preparation/v4-dense-surfaces-20261018-r1/`.
Each replacement has the 32 original flow vertices plus 480 deterministic
area-weighted triangle samples and unit normals; original queries must match
CAD vertices in the same frame and scale. The v3 target-object data contract
(`ref2dex.wam-v4-target-object.v3`, distinct from model version WAM-v4) now
rejects all underfilled/cyclic-repeated geometry.

The remote derived-recipe script audits all nine sources, rechecks previously
thinned windows referencing repaired geometry (retaining already-kept windows),
binds the new overlay and exercises real TRAIN/DEV mixed batches. User steering
requests this data job run first while interface fixes are verified in parallel.
Do not launch 80k on the older cyclic-repeat recipe. Planned new run identities:
`interface-dense512-cfix-smoke-20261018-r1` and
`pretrain-nine-v4-dense512-cfix-warm80k-20261018-r1`.
The focused data/model suite passes 17 tests including on-CAD sampling,
coordinate/scale mismatch rejection, immutable flow identities, label masks,
rank0 validation without collectives and matched DDP C gradients.

## 2026-10-18 remote qualification and corrected 80k launch

Runtime code is pinned to `992962ec` in the independent checkout
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v4-dense-20261018-r1/training-code`.
Remote and local focused tests both pass 18/18, including an observed hand side
that disappears for the entire future window. A four-GPU synthetic-module
check independently passed rank0-only evaluation followed by collective
training and a C-head parameter update; this is supplemented by the full
real-data run below, not used as a substitute for it.

Remote derived-recipe preparation completed in 27.06s. All 2,780 audited
canonical geometries pass distinct512/unit-normal checks. Rechecking 635 TRAIN
and 242 DEV blocked windows restored one source4 TRAIN window. The actual mixed
reader/collate passed on nine TRAIN and eight DEV sources (source8 has no DEV
windows in this frozen snapshot). TRAIN contains 3,218,740 windows and DEV
545,459; source probabilities and frozen v3 normalization values are retained.

`interface-dense512-cfix-smoke-20261018-r1` completed two optimizer updates on
four GPUs/global batch512 in 62.65s after startup. Full model: 98,716,909
parameters; model-only v3 step80000 transfer: 546/564 state entries, with C
modules initialized freshly. C gradients reach the head/query, shared trunk,
PTv3 geometry, hand encoder and goal. Step2 loss=2.871758 (hand=.149293,
object=.239547, C=2.482918), finite grad norm=2.326080. Fixed DEV objective
3.339636 -> 3.339501; two updates do not establish a learning benefit. Full
checkpoints and both DEV evaluations completed; rank0 peak allocated6,103MiB,
reserved7,052MiB. The smoke's ETA includes validation/checkpoint overhead and
must not be extrapolated as steady throughput.

The user-authorized 80k run was launched with supervisor PID2583130, GPUs
0/1/2/3, batch128 per rank/global512, two loader workers per rank, the same
v3 model-only parent and dense recipe. Maximum training time604800s, owned
artifacts8GB, free-space floor20GiB; its supervisor records GPU state every10s
and stops its own process group after30min without optimizer progress.
Output root:
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v4-dense-20261018-r1/outputs/ptv3-shared-wam-v4/`.
Local small receipts are under
`outputs/ptv3-shared-wam-v4/dense512-remote-evidence-20261018-r1/`.

The full run reached step20 with loss2.842769 (hand=.142880,
object=.222188, C=2.477701), finite grad norm1.951641 and a saved first-step
checkpoint. Step1->20 throughput=.37635 updates/s, provisional remaining ETA
59.03h; this early estimate is not a completion guarantee. Peak rank0 allocated
6,313MiB/reserved7,058MiB; observed physical usage about7.6--7.9GiB per card,
with utilization fluctuating as CPU labels/prefetch and GPU work alternate.
No OOM/nonfinite values or loader timeout observed. The next fixed DEV panel
is step1000; the 80k objective is still running, not completed.

Limitations of the stopped 992962ec run: this remains an offline Probe. Normalization is inherited rather
than refitted after thinning; native geometry resolution is now repaired, but
the native source8 snapshot is still the earlier subset. C metric values are
dimensionless despite the inherited evaluator's `mean_mm` serialization key;
only hand/object displacement metrics use millimetres. No robot improvement
or matched-v3 representation improvement is established.

## 2026-10-18 user audit at 28361da: objective/evaluation/resume correction

Classification: Blocker. Reproduced the supplied archive
`wamv4_audit_28361da.zip` in project tmp. Its scripts contain pinned source
excerpts, so their successful reproduction demonstrates the old defects; it is
not evidence that current repository code is fixed. Repository-level regressions
exercise the real current functions separately. No new architecture or data
collection is needed to distinguish the implementation errors.

The old live `pretrain-nine-v4-dense512-cfix-warm80k-20261018-r1` was stopped
through its own STOP at logged step1200 (loss1.989232), retaining step1,
step1000 and all logs. This was an administrative stop, not scientific evidence
against WAM or C. Its losses and best selection use the previous semantics and
must not be joined into the corrected training curve.

Code `7b6d28a5` makes these changes:

* Hand, motion-weighted object, contact BCE and positive-contact part CE all
  expose their own numerator/denominator. DDP uses world-size times each local
  numerator divided by its global denominator; all four denominators are
  reduced together. Runtime logs retain each rank's counts/weight sums.
  See the [DDP gradient contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).
* DEV accumulates sufficient statistics over each whole source in float64,
  then applies the existing source macro average. Evaluation preserves the
  training RNG stream and mode. Fixed predictions have identical scores and
  A/B ranking for microbatch1/8/16 and grouped/interleaved orders.
* Model forward requires only current geometry/hand/history/goal and optional
  current query validity. Future supervision masks are restricted to losses
  and metrics. `interaction_prediction_valid` describes the current observed
  object/query support. Standalone native sample construction also masks C
  whenever future object transport is unknown; the cleaned reader already
  rejected those windows.
* Initial baseline participates in best selection and gets a complete step0
  checkpoint. Recovery snapshots preserve best score, step, path and originating
  run. Resumption inherits that best and its checkpoint; retention preserves
  dependencies of retained recovery snapshots. The unsupported subset-resume
  flag fails before reading data. Full resume rejects legacy loss/evaluation
  semantics and changed data identity.
* Dense manifest v2 records each NPZ SHA-256. The recipe binds the manifest
  digest, and checkpoints bind the recipe and surface digest. Arrays are loaded
  from the verified bytes. The explicit upgrade tool creates new metadata
  referencing existing arrays; it does not resample or overwrite them.
* Removed mixed seven-channel C MAE, ambiguous `interaction_positive_only` and
  dummy zero `interaction_consistency`. Metrics carry explicit mm/1/nats units.
  C remains a parallel shared-context head; no candidate-trajectory conditioning
  or explicit trajectory--C consistency objective is claimed.

Objective identity: `v4.global-masked-components.v2`; evaluation identity:
`v4.source-macro-sufficient-statistics.v2`. Bound recipe:
`data/recipe-auditbound-20261018-r1` under the existing NAS root. Surface digest
`1e2ec0cdba3eca9825232bdf2df45db9401fbba39482038c099fbcbbc14e53f7`.
All 1,078 overlay geometries reuse the existing arrays and fixed flow queries.

Validation: local and remote focused suites pass29 tests. Actual NCCL/DDP
two-/four-rank complete-loss checks match the concatenated batch for grouped
and interleaved partitions, including an empty-supervision rank; maximum
gradient error5.82e-11/0. CPU regressions also reproduce the audit's A/B example
and confirm corrected source scores A=.9227453, B=1.6258998 regardless of
microbatch grouping or order. Hash tests reject moving the480 non-query points
while retaining the32 original queries, and reject rewriting the manifest.

Full real-data four-GPU/global512 `interface-auditnorm-smoke-20261018-r1`
completed steps0--2, gradient audit, DEV and checkpoint writes. Step2 loss
2.870777; its DEV was worse than the step0 score3.336509, so best remained0.
`interface-auditnorm-resume-20261018-r1` restored optimizer step2 and exact
CPU/CUDA RNG on all four ranks, then completed step4 with finite loss2.902905.
At the resumed baseline, DEV3.336770 did not replace the historical best0;
step4 DEV3.335858 improved it and correctly became best4.

The new 80k run is an exact continuation of these four corrected updates,
initialized originally from v3 step80000. It does not inherit the stopped
legacy v4 run. Supervisor2598776, GPUs0/1/2/3, global512, workers2 per rank,
same seven-day/8GB/20GiB guards. Code root:
`training-code-audit-20261018-r1`; run output:
`outputs/ptv3-shared-wam-v4/pretrain-nine-v4-auditnorm-warm80k-20261018-r1`
under the NAS root. Local small evidence:
`outputs/ptv3-shared-wam-v4/audit-28361da-evidence-20261018-r1/`.

These checks qualify the repaired implementation contracts, not overall WAM-v4
research success. The inherited normalization, source8 coverage limitation and
need for a matched robot-benefit experiment remain. Scheduler-to-WAM-to-executor
physical integration is outside this correction and is not claimed complete.

The new continuation reached step100, loss2.641316 (hand=.087060,
object=.169724, contact=.627148, positive-part CE=1.757385), finite gradient
norm2.045429. All four ranks restored optimizer step4 and exact RNG. Recent
segment=.35041 updates/s, provisional remaining63.34h; do not treat
this startup estimate as a completion guarantee. Physical GPU usage was about
8.0--8.3GiB per rank; utilization varies with CPU label preparation/prefetch.
Runtime verified: remote graspenv, Python3.8.20, Torch2.0.1+cu118, CUDA11.8.

Repository-wide `tools/verify.py --changed` still fails on six unchanged
trajectory-policy v2 cards missing required schema/seed fields. The failing
files have no diff from the pre-fix main55342ecd and were not rewritten as part
of this WAM audit correction. The experiment index check passes; the focused
implementation tests and real-data/runtime checks above are separate evidence.

### Input performance inspection (2026-10-18)

An engineering-only CPU profile used the pinned remote reader and actual
recipe, deterministic `draw(128,170,240,rank=1)`, one thread, and the same
16-sequence per-source cache as a training worker. No training process or
configuration was changed. Raw script/log are in project
`tmp/wam-v4-performance-20261018/`. Dataset initialization took8.81s
(one-time). Constructing128 windows under cProfile took6.667s and collate
.00793s. `interaction_field` cumulative5.400s (81% of sample construction),
including NumPy voxel uniqueness/sorting; array argsort self-time3.364s.
`numpy.load` cumulative.533s includes header/mapping work, so is not a pure
storage-wait measurement. These nested times must not be added together.
The probe measures input CPU costs, not total training stage percentages,
and includes cProfile overhead and a different cache history from live workers.

The live run reached220 updates at.34823 updates/s (~2.87s/update); the
80k/global512 plan implies40.96M sampled windows with replacement. Eight live
loader workers each consumed roughly96--97% of one CPU core. GPU power
was about99--118W at the snapshot while utilization read100/0/100/100;
this does not prove compute saturation because distributed waiting can also
occupy GPU kernels. Together with the input profile, repeated online C label
generation is a strong bottleneck candidate. GPU forward/backward and NCCL
wall-time shares have not yet been measured. Periodic DEV is rank0-only
(every1000 updates) while other ranks wait; the current early ETA has not
passed its first periodic validation and excludes that recurring overhead.

### Exact-label input efficiency comparison (2026-10-18)

Engineering qualification under the same experiment; no new architecture, data
selection, loss or scientific claim. Hypotheses: occupied-cell integer lookup
reduces repeated voxel sorting; workers4 may increase supply versus workers2;
removing duplicate native tensor construction and caching fixed DEV may reduce
remaining preparation. The decision is which verified implementation/worker
count to use for the unchanged80k continuation. Cheapest check first: exact
synthetic/real-reader comparisons before real GPU throughput.

Implementation `b65872f8` adds collision-free int64 keys with pre-encoding bounds
checks and exact original fallback on unsafe extents. Voxel origin, scaffold,
1cm/dilation policy and neighbour-sum arithmetic are unchanged. Clean reader
checks/provenance wrap direct v4 construction, canonical native selections are
prepared once, and fixed DEV has a256MiB CPU cache cap at the same microbatch8.
Optional CUDA events profile stages at normal log synchronization boundaries;
forward/loss includes collectives and stage/input-wait times are not additive
resource shares. TRAIN/DEV filter file digests are bound in new checkpoint
provenance; legacy parents lacking them are explicitly only bound at transition,
without claiming historical filter integrity.

Focused suites:53 passed locally and53 on remote Python3.8.20/graspenv,
Torch2.0.1+cu118. Tests include int64 extrema/overflow fallback, empty scaffold,
negative/shifted coordinates, dilation0/1/2 full targets/masks, native full
horizon and payload/index drift checks, cached DEV equality and boundedness.
The user-described external efficiency ZIP was not present locally; this is an
independent implementation of the supplied specification, not a claim that the
external263-case prototype was executed.

Remote bounded real-reader check uses deterministic72 source-balanced TRAIN
windows plus128 weighted TRAIN draws (200 total), repeats2, same16-sequence
cache policy. Every returned field, including C target/mask and native
provenance, matches the original implementation bitwise. Reference median
8.79249s; lookup-only3.83199s; full optimized3.92080s (~2.24x input construction).
The native addition is within noise here; no separate measured speed benefit
is claimed. These are CPU input measurements, not whole-training speedups.

Remote output root `tmp/efficiency-probe-20261018-r1/` under the existing NAS
artifact root; code `training-code-efficiency-20261018-r1`. Full-model four-rank
GPU comparison supervisor2620289/torchrun2620312 uses the immutable audited
step3000 checkpoint, global512,50 updates per case/10 warmup, original/optimized
inputs and workers2/4. All cases restore the same model/optimizer/per-rank RNG
and verify sample order. Benchmark updates write no checkpoints. Four RTX4090,
502GiB machine RAM and abundant free NAS space; max1200s, own process group only.

The prior corrected run was administratively stopped at logged3260; its
step3000 complete checkpoint, historical best and all logs remain. Unsaved
updates3001--3260 will be recomputed on the new continuation, rather than
pretending that a live in-memory state was preserved. This is not a reset to v3
or a change of the audited training objective.

Four-rank real-model benchmark completed with exit0 in404.56s. Each case
measures40 updates after10 warmup updates, global512, checkpoint step3000:

| Input | Workers/rank | Updates/s | Speed relative to original2 |
| --- | ---: | ---: | ---: |
| Original query/native path | 2 | .346243 | 1.00 |
| Original query/native path | 4 | .654890 | 1.89 |
| Exact lookup/direct native | 2 | .803132 | 2.32 |
| Exact lookup/direct native | 4 | 1.572836 | 4.54 |

Sample-order digests match on all ranks. The two optimized cases' complete
50-update per-rank loss sequences are bitwise identical to original-input
workers4. The first freshly constructed DDP case (original workers2) has small
numerical drift (maximum absolute loss difference.001738) from all subsequent
cases; its cause was not isolated and exact GPU trajectory equivalence to that
first case is not claimed. This does not affect the real-reader bitwise target
comparison or the identical later original/optimized loss sequences.

Mean CUDA/event/CPU stages in seconds for rank0, original2 -> optimized4:
input wait2.25359 -> .02255; H2D.00175 -> .00166; forward+loss.19992 -> .17894;
backward.40392 -> .40366; optimizer.02889 -> .02892. Across all ranks the optimized
input wait is.00059--.02805s. Original rank1's forward+loss took2.02186s,
consistent with collective waiting for slower data ranks, rather than interpreting
that CUDA duration as pure model compute. Timing covers steady updates, not
periodic validation/checkpoint or startup.

Real-model DEV at the same microbatch8 reports exactly equal objective and all
metrics with streaming/cached CPU inputs: objective2.5833346971694704. Streaming
6.04524s; cached1.71182s; cache preparation4.76752s once,44,060,416 bytes (~42MiB).
Microbatch16/32 was not substituted: the unchanged8 already removes repeated C
generation without introducing a numerical/configuration transition.

Small evidence copies: local `outputs/ptv3-shared-wam-v4/efficiency-evidence-20261018-r1/`
contains `cpu.json`, `gpu.json`, process metadata and benchmark log. Select
workers4 with prefetch2 and persistent workers; leave all training configuration,
recipe/statistics, model/loss and validation panel unchanged. Remaining77k
updates extrapolate to13.60h at benchmark speed; this is provisional until
sustained training and periodic evaluations/checkpoints are observed.

Full training entrypoint smoke `interface-fastinput-resume-20261018-r1`
(supervisor2621912) completed3000->3004 with exit0: exact CPU/CUDA RNG and
optimizer step3000 restored on all four ranks, gradient audit/DEV/checkpoint
writes passed, final loss1.877037 and finite gradient1.686120. Historical best
step3000/score2.570000832077475 remained best after the worse step3004 DEV.
New checkpoint provenance binds TRAIN filter
`57f1a5ceabebcaec1b43abad09248d5c744162f169a08ed24f984985e0fdcdc8` and DEV
`2ea724a34d4090e85217deefbf4669553cf76fdb9f006ecc921e84fe05569a26`.
Four-update smoke throughput is not used for ETA: final DEV took1.433s and
checkpoint work10.726s, dominating that short interval.

The production continuation `pretrain-nine-v4-fastinput-warm80k-20261018-r1`
launched from the resulting step3004 complete checkpoint with supervisor2622903,
GPUs0/1/2/3, workers4 per rank, global512, microbatch8 DEV cache and first100
updates instrumented. Same recipe/config/statistics/objective, full
optimizer/RNG resume (no model-only reinitialization), target80000 and original
7-day/8GB/20GiB guards. Previous runs/checkpoints/logs are retained.
Repository-wide verify still fails only on the six pre-existing trajectory-policy
v2 card schema/seed issues documented above; the experiment index and focused
WAM card checks pass.

Production continuation verified at step3100, loss1.861955, finite gradient1.755117,
recent1.57165 updates/s and provisional remaining13.59h (before recurring
validation/checkpoint overhead). Physical GPU memory about8.0--8.1GiB/rank.
All ranks restored optimizer step3004 and exact CPU/CUDA RNG; inherited filter
bindings match on this second resume. Best step3000 remains owned and retained.
