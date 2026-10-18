---
schema: ref2dex.probe.v2
probe_id: P-20261017-wam-v4-interaction-head
experiment_id: P-20261017-wam-v4-interaction-head
date: 2026-10-17
task: ptv3-shared-wam-v4
branch: main
git_commit: 7b6d28a5
claim_id: C1
hypothesis_family: HF-wam-interaction-head
probe_index_in_family: 1
seed_pool: probe
seeds: [170]
decision_changed_if_positive: keep the dynamic graph interface and run a matched tau-only versus tau-plus-C executor probe
decision_changed_if_negative: remove or redesign C before any long pretraining budget
status: RUNNING
run_id: pretrain-nine-v4-auditnorm-warm80k-20261018-r1
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
