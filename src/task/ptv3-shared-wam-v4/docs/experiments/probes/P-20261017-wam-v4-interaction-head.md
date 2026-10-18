---
schema: ref2dex.probe.v2
probe_id: P-20261017-wam-v4-interaction-head
experiment_id: P-20261017-wam-v4-interaction-head
date: 2026-10-17
task: ptv3-shared-wam-v4
branch: main
git_commit: 992962ec
claim_id: C1
hypothesis_family: HF-wam-interaction-head
probe_index_in_family: 1
seed_pool: probe
seeds: [170]
decision_changed_if_positive: keep the dynamic graph interface and run a matched tau-only versus tau-plus-C executor probe
decision_changed_if_negative: remove or redesign C before any long pretraining budget
status: RUNNING
run_id: pretrain-nine-v4-dense512-cfix-warm80k-20261018-r1
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

Limitations: this remains an offline Probe. Normalization is inherited rather
than refitted after thinning; native geometry resolution is now repaired, but
the native source8 snapshot is still the earlier subset. C metric values are
dimensionless despite the inherited evaluator's `mean_mm` serialization key;
only hand/object displacement metrics use millimetres. No robot improvement
or matched-v3 representation improvement is established.
