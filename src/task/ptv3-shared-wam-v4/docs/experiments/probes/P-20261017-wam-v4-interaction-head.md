---
schema: ref2dex.probe.v2
probe_id: P-20261017-wam-v4-interaction-head
experiment_id: P-20261017-wam-v4-interaction-head
date: 2026-10-17
task: ptv3-shared-wam-v4
branch: main
git_commit: a7d295c5
claim_id: C1
hypothesis_family: HF-wam-interaction-head
probe_index_in_family: 1
seed_pool: probe
seeds: [170]
decision_changed_if_positive: keep the dynamic graph interface and run a matched tau-only versus tau-plus-C executor probe
decision_changed_if_negative: remove or redesign C before any long pretraining budget
status: READY
run_id: v4-filtered-interface-and-warm80k-20261018-r1
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
interaction contract; older native geometry is cyclically repeated into the
fixed surface view without fabricating normals.

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
