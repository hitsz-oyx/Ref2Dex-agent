# PTv3-Shared-WAM v4

v4 preserves the v3 shared encoder and hand/object future forecasts, then adds
an object-centric interaction field. Every one of the nine sources is adapted
to a target-only view with 512 surface points and normals, 32 future-flow
queries, and 128 fixed interaction queries.

`C` has shape `B,O,24,128,7`: contact confidence plus six semantic hand-part
probabilities. Query positions and normals are inputs. Ground truth is created
from a deterministic 1cm voxelized semantic hand scaffold; no KNN is used.

Native geometry with fewer than 512 distinct samples must be rebuilt from CAD
triangles through a derived surface overlay; the reader rejects cyclic repeats.
The accepted identities are exactly 512 distinct target points, 32 preserved
flow identities and 128 distinct interaction queries. Main3 normals are
recovered from its frozen canonical artifacts by object identity. All entry
points share the 18-D feature contract: target=1 and static from the same H4
2 mm displacement threshold. A separate audit tool writes a derived
window-filter manifest that thins idle static and approach windows while always
keeping contact windows; moving-object non-contact windows have their own class
and retention ratio. Raw data is never modified.

`tools/run/train_pretrain.py --init-v3 <checkpoint>` performs a model-only
warm start. Compatible v3 trunk parameters transfer, while the interaction
field modules initialize freshly.

The audited objective `v4.global-masked-components.v2` separately normalizes
hand, motion-weighted object, contact BCE and positive-contact part CE using
global denominators in DDP. DEV first sums each component over a whole source,
then applies the source macro average. Reports include explicit metric units;
there is no mixed seven-channel C MAE or placeholder consistency loss.

C labels use a voxel origin derived only from the current observed interaction
queries; the future t+1 frame cannot shift the grid phase of later labels. The
label semantics and input semantics are part of the checkpoint data identity, so
an old checkpoint cannot silently resume on the changed view.

Forward inputs require current geometry, hand/history, object/goal and optional
`interaction_query_valid[B,O,Q]`; future labels/masks are used only by losses
and metrics. Invalid queries are cleaned before the query MLP; a NaN in a valid
query fails closed. Non-unit `interaction_scales` are rejected because C is a
probability head. Output `interaction_prediction_valid` is current query/object
validity. C and trajectories are parallel predictions of shared current
context: C does not condition on a proposed trajectory and no explicit
trajectory--C consistency objective is implemented.

Dense overlays use per-NPZ SHA-256 digests and a manifest digest bound into the
recipe and every checkpoint. Existing arrays can be bound with
`3d-data-preparation/tools/run/freeze_v4_surface_recipe.py` without resampling.
Normal resume requires the same objective/evaluation protocol and data identity;
historical best score/step/checkpoint is inherited, and the initial baseline
participates in selection. `--resume-reviewed-subset` is unsupported and fails
before loading data. Historical checkpoints remain readable for audit, but
legacy objective checkpoints cannot silently resume the corrected objective.

Online WAM-v4 integration lives in `wam-rl-policy`:
`wam_rl_policy.wam_v4_runtime` builds the 512/32/128 batch from the observed H4
history, a scheduler SE(3) `goal_delta` or a position-only task goal, and a
dense overlay; it exposes the `encode()` summary and the full `tau`/`C`
`forward()` path separately. `wam_rl_policy.tau_contract` converts v4's
30 Hz/24-step anchor-frame hand trajectory to the existing 60 Hz/8-step student
tau (`tau(0)=0`, interpolation, right/left slot, optional world rotation). The
summary path is explicitly `encode`-only and cannot be used as evidence that
tau/C reached the executor.

The exact integer voxel lookup sorts occupied scaffold cells only. Bounds and
int64 overflow checks preserve the original lookup as a fallback; voxel origin,
hand semantics, neighbour order and label arithmetic are unchanged. Native
windows construct the v4 view directly through the cleaned reader's frozen
checks/provenance and reuse prepared canonical geometry. Fixed DEV inputs are
cached in CPU memory (256MiB cap) at the configured microbatch size; use
`--no-dev-cache` to compare the streaming path. TRAIN and DEV filter file digests
are recorded in provenance and checked on future resumes. Older checkpoints
without those digests are bound at the first continuation; this cannot prove
their historical filter content.

`--profile-steps N` (0--2000, default0) records input wait and CUDA event times
for transfer, forward+loss, backward and optimizer at normal logging boundaries.
There is no extra per-stage synchronization. CUDA stage times include stream
waiting and collectives; input wait can overlap device work and the numbers must
not be treated as additive CPU/GPU utilization percentages. Periodic validation
and checkpoint wall time are also logged. The bounded engineering tool
`tools/audit/benchmark_input_pipeline.py` compares full real-reader fields against
the original implementation, or uses the same checkpoint/RNG/sample sequence
to compare reference/optimized input and workers2/4 with real CUDA/DDP updates.
Benchmark updates write no checkpoints and do not enter research model selection.
