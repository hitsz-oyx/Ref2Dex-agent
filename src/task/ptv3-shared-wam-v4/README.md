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
Original 32 flow identities are preserved. Main3 normals are recovered from its
frozen canonical artifacts by object identity. A separate audit tool writes a derived
window-filter manifest that thins idle static and approach windows while always
keeping contact windows; raw data is never modified.

`tools/run/train_pretrain.py --init-v3 <checkpoint>` performs a model-only
warm start. Compatible v3 trunk parameters transfer, while the interaction
field modules initialize freshly.

The audited objective `v4.global-masked-components.v2` separately normalizes
hand, motion-weighted object, contact BCE and positive-contact part CE using
global denominators in DDP. DEV first sums each component over a whole source,
then applies the source macro average. Reports include explicit metric units;
there is no mixed seven-channel C MAE or placeholder consistency loss.

Forward inputs require current geometry, hand/history, object/goal and optional
`interaction_query_valid[B,O,Q]`; future labels/masks are used only by losses
and metrics. Output `interaction_prediction_valid` is current query/object
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
