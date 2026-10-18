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
