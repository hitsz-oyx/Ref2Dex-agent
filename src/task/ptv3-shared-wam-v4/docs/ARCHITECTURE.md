# WAM-v4 architecture contract

v4 keeps the audited v3 goal-conditioned trunk and its 24-step hand/scene
forecasts. The scene is now target-object-only: the PTv3 geometry path receives
512 canonical surface points with aligned unit normals, while the future object
head keeps 32 fixed flow queries.

The interaction output is an object-centric field on 128 fixed target-object
queries:

```text
target surface[512] + hand[2x11] + local goal
                 |
                 v
          shared PTv3/WAM trunk  -----> tau_hand[24], tau_scene[24]
                 |
                 +---- object query position/normal x shared hand context x time
                                      |
                                      v
                      C[B, object, 24, 128, 7]
```

The seven channels are contact confidence followed by semantic hand-part
probabilities `[palm, thumb, index, middle, ring, little]`. Query position and
normal are inputs and are not redundantly regressed. Labels are built before
collation by rasterizing the canonical 2x11 hand scaffold into a 1cm integer
voxel grid, then reading contact and part occupancy at each object query. No
KNN or pairwise distance tensor is used.

All nine sources use the same derived tensor contract. Main3 already stores a
512-point canonical surface and normals in its canonical artifacts; the v4
adapter binds those normals back by object identity. Older native geometry with
fewer than 512 samples is retained and cyclically repeated to the fixed size,
with a provenance field marking the derived view. Raw payloads remain frozen.

Static idle and approach windows are thinned only in a derived manifest. Contact
static and contact moving windows are always retained. The manifest records the
hash-based policy and blocked indices, so a training recipe can bind it without
silently changing the source corpus.

The first v4 implementation does not model support planes, table collision, or
other objects. Those remain executor/planner concerns until a later ablation.
