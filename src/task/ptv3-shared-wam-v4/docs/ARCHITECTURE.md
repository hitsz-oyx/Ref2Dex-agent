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
adapter binds those normals back by object identity. Native geometry with fewer
than 512 distinct samples must be rebuilt from CAD triangles into a derived
dense overlay; the reader rejects cyclic repeats and underfilled surfaces. Raw
payloads remain frozen. The accepted surface identities are 512 distinct
target points, 32 preserved flow identities, and 128 distinct interaction
queries.

The 18-D scene features now have one contract at every entry point: channels
0--2 unit normals, 3--5/6--8/9--11 H4 velocity/acceleration/displacement, 12
target=1 for the target-only view, 13 static computed from the same H4
displacement threshold (2 mm), and 14--17 cleared. Main3, native and the online
runtime share this definition; a cross-task test binds the online constant to
the data reader.

C labels use a voxel origin derived only from the current observed interaction
queries (`min(current_points)-5cm`). The future t+1 frame must not choose the
grid phase for later frames. `INTERACTION_ORIGIN_SEMANTICS` and the v4 data
identity record this change, so old checkpoints cannot silently resume on the
new labels.

Static idle and approach windows are thinned only in a derived manifest. A
window with no observed contact, a still hand and a moving object is classified
`moving_object_noncontact`, not `static_idle`; it has its own retention ratio.
Contact static and contact moving windows are always retained. The manifest
records the hash-based policy and blocked indices, so a training recipe can bind
it without silently changing the source corpus.

The first v4 implementation does not model support planes, table collision, or
other objects. Those remain executor/planner concerns until a later ablation.
