# PTv3-Shared-WAM v4

v4 keeps the audited v3 input, geometry path, 24-step hand/scene forecasts and
`G_task` conditioning, and adds an explicit interaction head. The intended
planning chain is:

```text
G_task -> local 24-step goal/scheduler -> WAM -> (tau, C) -> executor
```

The new `C` output is a structured object-centric field with shape
`B,O,3,11,9` (three anchors at steps 8/16/24, eleven canonical hand points,
nine channels):

1. soft contact confidence;
2. approximate object-surface position in metres (3);
3. approximate surface normal (3);
4. normal approach velocity in metres/second;
5. tangential slip speed in metres/second.

`C` is derived from the existing hand and object future labels by
`interaction_target`. The first probe uses the existing 32 object query points
and radial normals as an explicit approximation; it is not dense-surface
supervision. The prediction is returned in physical units (`interaction`) and
fixed-scale units (`interaction_normalized`). Invalid object/keypoint slots are
masked to zero. A later dense-surface data view can keep the same head contract.

The training entry point accepts `--init-v3 <checkpoint>` for a model-only
warm-start. It transfers all compatible v3 weights and initializes the new C
relation modules afresh; v3 optimizer state and RNG are not resumed.

This is an architecture/probe implementation. It does not claim robot benefit
until a matched `tau-only` versus `tau+C` executor experiment is run. It reuses
the v3 nine-source reader, while using a new schema
`ref2dex.ptv3-shared-wam.v4`; v3 weights enter only through the explicit
model-only warm-start path above, never through an accidental full resume.
